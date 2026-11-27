# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 / sm_10x / default.

The loader retains FP8 or BF16 PLE storage and supports FP8 block-scaled or native
NVFP4 draft experts. Native NVFP4 experts use their checkpoint scalars and local window.

One target for the layouts `routing.py` admits: one rank (tp1, every head and
expert); at four ranks tp4 (attention TP4, routed experts split by intermediate), tep4
(attention TP4, experts EP4) and dep4 (attention DP, experts EP4); at two ranks dep2 (attention
DP, experts EP2), the layout of both disaggregated-serving engines, which is dep4 at half the
group, tp2 (attention TP2, every expert split by intermediate) and tep2 (attention TP2, experts
EP2); across nodes dep8. The core holds the static structure (this rank's share of the
weights, derived operands, per-manager pool views); `PrefillTarget` (any batch with context rows)
and `DecodeTarget` (generation only) run a step over it, passing every argument to every op at
the call site.

The forward is cut per rank; what the layout settles is settled once, at construction:

* **Attention and Gated-DeltaNet heads** are this rank's share (`weights.rank_dims`,
  `pool.pool_geometry`): at attention TP4 six q heads, one of the two KV heads, four GDN key
  and twelve value heads. The fused attention input projection row is
  `[q | k | v | gate | index q, k | v slot]`, 4352 wide; the QSA indexer is replicated.
* **The mixer output is a partial sum** under attention TP: o_proj and out_proj read this
  rank's heads. One plain sum (`comm/allreduce`, op `NONE`) completes it before the
  hyper-connection inject, which every rank runs on the same reduced rows. Under attention DP
  a rank holds every head and serves its own requests, so there is nothing to sum.
* **Routed experts** run on the trtllm-gen NVFP4 runner entered with the router logits
  (`fp4_block_scale_moe_renorm_routed`: its own Renormalize routing over the whole routing
  space). A rank holds an EP window of
  `num_experts / moe_ep_size` whole experts (tep4, dep4), or every expert at its quarter of the
  intermediate (tp4: 160 of 640 rows, stored zero-padded to 256, `weights.EXPERT_INTER_ALIGN`,
  with zero codes and scales, so the padding adds nothing). Under attention TP every rank holds
  every row and the partial output -- the window's, or the intermediate slice's -- is summed the
  same way. Under attention DP each rank holds its own rows: they are padded to the group's
  largest row count, gathered (`comm/allgather`), routed and run through the window, and
  reduce-scattered back (`comm/reducescatter`), each row's window partials summed once
  (`_expert_sum_op`); at dep4 and dep8 `DP_MOE_COMM_ENV=lamport|lamport_step` runs the pair as
  one Lamport launch each (`ops/moe/dep_lamport_round_trip`). A captured decode step routes the
  gathered rows, builds the window's pre-routed tables and quantizes them in one launch and enters the runner pre-routed
  (`DP_ROUTE_FRONT_ENV`, `ops/moe/moe_route_front__decode`). The router and the bf16 shared
  expert are replicated and run on this rank's rows, so the block output
  `routed + sigmoid(gate) * shared` is formed by the next hyper-connection site on the reduced
  routed rows.
* **Hyper-connections, PLE, the embedding** run replicated on every rank's rows;
  `lm_head` is the shell's (vocab-parallel under attention TP, gathered by the shell's head;
  whole under attention DP, over this rank's own rows).

Each hyper-connection site is `qwen4_exp_hc_inject_norm` (the previous block's injection
fused with this site's grouped norm; after a MoE block it also forms that block's output)
and then the mixer: the first projection with its activations (one tcgen05 launch,
`qwen4_exp_hc_decode_tc_first`, up to `_HC_TC_FIRST_MAX_ROWS` rows; past them `cublas_mm`,
fp32 out, then `qwen4_exp_hc_lowrank_act`) and `qwen4_exp_hc_up_gate_mean`. The MoE site's
routed input is that mixer output quantized to NVFP4 with linear block scales at the layer's
FC1 input global scale: in the mixer's own epilogue under attention TP
(`qwen4_exp_hc_up_gate_mean_fp4`), on the gathered rows under attention DP (`fp4_quantize`).
There is **no** `model.language_model.norm.weight`: the stack-final mixer is the final norm.

**One rank (tp1).** With nothing to sum or exchange, a rank holding the whole model takes forms
of its own, settled at construction (`one_rank`): the decode projections' one-rank dispatch
(`_one_rank_projection`); the hyper-connection sites' row-count forms (`hc_fast_sites`: the
decode GEMV and tcgen05 rows pairs, the prefill rstd sites), which the other layouts leave
out; a trtllm-gen
decode MoE whose finalize the next site folds in (`moe_fold`); the CuteDSL copy of the
routed experts for prefill steps past `_CUTE_DSL_MOE_MIN_ROWS` (`moe_prefill_copy`), or the
CuteDSL pipeline for every step (`moe_config.backend: CUTEDSL`, `_cute_dsl_routed`); an fp32
recurrent state, whose captured single-token decode runs `gdn_decode_gated_norm_pdl`; the
decode draft layer and its FP8 head; the family's row-tile pass for the NVFP4 pool's context
rows (`_sparse_gqa_nvfp4_context`); and the opt-in FP8 / NVFP4 copies of the dense
projections and heads (`_ONE_RANK_SWITCHES`: `_DENSE_FP8_ENV`, `_GDN_INPROJ_NVFP4_ENV`,
`_TARGET_HEAD_NVFP4_ENV`, `_MTP_HEAD_NVFP4_ENV`, `_MTP_MOE_NVFP4_ENV`, `_MTP_DENSE_FP8_ENV`).

**tep4.** A rank of attention TP4 with 128-expert windows takes some of those forms at its shard
for a captured generation-only step (`tep_fast`, `TEP_FAST_DECODE_ENV`, on by default): the
decode hyper-connection sites' rows pairs (`hc_decode_pairs`), the CuTe split-K sparse GQA over
its e4m3 KV head, and the MoE route front over its own rows. With `TEP_FINAR_ENV` (opt-in) such a
MoE block's finalize and TP all-reduce are one Lamport launch (`ops/moe/tep_finalize_allreduce`),
and with `TEP_FINAR_SHARED_ENV` (opt-in) that launch also adds the rank's quarter of the shared
expert at `_TEP_FINAR_SHARED_ROWS` rows;
with `TEP_MOE_FRONT_ENV` (opt-in) its router GEMM runs in the front's launch
(`ops/moe/moe_decode_front__decode`). With `TEP_INDEX_PREFETCH_ENV` (opt-in) a verify step's
trunk index scores at 1-2 requests read their keys before the PDL wait on the attention decode
prologue. With `TEP_GQA_E4M3_ENV` (opt-in) the decode GQA of the trunk and the draft layer runs
on the half-height e4m3 entry (`qsa_paged_sparse_gqa_e4m3_decode`) instead of the split-K one.
With `TEP_DRAFT_LAYER_ENV` (opt-in) a generation-only
draft step runs the draft layer on the one-rank decode form at the rank's shard
(`DecodeTarget._mtp_layer`), a draft-extend step's continuing rows included.

**QSA sparse attention.** The index plane is maintained on every step so a request can
cross `dense_seq_len_threshold` later without rebuilding its prefix; above it the sparse
core (index scores, block top-k and expansion, sparse GQA) replaces the dense op, and the
main K/V append becomes this layer's own call. The raw index key is stored before anything
normalizes it: the compression applies the norm and the rotation itself. Past one rank a
generation step's sparse core runs on the family's decode ops, which the FP8 target runs too,
at its row table: request-grouped index scores, the cluster top-k, and the CuTe sparse GQA over
a bf16 or e4m3 pool with the output gate in its store (`_decode_index_scores`, `_selection`,
`_sparse_gqa_decode`).

**KV storage.** The paged K/V pool holds the checkpoint's bf16, fp8-e4m3 under
`kv_cache_config.dtype: fp8` (`QM_FP8_KV_CACHE`) or NVFP4 under `nvfp4` (`QM_NVFP4_KV_CACHE`,
e2m1 pairs with e4m3 block scales in a second pool, second-level scales 1.0). The NVFP4
attention kernels exist for the paged-context path only. The QSA index and position planes
keep their own types under all three.

**Breakable prefill CUDA graph.** With `prefill_cuda_graph_backend: breakable` the engine
captures the core's `forward` (the body, not the shell) for a step with context rows, pads
the step up to the nearest bucket and replays it without calling the body. The shell
therefore calls `begin_step` before the body on every step: the asserts, the cache-manager
bind, the target choice and the prefill target's host bookkeeping all live there. The
prefill forward is row-wise over the `T` rows it is handed, with four breaks
(`catalog/graph/eager_on_graph`) that read the step: the vision splice, the PLE block, and
each layer's attention core and Gated-DeltaNet recurrence. Each writes rows `[:n]` of a fixed
output and zeroes rows `[n:]`.

**Host offload.** By default the PLE n-gram table lives in host memory the device addresses
(`weights._host_resident_empty`), one copy per rank; `TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD=0`
makes it device-resident. Output is identical.

**One-model MTP-Eagle.** With a `speculative_config` the core also holds the checkpoint's MTP
module (`MTPLayer`), replayed `max_draft_len` times by the engine's spec worker, and the trunk
runs every generation request's `max_draft_len + 1` verify rows:

* The draft layer is one more full-attention QSA layer, cut like a trunk attention layer (this
  rank's q heads, its KV head, the replicated indexer; every head under attention DP), over the
  manager the draft step is handed: under attention TP that is a separate draft KV cache, whose
  K/V, index and position pages hold the draft layer alone; under attention DP the engine
  builds none and the trunk's manager maps the draft layer too (`_bind_draft_cache_manager`).
  Its front end, router, shared expert and hyper-connections are replicated; its FP8
  block-scale experts run as this rank's window of `num_experts / tp_size` experts
  (`fp8_block_scale_moe_runner`'s window). Under attention TP the o_proj and expert partials
  are completed by the same plain sum as the trunk's. That window is the same at tp4 as at
  tep4: the FP8 experts' 128x128 weight blocks do not take the trunk's 160-row intermediate
  split, and a sum of whole-expert windows is the layer's output whichever way the trunk's
  experts are cut.
* Under attention DP each rank drafts for its own requests, and the draft MoE runs the trunk's
  round trip (`_expert_sum_op`): the group's rows gathered, routed and run through this rank's
  window, reduce-scattered back. The spec worker hands each draft step the group's row counts
  (the trunk step's tokens on the first, one row a request on the rest, attention-DP dummy
  requests included) without writing them into the metadata; `_dp_pad_rows` pads this rank's
  rows by them, so every rank issues the same collectives at every draft step.
* A generation-only draft-extend step past one rank runs the attention core and the tail on
  each request's continuing row only (`_draft_continue`); every row still writes its K/V and
  index key. Under attention DP it does so only when the group's counts show every rank
  generation-only, and its MoE round trip then pads by the group's request counts.
* A verify step's Gated-DeltaNet rows take the multi-token conv with per-step window snapshots
  and the stock Triton cached replay over the bf16 pool, which commits a full history in the
  kernel below `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests and leaves it to the
  manager's all-layer commit from there on, exactly as the stock layer hands it work items.
* What a verify row writes that the runtime does not roll back -- the PLE n-gram history and
  conv state, every attention layer's QSA index-K plane and the shared position plane -- is
  staged during the forward and rewritten to the accepted prefix by `SpeculativeRollback`,
  which the spec worker calls after its draft loop.
* IndexShare (`index_share_for_mtp_iteration`) runs the draft indexer once per draft loop:
  the draft-extend step captures each request's selection and the later draft steps rebuild
  theirs from it (`qsa_mtp_reuse_topk`).

Not assembled here: the legacy per-step recurrent verify (`TRTLLM_USE_GDN_REPLAY=0`), which
routing declines and the bind refuses.

Integration facts this file depends on:

1. The engine rebuilds the composite config and normalizes its text tower
   (`config_utils.load_pretrained_config`). The shell adopts that text config as
   `model_config.pretrained_config` before building the core, because the recurrent-state and
   PLE pool geometry are read from it.
2. `position_ids` arrives in the mRoPE layout (`[3, 1, T]`) even for text-only batches. The
   main rope reads all three rows; the QSA index plane takes coordinate 0 only.
"""

from __future__ import annotations

import functools
import math
import os
import re
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import (
    NoTarget,
    engine_kv_cache_quant_algo,
    step_contract_enabled,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    thop_attention,
)
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.allreduce import allreduce, allreduce_group_args
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.graph.eager_on_graph import eager_on_graph
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    MAX_CERTIFIED_TOKENS as _MOE_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp import archs
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.activation.fused_sigmoid_mul_inplace import (
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.activation.silu_and_mul_rows import (
    silu_and_mul_rows,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.draft_continue_rows__decode import (  # noqa: E501
    draft_continue_select,
    draft_rows_gather,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (
    qsa_canonical_selection,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_cold_index_codec import (
    qsa_cold_index_mark,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_expand_block_indices import (
    qsa_expand_block_indices,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_index_scores_fp8 import (
    _takes_ctx_tile,
    index_anchors_to_fp8,
    qsa_index_scores_fp8,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_index_scores_groups_cute__decode import (  # noqa: E501
    index_scores_groups as index_scores_groups_cute,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_index_scores_groups_cute__decode import (  # noqa: E501
    serves as _index_groups_cute_serves,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_mtp_reuse_topk import (
    qsa_mtp_reuse_topk,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_nvfp4_pages_to_bf16 import (
    qsa_nvfp4_pages_to_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (
    qsa_paged_index_scores,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_store import (
    qsa_paged_index_store,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_kv_store import (
    qsa_paged_kv_store,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_kv_store_nvfp4 import (
    qsa_paged_kv_store_nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_decode import (
    qsa_paged_sparse_gqa_decode,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_decode import (
    supports as _sparse_decode_supports,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_e4m3_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_e4m3_decode,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_e4m3_decode import (  # noqa: E501
    supports as _e4m3_gqa_supports,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_fp8_verify import (  # noqa: E501
    MIN_GROUPS as _BITMAP_OP_MIN_GROUPS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_fp8_verify import (  # noqa: E501
    fp8_sparse_gqa_verify_grouped,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_fp8_verify import (  # noqa: E501
    takes_grouped as _bitmap_takes_grouped,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4 import (
    qsa_paged_sparse_gqa_nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_decode,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_verify import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_verify,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_verify import (  # noqa: E501
    takes_grouped as _verify_takes_grouped,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_prefill_compress import (
    qsa_prefill_compress,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rollback_planes import (
    plane_table as _plane_table,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rollback_planes import (
    qsa_rollback_gather,
    qsa_rollback_restore,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_grouped_cute__decode import (  # noqa: E501
    SMALL_MIN_GROUPS as _GROUPED_SMALL_MIN_GROUPS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_grouped_cute__decode import (  # noqa: E501
    qsa_sparse_gqa_grouped_cute,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_rowtile import (
    qsa_sparse_gqa_rowtile as one_rank_sparse_gqa_rowtile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_splitk_cute__decode import (  # noqa: E501
    HEAD_DIM as _SPLITK_HEAD_DIM,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_splitk_cute__decode import (  # noqa: E501
    TILE as _SPLITK_TILE,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_splitk_cute__decode import (  # noqa: E501
    qsa_sparse_gqa_splitk_cute,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_spec_step_metadata__decode import (  # noqa: E501
    SpecStepMetadataPrologue,
    take_step_positions,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    MAX_ROWS as _TOPK_CLUSTER_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    PAIR_MAX_ROWS as _TOPK_CLUSTER_PAIR_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    qsa_topk_expand_cluster,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    supports as _topk_expand_cluster_supports,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    qsa_topk_expand_decode,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    supports as _topk_expand_supports,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    takes_context as _topk_takes_context,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_select import (
    qsa_topk_select,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_attn_decode_prologue import (  # noqa: E501
    SCORE_ROWS as _PROLOGUE_SCORE_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_attn_decode_prologue import (  # noqa: E501
    qwen4_exp_attn_decode_prologue as one_rank_attn_decode_prologue,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_qk_norm_rope import (
    qwen4_exp_qk_norm_rope,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.spec_rollback_commit import (
    spec_rollback_commit,
    spec_rollback_stage,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.verify_step_setup import (
    _BLOCK_S as _VERIFY_SETUP_MAX_SEQS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.verify_step_setup import (
    verify_step_setup,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.embedding.embed_rows__decode import (
    embed_rows,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import (
    MAX_ROWS as _GEMV_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import (
    TILING as _GEMV_TILING,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import bf16_gemv_pdl
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.fp8_gemv import (
    fp8_gemv_conv1d_op,
    fp8_gemv_conv1d_verify_op,
    fp8_gemv_op,
    pack_from_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.fp8_per_tensor_mm import (
    fp8_per_tensor_mm,
    fp8_per_tensor_weight,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.low_m_bf16_mm import (
    WIDE_MAX_ROWS as _LOW_M_WIDE_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.low_m_bf16_mm import low_m_bf16_mm
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    CONV_ROWS as _MID_CONV_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    INJECT_UNIT_COLS as _INJECT_UNIT_COLS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    MAX_ROWS as _MID_ROWS_MAX,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    MidWeight,
    conv_ready,
    mid_mm_conv_bf16,
    mid_mm_inject,
    mid_mm_raw,
    mid_rows_mm_op,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import Plan as MidPlan
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    precompile as mid_rows_precompile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    nvfp4_gemv_op,
    nvfp4_split_gemv_conv1d_op,
    nvfp4_split_gemv_conv1d_verify_op,
    nvfp4_split_gemv_op,
    precompile_split,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    pack_from_bf16 as nvfp4_pack_from_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    pack_split_from_bf16 as nvfp4_pack_split_from_bf16,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    precompile as nvfp4_gemv_precompile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_first_cute import (
    qwen4_exp_hc_decode_first_cute,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_mix_fp4_cute import (
    qwen4_exp_hc_decode_mix_cute,
    qwen4_exp_hc_decode_mix_fp4_cute,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_rows_tc import (
    MAX_ROWS as _HC_DECODE_ROWS_MAX,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_rows_tc import (
    qwen4_exp_hc_decode_rows_first,
    qwen4_exp_hc_decode_rows_mix,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_first_rescale_tc import (
    qwen4_exp_hc_first_rescale_tc,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_rstd import (
    qwen4_exp_hc_inject_rstd,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_lowrank_act import (
    qwen4_exp_hc_lowrank_act,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_up_gate_mean import (
    precompile as hc_up_precompile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_up_gate_mean import (
    qwen4_exp_hc_up_gate_mean,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_up_gate_mean_rstd import (
    qwen4_exp_hc_up_gate_mean_rstd,
    qwen4_exp_hc_up_gate_mean_rstd_fp4,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_post_inject import (
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
    dep_lamport_rs_inject as _rs_inject_op,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
    moe_decode_front__decode as _decode_front,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
    tep_finalize_allreduce as _TEP_FINAR,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import (  # noqa: E501
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.cute_dsl_nvfp4_grouped_gemm import (
    cute_dsl_nvfp4_grouped_gemm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_pack_gather import (
    PackGather as _PackGather,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_pack_gather import (
    transported_codes as dep_lamport_transported_codes,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    LamportSteps,
    dep_lamport_round_trip,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    _sanitized as dep_lamport_sanitized,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    buffers_from_pointers as dep_lamport_buffers_from_pointers,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    build as dep_lamport_build,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    build_on as dep_lamport_build_on,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.dep_lamport_round_trip import (
    workspace_bytes as dep_lamport_workspace_bytes,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_permuted import (
    fp4_block_scale_moe_renorm_permuted,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (
    DECODE_TACTICS as _MOE_DECODE_TACTICS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (
    ROUTING_RENORMALIZE,
    fp4_block_scale_moe_renorm_routed,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (
    _runner as _moe_runner,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_moe_renorm_cute_dsl import (
    fp4_moe_renorm_cute_dsl,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp8_block_scale_moe_runner import (
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp8_block_scale_moe_runner import (
    valid_tactics as fp8_moe_valid_tactics,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fused_sigmoid_gate_mul_add import (
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    LEAN_MAX_ROWS as _ROUTE_FRONT_LEAN_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    MAX_ROWS as _ROUTE_FRONT_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    NUM_EXPERTS as _ROUTE_FRONT_EXPERTS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    TOP_K as _ROUTE_FRONT_TOP_K,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    check_tables as _route_front_check_tables,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    compile_front as _route_front_compile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    moe_route_front,
    table_sizes,
    table_views,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    route_front_only as _route_front_only,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_route_front__decode import (
    routed_rows as _routed_rows,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_sort import moe_sort
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.moe_unpermute_inplace import (
    moe_unpermute_inplace,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.mtp_moe_combine_inject__decode import (  # noqa: E501
    mtp_moe_combine_inject,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.tep_finalize_allreduce import (
    build as tep_finar_build,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.tep_finalize_allreduce import (
    build_tp_sum as tep_tp_sum_build,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.tep_finalize_allreduce import (
    tep_finalize_allreduce,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm.flashinfer_gemma_rmsnorm import (
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ple.ple_gate_conv_inject_norm import (
    ple_gate_conv_inject_norm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ple.ple_ngram_embed import ple_ngram_embed
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.quantization.bf16_to_fp32 import (
    bf16_to_fp32,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.quantization.fp8_quantize_1x128 import (
    fp8_quantize_1x128,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_triton_update import (
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.cu_seqlens_from_seq_lens import (
    cu_seqlens_from_seq_lens,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.flashinfer_chunk_gated_delta_rule import (
    INPLACE_MAX_TOKENS as _GDN_INPLACE_MAX_TOKENS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.flashinfer_chunk_gated_delta_rule import (
    flashinfer_chunk_gated_delta_rule,
    flashinfer_chunk_gdn_inforward_split,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_conv_lean import (
    conv_lean as gdn_conv_lean,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_conv_lean import (
    precompile as gdn_conv_lean_precompile,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_conv_lean import (
    promote_x as gdn_promote_x,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_gated_norm_pdl import (
    MAX_ROWS as _GDN_DECODE_FUSED_MAX_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_gated_norm_pdl import (
    gdn_decode_gated_norm_pdl,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_rows_gated_norm import (
    gdn_decode_rows,
    gdn_decode_rows_gated_norm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_prefill_conv_post import (
    gdn_prefill_conv_post,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import (
    FLASHINFER_REPLAY_RING_SIZE,
    flashinfer_cached_replay_update,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype
from tensorrt_llm._torch.pyexecutor.runtime_hooks import QsaRuntimePolicy, RuntimeHooks
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.vocab_parallel import (
    VocabShardedLogits,
    vocab_parallel_argmax_enabled,
)
from tensorrt_llm._torch.utils import (
    ActivationType,
    is_gdn_replay_enabled,
    last_positive_power_of_2,
)
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from .. import ple_host_table as _ple_host
from . import pool as _pool
from . import vision as _vision
from . import weights as _weights
from .ops.attention import qsa_sparse_gqa_rowtile__prefill as _rowtile_prefill
from .ops.attention.qsa_fp8_pages_to_bf16__prefill import qsa_fp8_pages_to_bf16
from .ops.attention.qsa_sparse_gqa_rowtile__prefill import qsa_sparse_gqa_rowtile
from .ops.attention.qwen4_exp_attn_decode_prologue__decode import qwen4_exp_attn_decode_prologue
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import MAX_ROWS as _HC_TC_FIRST_ROWS
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import ROWSN_ROWS as _HC_ROWSN_ROWS
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import TILE64_MAX_ROWS as _HC_TILE64_MAX_ROWS
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import hc_first_ss as hc_tc_first_ss
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import precompile as hc_tc_first_precompile
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import (
    precompile_rowsn as hc_tc_first_precompile_rowsn,
)
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import (
    qwen4_exp_hc_decode_tc_first,
    qwen4_exp_hc_decode_tc_first_ss,
)
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import ss_plan as hc_tc_first_ss_plan
from .ops.hc.qwen4_exp_hc_up_gate_mean_fp4__moe import qwen4_exp_hc_up_gate_mean_fp4
from .ops.ssm.gdn_cached_replay_cute__verify import cached_replay_norm_update as gdn_replay_norm
from .ops.ssm.gdn_cached_replay_cute__verify import gdn_cached_replay_cute
from .ops.ssm.gdn_cached_replay_cute__verify import norm_tpc_ok as gdn_replay_norm_ok
from .ops.ssm.gdn_cached_replay_cute__verify import precompile as gdn_replay_precompile

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# `moe_config.backend` request -> the routed-expert backend the trunk runs.
_TRUNK_MOE_BACKENDS = {"AUTO": "TRTLLM", "TRTLLM": "TRTLLM", "CUTEDSL": "CUTEDSL"}
# The CuteDSL grouped GEMMs' MMA tile M, which is also the tile `moe_sort` pads each expert's
# rows to: the GEMMs walk the sort's tile tables. sm_103 admits 128 and 256.
_CUTEDSL_TILE_SIZE = 128
# The longest MTP draft the CuteDSL pipeline is certified at: its cells capture up to 128 rows
# (32 requests of 4 verify rows).
_CUTEDSL_MAX_DRAFT_LEN = 3

# Rows per request of the widest verify step (`max_draft_len + 1`), and of the widest
# one without the GDN cached replay (the per-step recurrent states).
_MAX_VERIFY_WIDTH = _pool.MAX_VERIFY_WIDTH
#: Strategies of the stock `AllReduce` module's own kernels, which the raw all-reduce op has no
#: branch for: the module calls the op at AUTO where it cannot run them, and so does `_tp_sum_op`.
_MODULE_ONLY_STRATEGIES = frozenset({int(AllReduceStrategy.MNNVL), int(AllReduceStrategy.SYMM_MEM)})
#: Overrides the stock model's attention-DP MoE round trip; `check_deployment` declines any value
#: but `ALLGATHER`, the round trip this target runs.
MOE_COMM_METHOD_ENV = "TRTLLM_FORCE_COMM_METHOD"
#: The transport of the attention-DP MoE round trip's two collectives: `nccl` (the default),
#: `lamport` or `lamport_step`, one Lamport launch each (`ops/moe/dep_lamport_round_trip`) for
#: every round trip of up to `dep_lamport_round_trip.MAX_ROWS` rows a rank (more keep NCCL): the
#: same rows in the same rank order, a -0.0 lane traveling as +0.0, the partials summed in a fixed
#: chain per owning rank (NCCL's ring order within a node; at dep8 across two nodes NCCL orders its
#: sum by its own topology, so there the sum is a same-precision reorder of NCCL's).
#: - `lamport`: the slot of a launch comes from a device counter the kernel reads and advances.
#: - `lamport_step`: from the host, `(base + the launch's index in the step) mod 3`; the shell's
#:   whole forward -- trunk, heads and every draft pass -- is one region (`LamportSteps`), at whose
#:   end one 1-thread launch moves the base. Each reader re-arms the vector it took and every
#:   receive loop issues its first probe of every peer before it waits on any. The same bytes in
#:   the same slots as `lamport`: bitwise the same results, fewer device-side atomics and loads.
#: A CTA polls for its peers' CTAs, so a launch needs all of its CTAs resident at once. Certified
#: on a dep4 worker (one node, torch symmetric memory) on SM103; written for dep8 over the two
#: nodes of one rack (`distributed.peer_buffers.PeerBuffers`: MNNVL memory), which a
#: `--segment`-style allocation keeps in one NVLink domain. `check_deployment` and the build hold
#: it to those layouts. Read once, when the target is built.
DP_MOE_COMM_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_COMM"
_DP_MOE_COMMS = ("nccl", "lamport", "lamport_step")
#: The layouts the Lamport transport serves, by attention-DP group size.
_LAMPORT_LAYOUTS = {4: "dep4", 8: "dep8"}
#: Off by default (class S). At a dep4 / dep8 rank on the Lamport transport
#: (`DP_MOE_COMM_ENV`), two launches of a captured decode step's trunk MoE block change, each at
#: the gathered row counts of the CUDA-graph batch sizes inside its own range:
#: - the round trip's Lamport gather releases its PDL dependents at its start
#:   (`_DP_ROUTER_FRONT_RELEASE_ROWS`); the rows, slots and bytes are the plain gather's. The
#:   gather is no PDL launch itself, so everything before it has completed when a dependent starts.
#: - the trunk router runs on `mid_rows_mm`'s bf16 kernel (`_DP_ROUTER_FRONT_ROWS` gathered rows):
#:   a PDL launch that streams its [512, 2560] weight while the gather publishes and polls, and
#:   reads the gathered rows after its own `griddepcontrol.wait`; cuBLAS elsewhere. Its logits
#:   differ from cuBLAS's by one bf16 ulp on 0.02-0.08% of the entries (the nearer one to the
#:   exact product), which can move a near-tie of the top-10. Every rank of the group runs the
#:   kernel the gathered row count picks, so the ranks agree on every row's experts.
#: The build checks every router row count against the fp32 product (one bf16 ulp) on every rank
#: and engages on every rank or none. Read once, when the target is built; set at another
#: layout or without the Lamport transport it warns and does nothing.
DP_ROUTER_FRONT_ENV = "TRTLLM_QWEN4_EXP_DP_ROUTER_FRONT"
#: Gathered rows the bf16 router serves (3..7 requests of four rows at dep4; the kernel loses to
#: cuBLAS from 8 a rank there), and, by group size, the gathered rows whose gather releases (dep4
#: 1..24 requests a rank; dep8 1..24, where 32 and more a rank measured a loss).
_DP_ROUTER_FRONT_ROWS = (33, 112)  # capacity: fixed: dispatch thresholds
_DP_ROUTER_FRONT_RELEASE_ROWS = {4: (16, 384), 8: (32, 768)}  # capacity: fixed: thresholds


def _dp_moe_comm() -> str:
    """`DP_MOE_COMM_ENV`, lower-cased, `nccl` when unset; ValueError on another value."""
    comm = os.environ.get(DP_MOE_COMM_ENV, "nccl").strip().lower() or "nccl"
    if comm not in _DP_MOE_COMMS:
        raise ValueError(f"{DP_MOE_COMM_ENV}={comm!r}: one of {', '.join(_DP_MOE_COMMS)}")
    return comm


# Recurrent-state cache dtypes the target serves, checked at bind: at one rank the checkpoint's
# fp32 too.
_SSM_STATE_DTYPES = (torch.bfloat16,)
_ONE_RANK_SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)

# The checkpoint this target is written for, read off the normalized composite's halves
# (`check_deployment`): the text tower's (layers, hidden, experts, attention heads, linear value
# heads, vocab) and the vision tower's (depth, hidden, heads, merge, out_hidden, num_pos,
# deepstack_indexes, act). A deepstack vision sibling has a merger list this target lacks.
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

_SF_VEC = _weights.SF_VEC
#: Rows up to which the MoE router runs `low_m_bf16_mm` (FlashInfer's low-M direct
#: and split-K kernels, PDL); more rows run cuBLAS. 32 is the top of both FlashInfer kernels'
#: policy.
_ROUTER_LOW_M_MAX_ROWS = 32  # capacity: fixed: a dispatch threshold, not a buffer size
_ROUTING_METHOD_RENORM = 1
_ACT_TYPE_SWIGLU = 0
_PAD_SLOT_ID = -1
#: Whether a captured decode step's projections on `low_m_bf16_mm` launch with programmatic
#: dependent launch (their FlashInfer kernels stream the weight before waiting on the
#: activation's producer; the cuBLAS branch ignores it): `TRTLLM_ENABLE_PDL`, on by default.
_DECODE_PROJ_PDL = get_env_enable_pdl()
#: The K at which `_decode_projection` runs `bf16_gemv_pdl` up to `_GEMV_MAX_ROWS` rows: a
#: tp4/tep4 rank's o and GDN out projections.
_GEMV_TP4_K = 1536
#: Widest decode projection that `_one_rank_projection` sends to `bf16_gemv_pdl` at 2..
#: `_GEMV_MAX_ROWS` rows: the shared expert's one-column gate logit (~3 us a layer at 4 rows
#: under `low_m_bf16_mm`).
_GEMV_NARROW_N = 8  # capacity: fixed: a dispatch threshold, not a buffer size
#: Whether a verify step's conv signals its dependents at its start and the CuTe replay after it
#: launches with programmatic dependent launch: the replay then streams the checkpoint and the
#: history (which nothing in the step writes ahead of it) while the conv runs, and reads the
#: conv's output only after `griddepcontrol.wait`. `TRTLLM_ENABLE_PDL`, on by default.
_GDN_REPLAY_PDL = get_env_enable_pdl()
#: Verify rows (1..8 requests of four) at which a tep4 rank's fast forms (`TEP_FAST_DECODE_ENV`)
#: launch the verify step's conv as a programmatic dependent of the input projection that writes
#: its `x` (`causal_conv1d_triton_update(launch_with_pdl_wait=True, hoist_x=True)`): the slots,
#: taps, pool and snapshot rows are read before `griddepcontrol.wait`, every x row after it and
#: before the first store. Same kernel, values and arithmetic: bitwise. GB300 tep4, together with
#: the finalize all-reduce's early release and the draft shared expert on the side stream,
#: -0.040..-0.077 ms a step at 1-4 requests.
_TEP_CONV_WAIT_ROWS = (4, 32)  # capacity: fixed: a dispatch range
_SOFTPLUS_THRESHOLD = 20.0
_L2_NORM_EPS = 1e-6
_PLE_CLAMP_MIN = 1e-6
#: Row counts up to which a hyper-connection mixer runs its first half (projection and
#: activations) as the one tcgen05 launch `qwen4_exp_hc_decode_tc_first` instead of cuBLAS
#: split-K (+ its reduce) + `qwen4_exp_hc_lowrank_act`: the entry's MAX_ROWS (two 128-row
#: MMA tiles). Every decode batch and MTP verify step a member captures is inside it
#: (the HC is replicated, so the rows are this rank's tokens); larger steps keep cuBLAS.
_HC_TC_FIRST_MAX_ROWS = _HC_TC_FIRST_ROWS
#: At one rank (`hc_fast_sites`), decode rows up to which a hyper-connection site runs as
#: `qwen4_exp_hc_decode_first_cute` + `qwen4_exp_hc_decode_mix_cute` (or its NVFP4 variant at an
#: MoE site): two GEMV launches, which win at one row and lose from two (GB300 micro-bench). From
#: there up to `_HC_DECODE_ROWS_MAX` the tcgen05 pair `qwen4_exp_hc_decode_rows_first` +
#: `qwen4_exp_hc_decode_rows_mix`; larger steps, and every step at the other layouts,
#: `qwen4_exp_hc_inject_norm` and the mixer. The two fast pairs form the mixer's low-rank
#: activations within 1-2 bf16 ulp of that path, not bitwise; the other layouts do not take
#: them.
_HC_DECODE_GEMV_MAX_ROWS = 1  # capacity: fixed: dispatch threshold, not a buffer bound
#: Decode rows from which a folded MoE block (`DecodeTarget._moe`) launches its router before
#: the shared expert: launched after the side stream's gate|up GEMM its CTAs waited for SMs
#: behind that GEMM's. GB300 micro-bench (24 layers of cold weights in one graph), us a block,
#: shared expert first vs router first: 2 rows 21.7 vs 22.4, 8 rows 46.5 vs 45.7, 32 rows
#: 112.9 vs 112.1, 128 rows 206.7 vs 204.4, 256 rows 275.6 vs 271.7.
_ROUTER_FIRST_MIN_ROWS = 8  # capacity: fixed: a dispatch threshold, not a buffer size
#: At one rank (`hc_fast_sites`), prefill steps of at least this many rows run each trunk
#: hyper-connection site on the un-normalized residual: `qwen4_exp_hc_inject_rstd` stores
#: `hc_out` and a `[T, C]` rstd instead of the `[T, C*H]` normed stream, and the rstd mixer
#: (`_hc_mixer_rstd`, `_hc_moe_mixer_rstd`) forms `normed` on chip. Smaller steps, and every step
#: at the other layouts, keep `qwen4_exp_hc_inject_norm` and the stored-normed mixer, which the
#: rstd mixer does not match bitwise (a generation row in a mixed step would round differently
#: from the same row in a decode step). GB300 micro-bench, us a site
#: (stored normed vs rstd): 2048 rows 57.0 vs 56.7, 3072 rows 82.4 vs 76.1, 4096 rows 104.0
#: vs 90.8, 8192 rows 186.2 vs 164.9, 16384 rows 360.1 vs 293.7.
_HC_RSTD_MIN_ROWS = 3072  # capacity: fixed: a dispatch threshold, not a buffer size
#: Decode rows up to which a bf16 pool's gated sparse GQA runs the one-launch decode kernel
#: (`qsa_paged_sparse_gqa_decode`) at one rank, and the index scores prefetch their block table.
_SPARSE_DECODE_MAX_ROWS = 64  # capacity: fixed: a dispatch threshold
#: The other layouts' sparse generation steps (trunk verify and draft rows) score, select and
#: attend on the family's decode ops at the FP8 target's row table (`_decode_index_scores`,
#: `_selection`, `_sparse_gqa_decode`):
#: - Whether their index scoring launches with PDL (its kernels wait before their first read);
#:   off with TRTLLM_ENABLE_PDL=0.
_INDEX_SCORES_PDL = get_env_enable_pdl()
#: - Rows from which a step with several rows a request scores over request groups
#:   (`qsa_paged_index_scores`' `rows_per_request`: one key gather serves a request's rows,
#:   bitwise its row-wise kernel); smaller steps keep the entry's own choice.
_INDEX_GROUP_MIN_ROWS = 17  # capacity: fixed: a dispatch threshold, not a buffer size
#: - At a tep4 rank under `TEP_FAST_DECODE_ENV`, further row counts that score over request
#:   groups: 2 requests of four verify rows, where the grouped kernel (2 tiles a program at 2
#:   groups) beat the visible-split one (GB300 tep4; 0 of 209M scores differ between them).
#:   4 rows keep one request's own kernel; 12 and 16 the visible split.
_TEP_INDEX_GROUP_ROWS = frozenset({8})  # capacity: fixed: a dispatch table
#: Off by default (opt-in: reads the index plane before the PDL wait, which is only right as long
#: as its predecessor writes no other index slot than the ones read again after it). At a tep4
#: rank with its fast forms, a captured verify step's trunk index scores of
#: `_TEP_INDEX_PREFETCH_ROWS` rows, launched behind the attention decode prologue, run
#: `qsa_paged_index_scores(prefetch=True, rows_per_request=p)`: the request-group kernel with each
#: request's pages and keys read before `griddepcontrol.wait` and the anchors from the request's
#: smallest position minus one compressed block (all the prologue's index store and compression
#: can write) read again after it. Bitwise the request-group kernel's scores, which are the
#: visible-split kernel's at these rows (0 of 209M scores differ). Read once, when the
#: target is built; set where it cannot apply it warns and does nothing.
TEP_INDEX_PREFETCH_ENV = "TRTLLM_QWEN4_EXP_TEP_INDEX_PREFETCH"
#: Rows that take it: 1 and 2 requests of four verify rows (GB300 tep4: -0.020 / -0.022 ms a step
#: at 1 / 2 requests; at 12 and 16 rows the scores are bandwidth-bound and the prefetching
#: program, one tile, cost +0.8..+2.8 us a layer in the op micro).
_TEP_INDEX_PREFETCH_ROWS = frozenset({4, 8})  # capacity: fixed: a dispatch table
#: Off by default (opt-in: a launch shape). At a tep4 rank with its fast forms, a generation step's
#: index scores on the visible-split kernel over more than one row tile (the rows of `p` > 1 a
#: request outside `_TEP_INDEX_PREFETCH_ROWS` and the grouped rows: the trunk's 3 and 4 requests of
#: four verify rows, and a draft-extend pass's under IndexShare; one row tile keeps the default
#: launch, which was not measured against it) launch one request slot a row tile instead of
#: `rows_per_tile` (a row tile is one request's rows, so the other slots only exit) and `<n>`
#: programs an SM instead of 16, which the split then spends on the requests that exist. The same
#: tiles and dots: bitwise the default launch. GB300 tep4 (one-node aggregated server, forced
#: acceptance 1.32, ~100k-token contexts, alternating on / off arms): n = 16, -0.009 / -0.016 ms a
#: step at 3 / 4 requests, 0 at 1-2 (single runs: n = 8 -0.007 / -0.010, n = 32 -0.008 / 0.000).
#: `TRTLLM_QWEN4_EXP_TEP_INDEX_SPLIT=<n>`, n in 1..64. Read once, when the target is built; set
#: where it cannot apply it warns and does nothing.
TEP_INDEX_SPLIT_ENV = "TRTLLM_QWEN4_EXP_TEP_INDEX_SPLIT"
#: The visible-split kernel's dot width (`qsa_paged_index_scores._SPLIT_DOT_WIDTH`): its row tile
#: is `dot width / index heads` rows.
_INDEX_SPLIT_DOT_WIDTH = 16  # capacity: fixed: the kernel's dot width


def _tep_index_split(core) -> int:
    """`TEP_INDEX_SPLIT_ENV` at this core: the visible-split programs an SM (0 = off)."""
    raw = os.environ.get(TEP_INDEX_SPLIT_ENV, "").strip()
    if not raw or raw == "0":
        return 0
    if not raw.isdigit() or not 1 <= int(raw) <= 64:
        raise ValueError(f"{TEP_INDEX_SPLIT_ENV}={raw!r}: expected programs an SM in 1..64")
    if core.tep_fast:
        return int(raw)
    logger.warning(
        f"{TEP_INDEX_SPLIT_ENV} is set but ignored: it serves a tep4 rank with "
        f"{TEP_FAST_DECODE_ENV} on"
    )
    return 0


#: - Request groups from which grouped scoring takes the CuTe kernel
#:   (`qsa_index_scores_groups_cute__decode`, 2 or 3 rows a request). It sums in another order
#:   (~7e-8 relative), so a row's top-k can differ only at a near-tie.
_INDEX_SCORES_CUTE_MIN_GROUPS = 8  # capacity: fixed: a dispatch threshold
#: - Requests from which rows over an e4m3 pool take `qsa_sparse_gqa_grouped_cute` (each selected
#:   K/V token loaded and widened once for a request's rows, over the union of their selections)
#:   instead of the per-row `qsa_sparse_gqa_splitk_cute`. Both run within the decode op's band of
#:   the fp32 reference, not bitwise `qsa_paged_sparse_gqa`, which keeps every call they decline.
_GROUPED_DECODE_MIN_REQUESTS = 32  # capacity: fixed: a dispatch threshold
#: Opt-in, attention DP only: `TRTLLM_QWEN4_EXP_DP_GQA_MIN_REQUESTS=<n>` (n in
#: `_GROUPED_SMALL_MIN_GROUPS`..`_GROUPED_DECODE_MIN_REQUESTS`; unset is the row above) takes the
#: request-grouped kernel from n requests of `p` rows a rank instead of 32, the split-K kernel
#: below n. Class S against split-K (another summation order; both within the decode op's band).
#: Below the op's own `MIN_GROUPS` (8) the op plans those launches on its first such call. What
#: was measured is a grouped verify kernel of the same lineage against the per-row kernel with a
#: separate merge and gate launch (not split-K): grouped from 8 requests a rank, and from 2 with
#: -0.12..-0.24 ms a step at 2-7 requests a rank (dep4, GB300); every dep8 point of the AgentX
#: front ran caps 3-28 a rank, where this table runs split-K. Read once, when the core is built;
#: set at a layout without attention DP it warns and does nothing.
DP_GQA_MIN_REQUESTS_ENV = "TRTLLM_QWEN4_EXP_DP_GQA_MIN_REQUESTS"
#: Opt-in, attention DP only: `TRTLLM_QWEN4_EXP_DP_GQA_BITMAP=1` runs the trunk verify step's gated
#: sparse GQA over an e4m3 pool on the bitmap-union grouped kernel
#: (`qsa_paged_sparse_gqa_fp8_verify`: the NVFP4 verify kernel with an e4m3 operand path, a one-read
#: loader, the union scan bounded by the highest selected token and its setup before the PDL wait,
#: two-wave splits at 39-76 requests) instead of `qsa_sparse_gqa_grouped_cute` / split-K, from
#: `_BITMAP_MIN_REQUESTS` requests of `p` rows (`DP_GQA_MIN_REQUESTS_ENV` when set) to
#: `_BITMAP_MAX_REQUESTS`, where the kernel takes the call (`_gqa_bitmap_serves`); every other call
#: keeps the row table. This is the kernel the dep4 / dep8 AgentX front ran (grouped from 8
#: requests, from 2 at caps 3-7). GB300 server A/B against the row table (device step at the cap):
#: -0.082 / -0.071 / -0.038 / -0.047 ms at dep8 12 / 24 / 36 / 40 requests a rank, -0.044 at dep4
#: 40; +0.112 at dep8 48 and +0.093 / +0.094 at dep4 48 / 52, hence the ceiling. Class S against the
#: other two (another union, split plan and summation order; all within the decode op's band of the
#: reference). The draft layer keeps the row table (its one-row passes:
#: `DP_DRAFT_GQA_GROUPED1_ENV`). Read once, when the core is built; set at a layout without
#: attention DP it warns and does nothing.
DP_GQA_BITMAP_ENV = "TRTLLM_QWEN4_EXP_DP_GQA_BITMAP"
#: Opt-in, attention DP only: `TRTLLM_QWEN4_EXP_DP_DRAFT_GQA_GROUPED1=1` runs the draft layer's
#: gated sparse GQA of a pass that holds one row a request (the draft decode passes, and the
#: draft-extend pass's continuing rows) over an e4m3 pool on the same bitmap-union kernel, one
#: request a group (`fp8_sparse_gqa_verify_grouped(..., rows_per_request=1)`: one launch with
#: the gate fused, and the merge where it splits), at `_DRAFT_GQA_G1_ROWS` rows where its plan
#: takes the call (`_gqa_bitmap_serves(..., one_row=True)`, SM103). It does not need
#: `DP_GQA_BITMAP_ENV`. Measured on GB300 dep4 against the per-row kernel, its merge and the gate
#: launch: -0.04..-0.06 ms a step at 6-32 requests a rank; every dep4 / dep8 point of the AgentX
#: front ran it. Not measured against the split-K kernel with the gate fused, which takes these
#: calls without it: that A/B decides whether it stays. Class S (another split plan and summation
#: order, within the decode op's band of the reference). Read once, when the core is built; set at
#: a layout without attention DP it warns and does nothing.
DP_DRAFT_GQA_GROUPED1_ENV = "TRTLLM_QWEN4_EXP_DP_DRAFT_GQA_GROUPED1"
#: Rows (= requests of the rank's draft pass) `DP_DRAFT_GQA_GROUPED1_ENV` serves. Below 4 the
#: grouped kernel's fixed cost did not pay in the micro (us a call saved at 1 / 2 / 3 / 4 / 5
#: rows: -2 / -1 / +0.5 / +2 / +10); 96 is the most rows the micro timed and checked.
_DRAFT_GQA_G1_ROWS = (4, 96)  # capacity: fixed: the measured row range
#: Opt-in and LOSSY against the stock bf16 index keys: `TRTLLM_QWEN4_EXP_INDEX_KEY_FP8=1` keeps
#: each four-token group's compressed index key (its anchor) as e4m3 in the first half of its
#: bf16 slot (`qsa_index_scores_fp8`): every anchor writer converts what it wrote
#: (`_index_anchors_fp8`) and every index score call reads it with the e4m3-rounded query
#: (`_index_scores`; a context call past 256 rows on the ctx-tile kernel). The plane keeps its
#: size and layout, so the cache manager and the KV transfer are unchanged, but a context worker
#: and the generation workers it hands requests to must all set it (the anchors travel as
#: written). Measured with it on every CTX and GEN worker of the dep4 / dep8 AgentX deployments:
#: GB300 dep4, the index-score kernels of a step 2.19 -> 1.82 ms at 96 requests a
#: rank, a 16384-row context step's scores 14.9 -> 9.5 ms; GSM8K 95.91 / 95.98; the AgentX
#: top-2048 block overlap with bf16 keys p50 0.976. Read once, when the core is built, at every
#: layout.
INDEX_KEY_FP8_ENV = "TRTLLM_QWEN4_EXP_INDEX_KEY_FP8"
#: Off by default (race-risk: two launches read step inputs before their grid-dependency wait).
#: At a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) with `INDEX_KEY_FP8_ENV` on and PDL: a
#: trunk attention layer's anchor convert behind the attention decode prologue is launched with
#: PDL (it releases its dependents, then waits, at its top), and the verify step's index scores
#: behind it take the flat grid's early launch (`qsa_index_scores_fp8(..., early=True)`): the
#: tile map and an L2 prefetch of the first tiles' key lines before the wait. Before the wait
#: the scores read only the step's positions, request indices and block table, which the
#: prologue also reads before its own wait. Bitwise. GB300 dep4: the scores' preamble off the
#: critical path, 2.7-3.3 us a layer at 7..25 requests a rank, -0.021..-0.072 ms a step in the
#: server A/B. Read once, when the target is built.
DP_SCORES_EARLY_ENV = "TRTLLM_QWEN4_EXP_DP_SCORES_EARLY"
#: Off by default (unmeasured on this kernel): at a dep4 / dep8 rank (`DP_TOPK_NARROW_ENV`) or a tep4
#: rank (`TEP_TOPK_NARROW_ENV`) on SM103, a generation step's cluster top-k
#: (`qsa_topk_expand_cluster`) stores its selection coalesced (`cowrite=True`): each CTA puts its
#: selected blocks, which hold consecutive positions of the row's list, into a shared position ->
#: block map, then writes its stretch of `topk_indices` and of the token selection with all its
#: threads, instead of each key's thread storing its block and four tokens at scattered columns.
#: The same values at the same columns. The measured `TOPK_NARROW=16w` was a 16-warp leader
#: selection plus this writer in our own leader-CTA cluster kernel (-1.05 us a launch, dep8 g12
#: -0.018 ms a step, TEP -0.016 ms); the kernel the stack runs has no leader CTA (every CTA narrows
#: through DSMEM histograms), so only the writer half has a counterpart, and its gain here is
#: unmeasured. `1` (or the measured `16w` / `w`) turns it on. Read once, when the target is built.
DP_TOPK_NARROW_ENV = "TRTLLM_QWEN4_EXP_DP_TOPK_NARROW"
TEP_TOPK_NARROW_ENV = "TRTLLM_QWEN4_EXP_TEP_TOPK_NARROW"
#: Spellings of the coalesced top-k write's switches that turn it on.
_TOPK_NARROW_ON = ("1", "w", "16w", "true", "on", "yes")
#: Off by default (opt-in: a cold-page codec over the KV cache manager's migration bridge, whose
#: callbacks run on the manager's copy thread). Context servers only, with `INDEX_KEY_FP8_ENV`,
#: block reuse, a host tier, MTP-3 and the `L - 2` in-forward snapshot: every anchor writer also
#: marks the groups it pooled (`qsa_cold_index_mark`, bytes 128-143 of the anchor slot, dead under
#: e4m3 anchors), and the KV cache manager stores the attention pages that leave the GPU without
#: the index bytes no kernel reads again (`qsa_cold_pages`): a page 532,864 -> 462,848 B. Every
#: byte a kernel reads comes back bitwise. Measured: +1.6% context requests/s on the context
#: servers of a dep4 x 2 deployment, more where restores from disk dominate. Read once, when the
#: core is built (the marks) and when the shell is built (the codec).
CTX_COLDIDX_ENV = "TRTLLM_QWEN4_EXP_CTX_COLDIDX"
#: Requests from which `DP_GQA_BITMAP_ENV` takes a step when `DP_GQA_MIN_REQUESTS_ENV` is unset:
#: the measured front's floor at caps 8-52.
_BITMAP_MIN_REQUESTS = 8  # capacity: fixed: a dispatch threshold
#: The most requests a rank `DP_GQA_BITMAP_ENV` takes a trunk verify step at: past it the row
#: table's grouped kernel was faster (dep4 and dep8, 48 and 52 requests a rank).
_BITMAP_MAX_REQUESTS = 40  # capacity: fixed: a dispatch threshold
#: Opt-in, attention DP only: `TRTLLM_QWEN4_EXP_DP_TOPK_CLUSTER_ROWS=<n>` (n in
#: `_TOPK_CLUSTER_MAX_ROWS` + 1..`_TOPK_CLUSTER_PAIR_MAX_ROWS`; unset is the op's own 37) lets a
#: generation step's selection (`_selection`, not a context part) take the cluster top-k up to
#: n rows, in clusters of two CTAs past 37 rows, instead of the one-CTA-a-row kernel. The same
#: selection (exact top-k, the tie rule), so not a numerics question; opt-in because only our own
#: cluster kernel was measured at these rows (dep4: 2 CTAs a row at 39-76 rows, with the verify
#: head shard x 277.0 -> 289.0 at cap 12; trunk 153-256 and draft 39-64 rows -0.19..-0.28 ms a
#: step at graphs 40-52, bitwise), not this one. Read once, when the core is built; set at a
#: layout without attention DP it warns and does nothing.
DP_TOPK_CLUSTER_ROWS_ENV = "TRTLLM_QWEN4_EXP_DP_TOPK_CLUSTER_ROWS"
#: Where `DP_TOPK_CLUSTER_ROWS_ENV` takes a launch past 37 rows (clusters of 2 CTAs), how the
#: rows sit on the clusters (`_topk_placement`; the selection is the same whatever the
#: placement): `(first rows, last rows, solo blocks, row map)` per launch kind, as measured on
#: the dep4 / dep8 AgentX front (GB300, our cluster kernel of the same placements). A row of at
#: most `solo blocks` blocks takes one CTA alone, its pair returning at once; with the row map
#: the shortest rows go two a cluster, one a CTA, so the launch fits one wave of `SMs / 2`
#: clusters (77-104 rows: 19.8-21.9 -> 13.0-13.2 us a launch; -0.10 / -0.13 ms a step at 20 /
#: 24 requests a rank). Trunk 153-256 and draft 39-64 rows (one CTA a row of up to 32768
#: blocks): -0.19..-0.28 ms a step at 40-52 requests a rank.
_TOPK_PLACEMENT = {
    "trunk": ((38, 76, 8192, False), (77, 104, 8192, True), (105, 256, 32768, False)),
    "draft": ((39, 64, 32768, False),),
}  # capacity: fixed: a dispatch table
_QSA_POSITION_AXES = _pool.QSA_POSITION_AXES
# The compressed index keys rotate at coordinate 0 alone (no mRoPE section), as the index
# query does.
_QSA_MROPE_SECTION = None
# The draft experts' activation scales are fp32, not the UE8M0 power-of-two form.
_MTP_MOE_USE_UE8M0 = False
# Rows per draft expert runner call: the FP8 runner's windows are certified up to one 8192-row
# chunk (`fp8_block_scale_moe_runner`). The rows an attention-DP group gathers (up to
# its size times a rank's) run in chunks of it.
_MTP_MOE_MAX_ROWS = 8192

# A context launch of at least this many rows over a bf16 or fp8 pool takes
# `qsa_sparse_gqa_rowtile` (tensor cores, K/V tiles shared by a group of rows); below it the
# per-row kernel is faster.
_ROWTILE_MIN_ROWS = 2048  # capacity: fixed: a dispatch threshold, not a buffer size
# Over an fp8 pool the launch reads its requests' pages widened into a bf16 scratch; a launch
# whose scratch would pass this many bytes stays on the per-row kernel.
_FP8_ROWTILE_MAX_SCRATCH = 1 << 30  # capacity: fixed: bounds a per-call scratch allocation
# A step past `_ROWTILE_MIN_ROWS` context rows whose fp8 scratch is cut into groups
# (`_fp8_context_groups`) runs a group of at least this many rows on the row-tile pass. GB300,
# 24 q / 2 kv heads, 200k-token histories: the widen and the pass take 0.55-0.64 ms a group
# whatever its rows, the per-row kernel about 0.8 us a row (0.49 ms at 512 rows, 0.69 ms at 768).
_FP8_GROUP_ROWTILE_MIN_ROWS = 768  # capacity: fixed: a dispatch threshold, not a buffer size


#: The context-step forms this process took, each with the rows of the call that first took it
#: (`_ctx_form`). Diagnostic only: nothing dispatches on it.
_CTX_FORMS: dict = {}


def _ctx_form(name: str, rows: int) -> None:
    """Record that a context call took form `name` (the index scores' ctx tile, the context
    top-k launch, the row-tile group plan, the in-place row-tile output, the fp8 groups, the
    one-launch MoE, the in-forward split, or the path taken instead). The first time a form is
    seen, log one line with every form seen so far, so a log shows which context forms engaged
    and which did not (the context counterpart of `_decode_forms`)."""
    if name in _CTX_FORMS:
        return
    _CTX_FORMS[name] = rows
    logger.info(
        "context forms: " + ", ".join(f"{k} (first at {v} rows)" for k, v in _CTX_FORMS.items())
    )


# A step with more context rows than this over an NVFP4 pool attends them on the row-tile pass
# over a bf16 scratch of their requests' pages (`_sparse_gqa_nvfp4_context`): at one rank 16384
# context rows 1.36 ms a layer there against 77-948 ms for the NVFP4 per-row kernel.
_NVFP4_ROWTILE_MIN_ROWS = 256  # capacity: fixed: a dispatch threshold, not a buffer size
# The bf16 K/V scratch those context rows read holds every page of each context request's KV
# history. Consecutive context requests are dequantized and attended in groups whose pages fit
# this many bytes; a request with more pages forms a group of its own.
_NVFP4_CONTEXT_SCRATCH_BYTES = 128 << 20  # capacity: fixed: a scratch memory budget

#: Set to 1 to run the GDN input / output and attention input / output projections in FP8
#: (`fp8_per_tensor_mm`: e4m3 weights with a per-tensor scale, the activation quantized per
#: call with a dynamic per-tensor scale) wherever a call reaches `_DENSE_FP8_MIN_ROWS` rows, and
#: decode rows on `fp8_gemv` over packed FP8 copies with 128x128 block scales (`_FP8_GEMV_ROWS`)
#: or `mid_rows_mm` (`_MID_MM_ROWS`). A lossy mode beside the checkpoint's bf16 projections
#: (gsm8k-gated), certified at tp1; off by default and at every other layout.
_DENSE_FP8_ENV = "TRTLLM_MODELING_V2_DENSE_FP8"

#: Set to 1 to give each GDN layer an NVFP4 copy of its input projection (e2m1 values, one e4m3
#: scale per 16 hidden elements of a row, an fp32 global scale; `nvfp4_gemv.pack_split`, built
#: at load from the bf16 weight) and run its decode rows -- up to `_INPROJ_NVFP4_MAX_ROWS` -- on
#: `fp8_gemv`'s kernel over it, the conv step as the GEMV's epilogue in a captured step
#: (single-token and verify). Lossy (gsm8k-gated), certified at tp1; off by default.
_GDN_INPROJ_NVFP4_ENV = "TRTLLM_MODELING_V2_GDN_INPROJ_NVFP4"

#: Rows from which a projection takes the per-tensor FP8 GEMM, per projection kind: the input
#: projections from 2 rows (one row stays on the GEMV entries, whose PDL overlap the two
#: quantization launches would break), the K=6144 output projections only where the GEMM saving
#: outgrows the activation quantization's two launches (GB300 micro-bench: cuBLASLt bf16 8-10 us
#: vs FP8 6-7 us + ~4 us below ~512 rows).
_DENSE_FP8_MIN_ROWS = {"in": 2, "out": 512}  # capacity: fixed: dispatch thresholds

#: Row ranges `(lo, hi)` in which a projection with a packed FP8 decode copy runs `fp8_gemv`
#: (FP8 weights, bf16 activations, one PDL launch), per kind; each upper end is that kind's
#: crossover against the next entry (GB300 micro, us): in 8.7 vs ~9 `mid_rows_mm` at 16 rows;
#: out 6.4 vs 8.4 bf16 at 16; gate|up 2.6 vs 4.0 at 16; down 1.7 vs 2.9 at 8.
_FP8_GEMV_ROWS = {"in": (1, 16), "out": (1, 16), "sh_gu": (1, 16), "sh_dn": (1, 8)}  # dispatch

#: Row ranges in which a projection runs `mid_rows_mm` (tcgen05, the per-tensor FP8 codes or the
#: bf16 weight) between `fp8_gemv` and the GEMM entries.
_MID_MM_ROWS = {
    "in": (17, 256),
    "out": (17, 128),
    "sh_gu": (17, 64),
    "sh_dn": (0, -1),
    "router": (17, 64),
}  # capacity: fixed: dispatch thresholds

#: `_mid` weight-name patterns -> the `_MID_MM_ROWS` kind their projections dispatch as.
_MID_KINDS = (
    (r"l\d+_(attn_in_w|in_proj_w)", "in"),
    (r"l\d+_(o_w|out_w)", "out"),
    (r"l\d+_sh_gu", "sh_gu"),
    (r"l\d+_sh_dn", "sh_dn"),
    (r"l\d+_router", "router"),
)

#: Rows up to which a GDN layer's decode in_proj reads the NVFP4 copy (`_GDN_INPROJ_NVFP4_ENV`).
#: GB300 micro (us, conv included, behind a PDL predecessor): 6.4 / 6.1 at 1 / 4 rows against
#: fp8_gemv_conv1d's 7.9 / 7.7; alone 7.1 against 7.9 at 8 rows, 9.1 against 8.6 at 16.
_INPROJ_NVFP4_MAX_ROWS = 8  # capacity: fixed: a dispatch threshold, at most fp8_gemv's MAX_ROWS

#: Rows up to which a captured verify step's conv runs in the in_proj GEMV's epilogue
#: (`fp8_gemv_conv1d_verify`, certified at 2..16 rows); more rows take the Triton conv.
_VERIFY_CONV_FUSED_MAX_ROWS = 16  # capacity: fixed: the entry's certified rows

#: Rows up to which a captured single-token step runs the conv step as `fp8_gemv_conv1d`'s
#: epilogue over the packed FP8 in_proj copy (dense FP8 mode).
_FP8_CONV_FUSED_MAX_ROWS = 8  # capacity: fixed: a dispatch threshold


#: The spellings of a `TRTLLM_MODELING_V2_*` switch: these turn it on, these off.
_SWITCH_ON = frozenset({"1", "true", "yes", "on"})
_SWITCH_OFF = frozenset({"0", "false", "no", "off"})


def _switch(env: str, default: bool = False) -> bool:
    """A `TRTLLM_MODELING_V2_*` switch: unset or empty is `default`; any spelling outside
    `_SWITCH_ON` / `_SWITCH_OFF` raises."""
    raw = os.environ.get(env, "")
    value = raw.strip().lower()
    if not value:
        return default
    if value in _SWITCH_ON:
        return True
    if value in _SWITCH_OFF:
        return False
    raise ValueError(f"{env}={raw!r}: expected 1 / 0 (or true / false, yes / no, on / off)")


def _gemv_ready(x: torch.Tensor) -> bool:
    """`fp8_gemv`'s activation layout: unit column stride, rows 16-byte aligned."""
    return x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0


def _mid_rows(kind: str, rows: int) -> bool:
    lo, hi = _MID_MM_ROWS[kind]
    return lo <= rows <= hi


def _dense_mm(x, weight_t, fp8, gemm, kind: str, packed=None, mid=None) -> torch.Tensor:
    """A dense projection: `fp8_gemv` over the packed decode copy (`packed`) within
    `_FP8_GEMV_ROWS[kind]` rows, `mid_rows_mm` over `mid` (a `MidWeight`) within
    `_MID_MM_ROWS[kind]` rows, the per-tensor FP8 GEMM (`fp8` = `(codes_t, scale)`) from
    `_DENSE_FP8_MIN_ROWS[kind]` rows, else the step's bf16 entry `gemm` -- every call when the
    dense FP8 mode is off (all copies None)."""
    rows = x.shape[0]
    lo, hi = _FP8_GEMV_ROWS.get(kind, (1, 0))
    if packed is not None and lo <= rows <= hi and _gemv_ready(x):
        return fp8_gemv_op(x, packed)
    if mid is not None and _mid_rows(kind, rows) and (not mid.fp8 or _gemv_ready(x)):
        return mid_rows_mm_op(x, mid)
    if fp8 is not None and rows >= _DENSE_FP8_MIN_ROWS[kind]:
        return fp8_per_tensor_mm(x, *fp8)
    return gemm(x, weight_t)


def _draft_mm(x: torch.Tensor, weight_t: torch.Tensor, packed, gemm) -> torch.Tensor:
    """A draft-layer projection: `fp8_gemv` over its packed FP8 copy (`_MTP_DENSE_FP8_ENV`) at up
    to `_MTP_DENSE_FP8_MAX_ROWS` rows, else `gemm` over the bf16 `weight_t`."""
    if packed is not None and x.shape[0] <= _MTP_DENSE_FP8_MAX_ROWS and _gemv_ready(x):
        return fp8_gemv_op(x, packed)
    return gemm(x, weight_t)


def _precompile_mid(mid: dict) -> None:
    """Compile every `mid_rows_mm` variant the decode dispatch can reach, once per weight
    geometry: a variant first met by a serving step would stall it for the compile."""
    seen = {}
    for name, mw in mid.items():
        kind = next(k for rx, k in _MID_KINDS if re.fullmatch(rx, name))
        lo, hi = _MID_MM_ROWS[kind]
        key = (mw.n, mw.k, mw.fp8, None if mw.scale is None else mw.scale.numel())
        if lo <= hi and key not in seen:
            seen[key] = mid_rows_precompile(mw, max(lo, 1), hi)


#: Set to 1 to give the FP8 draft head an NVFP4 copy of its lm_head (e2m1 values, one e4m3
#: scale per 16 hidden elements of a vocabulary row, an fp32 global scale; built at load) and run
#: its decode rows -- up to `_FP8_GEMV_HEAD_ROWS` -- on `nvfp4_gemv` (GB300, 248320 x 2560: 54-57
#: us at 1-8 rows, 66-67 at 9-16, against fp8_gemv's 87-90 and 96-98). The verify head is
#: untouched: outputs are unchanged and the acceptance length is the cost to watch. Certified at
#: tp1; 0 (default): `fp8_gemv`.
_MTP_HEAD_NVFP4_ENV = "TRTLLM_MODELING_V2_MTP_HEAD_NVFP4"

#: Set to 1, under the dense FP8 mode, to run the target head's decode rows (up to
#: `_FP8_GEMV_HEAD_ROWS`) on `nvfp4_gemv` over an NVFP4 copy of lm_head -- the draft head's under
#: `_MTP_HEAD_NVFP4_ENV`, else one built at load -- instead of `fp8_gemv`. Lossy: the verify
#: logits change. Certified at tp1; 0 (default): `fp8_gemv`.
_TARGET_HEAD_NVFP4_ENV = "TRTLLM_MODELING_V2_TARGET_HEAD_NVFP4"

#: Rows up to which the vocabulary projection (the draft head; under the dense FP8 mode the
#: verify head too) runs `fp8_gemv` on the packed copy; more rows take cuBLASLt over the
#: per-row-scaled copy. GB300, 248320 x 2560: fp8_gemv 86-98 us at 1-16 rows against 107-134,
#: then 142 at 20 rows and 172 at 32 against a flat ~108.
_FP8_GEMV_HEAD_ROWS = 16  # capacity: fixed: a dispatch threshold, at most fp8_gemv's MAX_ROWS

#: Rows up to which the FP8 vocabulary projection's rows past `_FP8_GEMV_HEAD_ROWS` run
#: `mid_rows_mm` over the per-row-scaled copy (activation quantized per row); more rows keep
#: cuBLASLt. GB300, 248320 x 2560, us at 17 / 32 / 64 / 128 rows: 99.7 / 101.5 / 106.0 / 125.8
#: against 110.0 / 111.9 / 107.7 / 137.5.
_MID_HEAD_ROWS = 128  # capacity: fixed: a dispatch threshold, at most mid_rows_mm's MAX_ROWS

#: Set to 1 to run the draft layer's dense projections (`fc_e`, `fc_h`, the fused attention
#: input, `o`, the shared expert's gate|up and down) on `fp8_gemv` over packed FP8 copies at decode
#: rows (up to `_MTP_DENSE_FP8_MAX_ROWS`). The draft only proposes tokens, so outputs are unchanged;
#: only the acceptance length can move. Certified at tp1; 0 (default): bf16.
_MTP_DENSE_FP8_ENV = "TRTLLM_MODELING_V2_MTP_DENSE_FP8"
_MTP_DENSE_FP8_MAX_ROWS = 16  # capacity: fixed: a dispatch threshold, at most fp8_gemv's MAX_ROWS

#: Set to 1 to run the draft layer's routed experts in NVFP4 on the trunk's trtllm-gen runner: at
#: load the checkpoint's FP8 128x128 block-scaled experts are requantized (per-expert weight global
#: scales, 16-element e4m3 block scales) into the trunk's layout and the FP8 copies are freed. The
#: draft only proposes tokens, so this can change the acceptance length, not outputs. Certified at
#: tp1; 0 (default) retains the checkpoint's FP8 or native NVFP4 format.
_MTP_MOE_NVFP4_ENV = "TRTLLM_MODELING_V2_MTP_MOE_NVFP4"

#: The NVFP4 draft MoE's two activation global scales are static, `448 * 6 / A` with `A` this factor
#: times the largest calibrated amax of the trunk's routed experts (its `input_scale`s, FC1 input and
#: FC2 input; an FP8 draft carries none). Native NVFP4 uses its own checkpoint scales.
#: A global scale only has to keep the e4m3 block scales
#: inside their range, so the margin costs nothing until blocks are ~1e-3 of `A`.
_MTP_MOE_NVFP4_FC1_MARGIN = 8.0
_MTP_MOE_NVFP4_FC2_MARGIN = 4.0

_PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"

#: The opt-in conversions certified at tp1; other layouts retain checkpoint formats.
_ONE_RANK_SWITCHES = (
    _DENSE_FP8_ENV,
    _GDN_INPROJ_NVFP4_ENV,
    _TARGET_HEAD_NVFP4_ENV,
    _MTP_HEAD_NVFP4_ENV,
    _MTP_MOE_NVFP4_ENV,
    _MTP_DENSE_FP8_ENV,
)


def _check_switches(one_rank: bool) -> None:
    """Read every opt-in switch at construction, at any layout, so a misspelled value raises,
    and warn about one set where it takes no effect: away from tp1, or the target head's NVFP4
    copy without the dense FP8 mode whose head it replaces."""
    on = [env for env in _ONE_RANK_SWITCHES if _switch(env)]
    if on and not one_rank:
        logger.warning(
            f"{', '.join(on)}: certified at tp1 only; this layout runs the bf16 paths and "
            "ignores them"
        )
    if one_rank and _TARGET_HEAD_NVFP4_ENV in on and _DENSE_FP8_ENV not in on:
        logger.warning(
            f"{_TARGET_HEAD_NVFP4_ENV}=1 takes effect only with {_DENSE_FP8_ENV}=1; the target "
            "head stays bf16"
        )


#: At one rank, context-step MoE launches of at least this many rows (and at most
#: `_CUTE_DSL_MOE_MAX_ROWS`, the op's certified top) run on the CuTe DSL MoE
#: (`fp4_moe_renorm_cute_dsl`, over the second expert copy `weights.py` keeps in its layout);
#: smaller ones, every decode one among them, keep the trtllm-gen runner. GB300 micro-benches,
#: us a layer, CuTe DSL vs trtllm-gen: 2048 rows 275 vs 273, 8192 rows 451 vs 555, 16384 rows
#: 656 vs 882.
_CUTE_DSL_MOE_MIN_ROWS = 4096  # capacity: fixed: a dispatch threshold, not a buffer size
_CUTE_DSL_MOE_MAX_ROWS = 16384  # capacity: fixed: the op's certified rows
#: At one rank without that copy, a context-step launch of more than `_MOE_MAX_ROWS` rows and at
#: most this many is one trtllm-gen runner call (the routed op's certified rows over the whole
#: layer): the runner streams the layer's experts once instead of once a chunk. GB300, 16384
#: rows: 1356 -> 1059 us a layer, an engine step's GPU forward 225.4 -> 212.5 ms. A CTX whose
#: `max_num_tokens` is 24576 steps that many rows in one launch too (every row bitwise the
#: chunks at 20480 and 24576 rows); with that chunk size its saturated context loop took 5-7%
#: more tokens a second than at 16384.
_MOE_ONE_LAUNCH_MAX_ROWS = 24576  # capacity: fixed: the routed op's certified rows
# Whether one rank keeps that second copy (about 1.3 GiB a layer). No engine arg owns it, hence
# an env var read at construction (`_switch`, default on); off drops the copy and every launch
# runs on trtllm-gen.
_MOE_PREFILL_COPY_ENV = "TRTLLM_MODELING_V2_MOE_PREFILL_COPY"
#: Switch, on by default: a captured generation-only step at an attention-DP rank on SM103
#: (trtllm-gen trunk MoE, a 128-expert window of the 512 experts at dep4, a 64-expert one at dep8,
#: `_ROUTE_FRONT_WINDOWS`) runs each trunk MoE block's
#: routing, the window's pre-routed tables and the NVFP4 quantization of the gathered rows in
#: one launch (`ops/moe/moe_route_front__decode`) and enters the runner pre-routed at the tactic
#: the tables were built for (`_ROUTE_FRONT_TACTICS`): it replaces `fp4_quantize` and the
#: runner's own routing launches. The same bits (the build checks every layer once, `_RouteFront`).
#: `0` keeps today's path. Read once, when the target is built.
DP_ROUTE_FRONT_ENV = "TRTLLM_QWEN4_EXP_DP_ROUTE_FRONT"
#: Gathered rows a route front serves: from 3 requests of 4 rows at dep4 to the kernel's last.
_ROUTE_FRONT_MIN_ROWS = 48  # capacity: fixed: a dispatch threshold
#: The expert window the route front and `_ROUTE_FRONT_TACTICS` are certified for (dep4, tep4).
_ROUTE_FRONT_WINDOW = 128  # capacity: fixed: the certified window, not a buffer size
#: The window an attention-DP group of this size serves the route front at: dep4's 128 experts
#: (`_ROUTE_FRONT_TACTICS`) and dep8's 64 (`_ROUTE_FRONT_TACTICS_W64`). dep2's 256 keeps the
#: runner's routing.
_ROUTE_FRONT_WINDOWS = {4: _ROUTE_FRONT_WINDOW, 8: 64}  # capacity: fixed: certified windows
#: `(gathered rows below, (tileN, FC1 x FC2 config))`: the trtllm-gen tactic pinned per row
#: count, the tile the tables are built for. A 128-expert window under real routing is hit by
#: 45-100 experts with a few pairs each, so the smaller tiles cut an expert into more tiles than
#: it needs past 96 rows (GB300 dep4, every listed config bitwise one another). A tactic the
#: runner does not list for the shape falls back to `_ROUTE_FRONT_TACTICS_BASE`'s.
_ROUTE_FRONT_TACTICS = ((96, (8, 53)), (192, (16, 32)), (704, (32, 32)), (1025, (64, 4)))
_ROUTE_FRONT_TACTICS_BASE = ((256, (8, 53)), (512, (16, 32)), (1025, (32, 20)))
#: The same for dep8's 64-expert window (8 ranks of 3..32 requests of four rows). The runner sizes
#: its configs for "every pick of a row lands in this window" (a dep8 window gets 1.25 of a row's
#: 10 picks), so it lists tile 32 only at 256 rows, tile 64 and up at 512-768 and tile 128 and up
#: from 1024; the smaller tiles below run fewer, fuller tiles. GB300, 8 ranks over two nodes, each
#: on its own window under recorded routing: [32, 32] 40.2 against the listed [64, 4] 41.8 us a
#: layer at 512 rows, [64, 5] 50.6 against [128, 4] 54.6 at 1024; [32, 32] up to 895 rows took
#: -0.10 / -0.05 ms a step at 20 / 24 requests a rank against [64, 4] from 640 rows. A tactic the
#: runner does not list for the row count is pinned only once its window output is bitwise a
#: listed tactic's (`_route_front_check` against `_route_front_w64_tactic`'s listed reference);
#: a row count whose check fails keeps the runner's routing.
_ROUTE_FRONT_TACTICS_W64 = ((96, (8, 53)), (192, (16, 32)), (896, (32, 32)), (1025, (64, 5)))
#: The high-cap row tables: at a dep4 / dep8 rank the decode forms that would stop at 1024 gathered
#: rows (32 requests a rank at dep8 MTP-3) or at 128 / 152 rows a rank serve the CUDA-graph batch
#: sizes past them, up to 56 requests a rank. Unset, a dep8 rank takes `_DP_HICAP_DEFAULT` (`route`
#: and `fin`: both bitwise and held by build checks) and a dep4 rank none. `0` turns every part off;
#: `1` / `all` takes every part at both layouts; a comma list takes exactly those parts:
#: - `route` (bitwise): dep8's route front and the forms that ride on it (the packed gather's
#:   routing-only front, the owner top-k gather under `DP_MOE_OWNK_ENV=all`, the fast path, the
#:   early release and the fused finalize reduce-scatter) at 1025..`_DP_HICAP_ROUTE_MAX_ROWS`
#:   gathered rows: the launch's three / four tokens a warp on the one-token fast path, at
#:   `_DP_HICAP_TACTICS_W64`'s tactic, held by the same build checks (`_route_front_check`
#:   against the runner's own routing, `_dp_pack_check`) as every other row count.
#: - `fin` (bitwise): the fused finalize reduce-scatter (`DP_MOE_FIN_RS_ENV`) past 152 rows a
#:   rank, where `streams_per_cta`'s two-streams-a-CTA layout would run two waves: its layouts end
#:   at one resident wave (`_DP_HICAP_FIN_CTAS`), so from there every stream of a row shares one
#:   CTA and the grid stays resident (any layout reads the same bytes and computes the same bits).
#:   Ignored where `DP_RS_INJECT_LAYOUTS_ENV` names the layouts.
#: - `head` (class S, as the head shard itself): the sharded verify head (`DP_HEAD_SHARD_ENV`) to
#:   `_DP_HICAP_HEAD_VERIFY` requests a rank (the full-vocabulary head and the stock sampler
#:   past 32 today).
#: Read once, when the target is built.
DP_HICAP_ENV = "TRTLLM_QWEN4_EXP_DP_HICAP"
_DP_HICAP_PARTS = ("route", "fin", "head")
#: The parts a rank takes with `DP_HICAP_ENV` unset, by attention-DP size.
_DP_HICAP_DEFAULT = {8: frozenset({"route", "fin"})}
#: Gathered rows dep8's route front serves under `route`: 56 requests a rank of four rows.
_DP_HICAP_ROUTE_MAX_ROWS = 1792  # capacity: fixed: the measured caps' last
#: dep8's tactic past 1024 rows: tile 64 (the runner lists tile 128 and up there, so the build
#: holds it to a listed tactic's window output, `_route_front_w64_tactic`).
_DP_HICAP_TACTICS_W64 = ((_DP_HICAP_ROUTE_MAX_ROWS + 1, (64, 5)),)
#: CTAs of one resident fused finalize wave on GB300 (`FIN_CTAS_PER_SM` x 152 SMs).
_DP_HICAP_FIN_CTAS = 304  # capacity: fixed: a dispatch threshold
#: Requests a rank the sharded verify head serves under `head`.
_DP_HICAP_HEAD_VERIFY = (1, 56)  # capacity: fixed: dispatch thresholds


def _dp_hicap(core=None) -> frozenset:
    """`DP_HICAP_ENV` at `core`'s rank: the parts it names, `_DP_HICAP_DEFAULT`'s for the rank's
    attention-DP size where it is unset (none without a core), empty where it is off; ValueError on
    another spelling."""
    raw = os.environ.get(DP_HICAP_ENV, "")
    value = raw.strip().lower()
    if not value:
        return _DP_HICAP_DEFAULT.get(getattr(core, "dp_size", 1), frozenset())
    if value in _SWITCH_OFF:
        return frozenset()
    if value in _SWITCH_ON or value == "all":
        return frozenset(_DP_HICAP_PARTS)
    parts = frozenset(p.strip() for p in value.split(",") if p.strip())
    if not parts or not parts <= frozenset(_DP_HICAP_PARTS):
        raise ValueError(
            f"{DP_HICAP_ENV}={raw!r}: expected 0, 1, all or a comma list of "
            f"{', '.join(_DP_HICAP_PARTS)}"
        )
    return parts


def _dp_hicap_route_rows(core) -> int:
    """The most gathered rows the attention-DP route front serves at this rank: `MAX_ROWS`, or
    `_DP_HICAP_ROUTE_MAX_ROWS` at dep8 (the 64-expert window) under `DP_HICAP_ENV`'s `route`."""
    if (
        core.dp_size == 8
        and core.local_experts == _ROUTE_FRONT_WINDOWS[8]
        and "route" in _dp_hicap(core)
    ):
        return _DP_HICAP_ROUTE_MAX_ROWS
    return _ROUTE_FRONT_MAX_ROWS


#: Switch, on by default: where a route front was built over dep4's 128-expert or dep8's 64-expert
#: window, its launches of `_ROUTE_FRONT_FAST2_ROWS` gathered rows (two tokens a warp) route each
#: token on the kernel's one-token fast path (`moe_route_front(..., fast2=)`) instead of both on
#: the full per-lane sort, in the form `_ROUTE_FRONT_FAST2_FORMS` lists for the window. The same
#: experts, weights, codes and tables, bit for bit (the build's check against the runner's own
#: routing still gates every row count; a row count that fails it keeps the two-token sort).
#: GB300, dep8 (form 1), us a launch: 8.72 -> 5.41 at 544 rows, 8.71 -> 6.27 at 768, 8.93 -> 7.41 at
#: 1024; server step at 17 / 18 / 20 / 24 requests a rank: -0.15..-0.23 / -0.166 / -0.145 / -0.134
#: ms. dep4 (form 2, the second token's logits loaded first): 9.34 -> 5.94 / 9.42 -> 5.92 / 9.39 ->
#: 6.37 / 9.38 -> 6.65 at 528 / 640 / 768 / 832 rows (form 1: 6.41 at 528); server step at 33 / 40 /
#: 48 requests a rank -0.15 / -0.12 / -0.09..-0.12 ms. `0` keeps the two-token sort; `2` takes
#: form 2 at both windows (an A/B value). Read once, when the target is built.
DP_ROUTE_FAST2_ENV = "TRTLLM_QWEN4_EXP_DP_ROUTE_FAST2"
#: Gathered rows a launch takes the fast path at: two tokens a warp, up to the launch's last.
_ROUTE_FRONT_FAST2_ROWS = (513, 1024)  # capacity: fixed: dispatch thresholds
#: The fast path's form per expert window, as measured: dep4's 128 experts with the second token's
#: logits loaded first (2), dep8's 64 one token after the other (1).
_ROUTE_FRONT_FAST2_FORMS = {128: 2, 64: 1}  # capacity: fixed: a dispatch table
#: Switch, on by default: an attention-DP route front past the lean form's rows
#: (`_ROUTE_FRONT_LEAN_MAX_ROWS`, which releases at its start already) releases its dependent, the
#: runner's FC1, right after its own `griddepcontrol.wait`, which is then the launch's first
#: statement, instead of after its last global access (`moe_route_front(..., early=True)`). FC1's
#: own wait still holds its reads until the front has completed, so the bits are the same; FC1's
#: launch and prologue overlap the routing. Measured as `DP_MOE_ROUTE_FRONT_EARLY` on every dep4 /
#: dep8 front point; d6a micro inside the dep4 round trip at 24 / 32 requests a rank (384 / 512
#: gathered rows): -0.385..-0.395 / -0.406..-0.411 ms a step with the release against -0.349..-0.358
#: / -0.370..-0.379 without, about 0.035 ms for the release. Engages only with PDL. `0` keeps the
#: end release. Read once, when the target is built.
DP_ROUTE_FRONT_EARLY_ENV = "TRTLLM_QWEN4_EXP_DP_ROUTE_FRONT_EARLY"
#: Switch, on by default: a tep4 rank (attention TP over four ranks without attention DP, the
#: routed experts as 128-expert windows) runs a captured generation-only step on the forms the
#: one-rank target runs, at its shard:
#: - every trunk hyper-connection site of up to `_HC_DECODE_ROWS_MAX` rows on the one-rank GEMV
#:   or tcgen05 rows pair (`hc_decode_pairs`). The sites are replicated work on the all-reduced
#:   rows, so the four ranks still compute the same bits; `hc_out` is bitwise, the mixer output
#:   the pair's rounding of the same sums (1-2 bf16 ulp from the three-launch site's).
#: - the gated sparse GQA of the rank's 6 q heads on its one e4m3 KV head, up to
#:   `_TEP_SPLITK_MAX_ROWS` rows, on `qsa_sparse_gqa_splitk_cute` (the gate in its merge) in
#:   place of the per-row Triton kernel and the gate launch: exactly widened K/V, q and P in f16,
#:   fp32 accumulation, so the output moves by an ulp on some elements.
#: - each trunk MoE block on the route front (`_RouteFront`) up to `_TEP_ROUTE_FRONT_MAX_ROWS`
#:   rows, the runner entered pre-routed over the window on the site's NVFP4 codes. Routing and
#:   codes are bitwise (build-checked); the runner runs `DECODE_TACTICS`' tactic where the stock
#:   window call runs the AutoTuner's, so the block is bitwise today's as far as the trtllm-gen
#:   tactics agree (measured bitwise one another on the whole layer, not on the window).
#: - the GDN verify conv at `_TEP_CONV_WAIT_ROWS` rows as a PDL dependent of the input projection
#:   (bitwise).
#: The first two are same-precision reorders: a draft or target token can move at a near-tie.
#: `0` keeps today's forms. Read once, when the target is built; set to 1 at another layout it
#: warns and does nothing.
TEP_FAST_DECODE_ENV = "TRTLLM_QWEN4_EXP_TEP_FAST_DECODE"
#: The group the fast forms are written and measured for: tep4 (6 q heads on 1 KV head a rank).
_TEP_RANKS = 4  # capacity: fixed: the certified layout
#: Rows per (q heads, KV heads, pool dtype) up to which a tep4 rank sends a gated decode GQA to
#: `qsa_sparse_gqa_splitk_cute` beyond the op's own table (`serves`, which the FP8 target routes
#: by): 1-4 requests of four verify rows over the e4m3 pool.
_TEP_SPLITK_MAX_ROWS = {(6, 1, torch.float8_e4m3fn): 16}
#: Rows up to which a tep4 rank's trunk MoE takes the route front (1-4 requests of four rows).
#: Its tactic is `DECODE_TACTICS`' pin for the row count's bucket: measured on the whole
#: 512-expert layer (one rank), where the tactics compared were bitwise one another. A stock tep4
#: window call is not pinned (`fp4_block_scale_moe_renorm_routed` pins the whole layer only) and
#: runs the AutoTuner's tactic; the build check compares at the pinned tactic.
_TEP_ROUTE_FRONT_MAX_ROWS = 16  # capacity: fixed: a dispatch threshold


def _dp_gqa_min_requests(core) -> int:
    """`DP_GQA_MIN_REQUESTS_ENV` at this core's layout: the requests from which a generation
    step's rows take `qsa_sparse_gqa_grouped_cute` (`_GROUPED_DECODE_MIN_REQUESTS` unset or at a
    layout without attention DP). A value outside the op's range raises."""
    raw = os.environ.get(DP_GQA_MIN_REQUESTS_ENV, "").strip()
    if not raw:
        return _GROUPED_DECODE_MIN_REQUESTS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not _GROUPED_SMALL_MIN_GROUPS <= value <= _GROUPED_DECODE_MIN_REQUESTS:
        raise ValueError(
            f"{DP_GQA_MIN_REQUESTS_ENV}={raw!r}: expected an integer in "
            f"{_GROUPED_SMALL_MIN_GROUPS}..{_GROUPED_DECODE_MIN_REQUESTS}"
        )
    if core.dp_size == 1:
        logger.warning(
            f"{DP_GQA_MIN_REQUESTS_ENV} is set but ignored: it serves attention-DP ranks; this "
            f"rank has attention DP {core.dp_size}"
        )
        return _GROUPED_DECODE_MIN_REQUESTS
    return value


def _dp_gqa_bitmap(core) -> int:
    """`DP_GQA_BITMAP_ENV` at this core's layout: the requests from which the trunk verify
    step's sparse GQA takes the bitmap-union kernel (`DP_GQA_MIN_REQUESTS_ENV` when set, else
    `_BITMAP_MIN_REQUESTS`), or 0 (off, or a layout without attention DP). Reads
    `core.dp_gqa_min_requests`, so it runs after `_dp_gqa_min_requests`."""
    if not _switch(DP_GQA_BITMAP_ENV):
        return 0
    if core.dp_size == 1:
        logger.warning(
            f"{DP_GQA_BITMAP_ENV} is set but ignored: it serves attention-DP ranks; this rank "
            f"has attention DP {core.dp_size}"
        )
        return 0
    if os.environ.get(DP_GQA_MIN_REQUESTS_ENV, "").strip():
        return core.dp_gqa_min_requests
    return _BITMAP_MIN_REQUESTS


def _dp_draft_gqa_g1(core) -> bool:
    """`DP_DRAFT_GQA_GROUPED1_ENV` at this core's layout: whether the draft layer's one-row passes
    take the bitmap-union kernel (off at a layout without attention DP, with one warning)."""
    if not _switch(DP_DRAFT_GQA_GROUPED1_ENV):
        return False
    if core.dp_size == 1:
        logger.warning(
            f"{DP_DRAFT_GQA_GROUPED1_ENV} is set but ignored: it serves attention-DP ranks; this "
            f"rank has attention DP {core.dp_size}"
        )
        return False
    return True


def _dp_scores_early(core, device) -> bool:
    """`DP_SCORES_EARLY_ENV`, read once at build: True at a dep4 / dep8 rank on SM103 with the e4m3
    index anchors and PDL on; set where it cannot apply: one warning, False."""
    if not _switch(DP_SCORES_EARLY_ENV):
        return False
    why = None
    if not _dp_decode_rank(core, device):
        why = "it serves a dep4 / dep8 rank on SM103"
    elif not getattr(core, "index_key_fp8", False):
        why = f"it serves the e4m3 index anchors ({INDEX_KEY_FP8_ENV}), which are off"
    elif not _INDEX_SCORES_PDL:
        why = "programmatic dependent launch is off (TRTLLM_ENABLE_PDL=0)"
    if why is not None:
        logger.warning(f"{DP_SCORES_EARLY_ENV} is set but ignored: {why}")
        return False
    logger.info(
        f"trunk verify index scores launched early behind a PDL anchor convert "
        f"({DP_SCORES_EARLY_ENV}=1): rank {core.dp_rank} of {core.dp_size}"
    )
    return True


def _topk_cowrite_build(core, device) -> bool:
    """`DP_TOPK_NARROW_ENV` at a dep4 / dep8 rank, `TEP_TOPK_NARROW_ENV` at a tep4 rank, read once
    at build: True on SM103 with the layout's switch on; set where it cannot apply: one warning,
    False. A value outside `_TOPK_NARROW_ON` and `0` raises."""
    dp = core.dp_size > 1
    env = DP_TOPK_NARROW_ENV if dp else TEP_TOPK_NARROW_ENV
    on = False
    for name in (DP_TOPK_NARROW_ENV, TEP_TOPK_NARROW_ENV):
        raw = os.environ.get(name, "").strip()
        value = raw.lower()
        if value and value not in _TOPK_NARROW_ON and value not in ("0", "false", "off", "no"):
            raise ValueError(f"{name}={raw!r}: expected 1 (or 16w / w) or 0")
        if value in _TOPK_NARROW_ON:
            if name == env:
                on = True
            else:
                logger.warning(
                    f"{name} is set but ignored: it serves "
                    + ("a tep4 rank" if dp else "a dep4 / dep8 rank")
                    + f"; this rank has attention DP {core.dp_size}"
                )
    if not on:
        return False
    serves = (
        _dp_decode_rank(core, device)
        if dp
        else (
            _tep4_layout(core)
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == archs.SM_103
        )
    )
    if not serves:
        logger.warning(
            f"{env} is set but ignored: it serves a dep4 / dep8 or a tep4 rank on SM103; this rank "
            f"has attention DP {core.dp_size}, attention TP {core.attn_tp_size}"
        )
        return False
    try:
        bad = _topk_cowrite_check(core, device)
    except Exception as e:  # noqa: BLE001 -- this rank alone declines; no collective follows
        bad = f"the check raised {type(e).__name__}: {e}"
    if bad:
        # Rank-local: the selection is the same bits either way, so ranks may differ here.
        logger.warning(f"{env} is set but ignored at rank {core.mapping.tp_rank}: {bad}")
        return False
    raw = os.environ.get(env, "").strip()
    logger.info(
        f"cluster top-k stores its selection coalesced ({env}={raw}): rank {core.mapping.tp_rank} "
        f"of {core.mapping.tp_size}"
        + (
            "; the measured 16-warp leader selection has no counterpart in this kernel, only "
            "the writer half"
            if raw.lower() == "16w"
            else ""
        )
    )
    return True


#: Score-table width of the coalesced top-k write's build check (262144 tokens with MTP's pages).
_TOPK_COWRITE_CHECK_WIDTH = 65568  # capacity: fixed: a check shape


def _topk_cowrite_check(core, device) -> str:
    """The cluster top-k with and without `cowrite` over the same random rows (windows from a few
    blocks to the whole table, ties at the threshold), at every generation row count of this rank
    the cluster serves (the trunk's `max_verify_rows` x a graph batch size, a draft step's batch
    size, and one row) under each placement `_selection` gives it past 37 rows (trunk and draft):
    every launch plan (cluster size, `solo_blocks`, row map) the rank runs, on the operand dtypes
    `_selection` passes (int64 positions, int32 sequence lengths); the row count is a runtime
    argument, so the compiles are one per plan. "" when `topk_indices` and the token selection are
    bitwise equal, else what differs. Rank-local (no collective)."""
    limit = getattr(core, "dp_topk_cluster_rows", _TOPK_CLUSTER_MAX_ROWS)
    sizes = getattr(core.model_config, "cuda_graph_batch_sizes", None) or ()
    draft = {int(b) for b in sizes if 1 <= int(b) <= limit}
    rows_list = sorted({1, *_graph_rows(core, 1, limit), *draft})
    width = _TOPK_COWRITE_CHECK_WIDTH
    k, cr = core.block_topk, core.compress_ratio
    gen = torch.Generator(device=device).manual_seed(20261126)
    for rows in rows_list:
        scores = torch.randn([rows, width], generator=gen, device=device)
        # Coarse scores: exact ties inside the top-k and at its threshold.
        scores = scores.mul(8).round().div(8).float()
        ends = torch.randint(1, width + 1, [rows], generator=gen, device=device).to(torch.int32)
        ends[0] = width
        pos = (ends.to(torch.int64) * cr - 1).clamp(min=0)
        lens = (pos + 1).to(torch.int32)
        places = [{}]
        if limit > _TOPK_CLUSTER_MAX_ROWS:
            places = [_topk_placement(scores, False)]
            if rows in draft and _topk_placement(scores, True) != places[0]:
                places.append(_topk_placement(scores, True))
        for place in places:
            outs = []
            for cowrite in (False, True):
                topk = torch.full([rows, k], -7, dtype=torch.int32, device=device)
                sel = qsa_topk_expand_cluster(
                    scores, ends, pos, lens, topk, cr, core.token_topk, cowrite=cowrite, **place
                )
                outs.append((topk, sel))
            torch.cuda.synchronize(device)
            (t0, s0), (t1, s1) = outs
            where = f"{rows} rows" + (f" ({place})" if place else "")
            if not torch.equal(t0, t1):
                return f"topk_indices differ at {where}"
            if not torch.equal(s0, s1):
                return f"the token selection differs at {where}"
    return ""


def _dp_topk_cluster_rows(core) -> int:
    """`DP_TOPK_CLUSTER_ROWS_ENV` at this core's layout: the most rows a generation step's
    selection takes the cluster top-k at (`_TOPK_CLUSTER_MAX_ROWS` unset or at a layout without
    attention DP). A value outside the op's range raises."""
    raw = os.environ.get(DP_TOPK_CLUSTER_ROWS_ENV, "").strip()
    if not raw:
        return _TOPK_CLUSTER_MAX_ROWS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not _TOPK_CLUSTER_MAX_ROWS < value <= _TOPK_CLUSTER_PAIR_MAX_ROWS:
        raise ValueError(
            f"{DP_TOPK_CLUSTER_ROWS_ENV}={raw!r}: expected an integer in "
            f"{_TOPK_CLUSTER_MAX_ROWS + 1}..{_TOPK_CLUSTER_PAIR_MAX_ROWS}"
        )
    if core.dp_size == 1:
        logger.warning(
            f"{DP_TOPK_CLUSTER_ROWS_ENV} is set but ignored: it serves attention-DP ranks; this "
            f"rank has attention DP {core.dp_size}"
        )
        return _TOPK_CLUSTER_MAX_ROWS
    return value


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _topk_placement(scores, draft: bool) -> dict:
    """`qsa_topk_expand_cluster`'s placement knobs for a generation launch of `scores`' rows
    (`_TOPK_PLACEMENT`; empty where the table has no row). Host side, on shapes only."""
    rows = scores.shape[0]
    for lo, hi, solo, row_map in _TOPK_PLACEMENT["draft" if draft else "trunk"]:
        if lo <= rows <= hi:
            pairs = _sm_count(scores.device.index or 0) // 2 if row_map else 0
            return dict(solo_blocks=solo, pairs=pairs)
    return {}


def _tep4_layout(core) -> bool:
    """Whether this core is a tep4 rank: attention TP over `_TEP_RANKS` ranks without attention
    DP, the routed experts as `_TEP_RANKS` whole-expert windows."""
    return (
        not core.one_rank
        and core.dp_size == 1
        and core.attn_tp_size == _TEP_RANKS
        and core.mapping.world_size == _TEP_RANKS
        and core.local_experts * _TEP_RANKS == core.num_experts
    )


def _tep_fast_decode(core) -> bool:
    """`TEP_FAST_DECODE_ENV` at this core's layout: whether it is a tep4 rank with the switch on
    (default). Set to 1 at another layout: one warning, False."""
    on = _switch(TEP_FAST_DECODE_ENV, default=True)
    tep4 = _tep4_layout(core)
    if on and not tep4 and os.environ.get(TEP_FAST_DECODE_ENV, "").strip():
        logger.warning(
            f"{TEP_FAST_DECODE_ENV} is set but ignored: it serves a tep4 rank (attention TP "
            f"{_TEP_RANKS} without attention DP, {_TEP_RANKS} expert windows); this rank has world "
            f"size {core.mapping.world_size}, attention TP {core.attn_tp_size}, attention DP "
            f"{core.dp_size}, {core.local_experts} of {core.num_experts} experts"
        )
    return on and tep4


#: Off by default (opt-in: a same-precision reorder of the draft side). At a tep4 rank with MTP,
#: a generation-only draft step runs the draft layer as `DecodeTarget._mtp_layer`, the one-rank
#: decode form, over this rank's shard instead of the generic pass (`_mtp_front`, `_mtp_qkv`,
#: `_mtp_index_branch`, `_mtp_tail`):
#: - the front end's two projections, the fused `[q | k | v | gate | index | pad]` attention input
#:   (one GEMM over a second copy of the rank's draft q / k / v and index rows instead of four
#:   GEMMs and three copies), the output projection and the shared expert's gate|up on the
#:   decode entry (`core.decode_mm`) on a captured step; the embedding branch of the front end on
#:   the side stream beside the hidden branch;
#: - both hyper-connection sites as the trunk's decode sites (`_hc_site`: the rows pairs with
#:   `TEP_FAST_DECODE_ENV`), one norm + rope launch for the q / k and index planes;
#: - the sparse core on the multi-rank decode ops the generic pass runs (`_draft_sparse`), and a
#:   draft-extend step's continuing rows (`_draft_continue`): every row writes its K/V and index
#:   key, each request's continuing row alone runs the scores, the GQA, the output projection,
#:   the second site and the MoE;
#: - the draft MoE's shared expert on the side stream beside the routed window, and its
#:   combination and injection in one launch (`mtp_moe_combine_inject`).
#: The two partials the generic pass sums (the output projection's, the routed window's) are
#: summed the same way. Draft tokens can move at a near-tie (other GEMM kernels, the rows-pair
#: rounding at the sites); the target side reads none of this layer's output. Read once, when the
#: target is built; set at another layout or without MTP it warns and does nothing.
TEP_DRAFT_LAYER_ENV = "TRTLLM_QWEN4_EXP_TEP_DRAFT_LAYER"


def _tep_draft_layer(core) -> bool:
    """`TEP_DRAFT_LAYER_ENV` at this core's layout: whether a tep4 rank with MTP runs the draft
    layer as `DecodeTarget._mtp_layer`. Off by default; set where it cannot apply, one warning."""
    if not _switch(TEP_DRAFT_LAYER_ENV):
        return False
    if _tep4_layout(core) and core.mtp_enabled:
        return True
    logger.warning(
        f"{TEP_DRAFT_LAYER_ENV} is set but ignored: it serves a tep4 rank with MTP; this rank has "
        f"world size {core.mapping.world_size}, attention TP {core.attn_tp_size}, attention DP "
        f"{core.dp_size}, {core.local_experts} of {core.num_experts} experts, MTP "
        f"{'on' if core.mtp_enabled else 'off'}"
    )
    return False


#: Off by default (class S: another GEMM shape and another norm + rope kernel than the split
#: form, draft tokens only). At a dep4 / dep8 rank with MTP, a draft step of the generic pass forms
#: its attention input as one rank does (`DecodeTarget._mtp_attn_front`): one GEMM against the
#: packed `[q | k | v | gate | index | pad]` operand, one norm + rope launch for both planes, the
#: index key's store and compression on the side stream. It replaces four cuBLAS GEMMs on one
#: input, three copies, two norm + rope launches and a serial index branch. The packed operand is
#: a second copy of the draft layer's q, k, v and index weights (about 69 MiB a rank); the split
#: ones stay for the draft step with context rows. Measured on GB300 dep4 inside a step bundle
#: with the draft launch order (0.08-0.2 ms a step derived). Read once, when the target is built.
DP_MTP_ATTN_PACK_ENV = "TRTLLM_QWEN4_EXP_DP_MTP_ATTN_PACK"


def _dp_mtp_attn_pack(core) -> bool:
    """`DP_MTP_ATTN_PACK_ENV` at this core's layout: an attention-DP rank of a measured group size
    (dep4, dep8) with MTP. Set where it cannot apply: one warning, False."""
    if not _switch(DP_MTP_ATTN_PACK_ENV):
        return False
    if (
        not core.one_rank
        and core.dp_size in _DP_DECODE_SIZES
        and core.attn_tp_size == 1
        and core.mtp_enabled
    ):
        return True
    logger.warning(
        f"{DP_MTP_ATTN_PACK_ENV} is set but ignored: it serves a dep4 / dep8 rank with MTP; this "
        f"rank has attention DP {core.dp_size}, attention TP {core.attn_tp_size}, MTP "
        f"{'on' if core.mtp_enabled else 'off'}"
    )
    return False


#: Off by default (opt-in: the router logits come from another GEMM, a same-precision reorder). At a
#: tep4 rank whose trunk MoE takes the route front (`TEP_FAST_DECODE_ENV`), a captured block of a
#: row count the route front serves, up to `moe_decode_front.MAX_ROWS`, forms its router logits in
#: the route front's launch (`ops/moe/moe_decode_front__decode`: the 512-expert router GEMM on the
#: kernel's low-M warp MMA, then the top-10, the window's tables and the NVFP4 codes) instead of
#: `_decode_router` (FlashInfer's low-M GEMM) and then the route front: one launch fewer a block.
#: Given its logits the block is bitwise the route front's; a logit can differ from
#: `_decode_router`'s in its last bit, so a row whose 10th and 11th logits are that close can pick
#: the other expert. Every rank must pick alike (each (row, slot) is served by exactly one
#: window): the logits are the same bits on every rank (same rows, same kernel), and the row
#: counts are agreed over the group at build (`_tep_decode_front_build`, collective). Read once,
#: when the target is built; set where it cannot apply it warns and does nothing.
TEP_MOE_FRONT_ENV = "TRTLLM_QWEN4_EXP_TEP_MOE_FRONT"


#: Off by default (opt-in: other GEMM kernels at rows that ran on cuBLAS, a same-precision
#: reorder). At a tep4 rank with its fast forms (`TEP_FAST_DECODE_ENV`), a generation-only step's
#: bf16 projections take FlashInfer's split-K low-M kernel with PDL (`low_m_bf16_mm`'s `wide`
#: rows) where they ran on cuBLAS:
#: - the attention input, the GDN input and the attention o / GDN out projections at 9-16 rows
#:   (`wide` 1: the tactics measured at 16 rows; o / GDN out stay on cuBLAS past 16, where it is
#:   faster);
#: - the PLE key and value projections at 9-16 rows, and the trunk shared expert's gate|up and
#:   down at up to 16 rows (`wide` 2: split-K's default tactic where it takes the shape).
#: Every branch accumulates in fp32 and rounds once; the sums run in another order. Read once,
#: when the target is built; set at another layout it warns and does nothing.
TEP_PROJ_ENV = "TRTLLM_QWEN4_EXP_TEP_PROJ_SPLITK"
#: Rows from which the PLE projections take the split-K kernel under `TEP_PROJ_ENV` (measured at
#: 12 and 16; fewer rows keep cuBLAS).
_TEP_PLE_WIDE_MIN_ROWS = 9  # capacity: fixed: a dispatch threshold


#: Off by default (opt-in: another kernel of the same precision for the rows the shared split-K
#: serves; to be decided by an A/B against it). At a tep4 rank with its fast forms on SM103, the
#: gated sparse GQA of a decode or verify step of up to `_TEP_SPLITK_MAX_ROWS`' rows over the
#: e4m3 pool -- the trunk's and the draft layer's -- runs `qsa_paged_sparse_gqa_e4m3_decode`
#: instead of `qsa_sparse_gqa_splitk_cute`: half-height warp MMAs (the shard's 6 query heads fill
#: rows 0..7), K and V fragments built from the raw e4m3 rows with an exact widen, q and P in
#: bf16, fp32 accumulation, an online softmax over a split's 64-token tiles, then the bf16 decode
#: entry's merge with the output gate. The output moves by an ulp on some elements against the
#: split-K kernel. Read once, when the target is built; set where it cannot apply it warns and
#: does nothing.
TEP_GQA_E4M3_ENV = "TRTLLM_QWEN4_EXP_TEP_GQA_E4M3"
#: On by default (bitwise; a row table). At a tep4 rank with its fast forms on SM103, a sparse
#: trunk decode or verify step of up to `_TEP_ATTN_INPLACE_MAX_ROWS` rows over the e4m3 pool runs
#: the family's one-launch prologue in place on the projection row (`one_rank_attn_decode_prologue`,
#: the one-rank form) instead of the target's packed one: no packed `[q | k | v]` copy, the
#: sparse GQA reading the query heads as views of the projection. The same arithmetic in both
#: launches (the shared norm + rope tile and launch plan, the same e4m3 conversion of the same
#: bf16 values, the same compression); the build checks the bytes on this rank's geometry and
#: keeps the packed form where they differ. Read once, when the target is built.
TEP_ATTN_INPLACE_ENV = "TRTLLM_QWEN4_EXP_TEP_ATTN_INPLACE"
#: Rows up to which `TEP_ATTN_INPLACE_ENV` engages: one request of an MTP-3 verify step, or up to
#: four decode rows (the 4-rank micro: 1.06 us a layer saved at 4 rows, nothing at 8).
_TEP_ATTN_INPLACE_MAX_ROWS = 4  # capacity: fixed: a dispatch threshold
#: Off by default (opt-in, bitwise). At a tep4 rank with its draft
#: layer on the multi-rank decode ops (`TEP_DRAFT_LAYER_ENV`) and its fast forms on SM103, a draft
#: pass over the e4m3 pool (not an IndexShare reuse) runs the draft layer's norm + rope, K/V
#: append, raw index-key store and group compression as the trunk does, in one launch
#: (`one_rank_attn_decode_prologue`, in place on the projection row, `p` rows a request) instead
#: of the four launches over two streams (`qwen4_exp_qk_norm_rope`, then `qsa_paged_index_store`
#: and `qsa_prefill_compress` on the side stream, the join, `qsa_paged_kv_store`). The prologue's
#: own contract is to write those four entries' bytes; the build checks it on this rank's
#: geometry (`_tep_draft_prologue_check`) and keeps the four launches where anything differs.
#: GB300 tep4 one-node server (forced acceptance 1.32, ~100k-token contexts, alternating on / off
#: arms): -0.009 / -0.009 / -0.014 / -0.014 ms a step at 1 / 2 / 3 / 4 requests (9 launches a step
#: fewer); texts identical.
#: Read once, when the target is built.
TEP_DRAFT_PROLOGUE_ENV = "TRTLLM_QWEN4_EXP_TEP_DRAFT_PROLOGUE"


def _tep_draft_prologue_build(core, device) -> bool:
    """`TEP_DRAFT_PROLOGUE_ENV`, read once at build: True at a tep4 rank with its draft layer and
    fast forms on SM103 whose self-check passes; False elsewhere (one warning where set)."""
    if not _switch(TEP_DRAFT_PROLOGUE_ENV):
        return False
    if not (
        core.tep_fast
        and getattr(core, "tep_draft_layer", False)
        and torch.cuda.get_device_capability(device) == archs.SM_103
    ):
        logger.warning(
            f"{TEP_DRAFT_PROLOGUE_ENV} is set but ignored: it serves a tep4 rank with "
            f"{TEP_FAST_DECODE_ENV} and {TEP_DRAFT_LAYER_ENV} on, on SM103"
        )
        return False
    try:
        bad = _tep_draft_prologue_check(core, device)
    except Exception as e:  # noqa: BLE001 -- an opt-in form declines rather than stop the load
        bad = [f"{type(e).__name__}: {e}"]
    if bad:
        logger.warning(
            f"{TEP_DRAFT_PROLOGUE_ENV}: the one-launch draft prologue is not bitwise the four "
            f"launches at tp rank {core.mapping.tp_rank} ({'; '.join(bad[:4])}); the rank keeps "
            "the four launches"
        )
        return False
    logger.info(
        f"tep4 one-launch draft prologue ({TEP_DRAFT_PROLOGUE_ENV}=1): tp rank "
        f"{core.mapping.tp_rank}, draft passes over an e4m3 pool"
    )
    return True


def _tep_draft_prologue_check(core, device) -> list[str]:
    """Run the draft layer's four launches and the one-launch prologue on one synthetic input of
    this rank's geometry (an e4m3 K/V pool as the engine lays it out, a bf16 index plane): a
    draft-extend pass of two requests of four rows whose rows end a group with earlier members in
    the launch, and four decode rows, one a request, one a group end. Every position lies inside
    its request's four table pages (a row past its table writes nothing in either form, but the
    two forms differ on what an unmapped page receives). Returns what differs: the K/V pool, the
    index plane, the position cache, the index query or the q / k heads."""
    tpb, width, pages = 32, 4, 40
    hq, hk, dim = core.heads_q, core.heads_kv, core.head_dim
    g = torch.Generator(device=device).manual_seed(4711)
    bad = []
    # Request 0's group ends on a page's last slot with its three earlier members in the launch;
    # request 1's group ends at its third row (two earlier members in the launch, one cached).
    cases = ((4, (28, 29, 30, 31, 113, 114, 115, 116)), (1, (95, 33, 127, 70)))
    for rows_per_request, pos in cases:
        rows = len(pos)
        request = torch.tensor(
            [r // rows_per_request for r in range(rows)], dtype=torch.int32, device=device
        )
        logical = torch.tensor(pos, dtype=torch.int64, device=device)
        requests = rows // rows_per_request
        block_table = torch.randperm(pages, generator=g, device=device).to(torch.int32)
        block_table = block_table[: requests * width].reshape(requests, width).contiguous()
        ids = torch.stack([logical, logical + 3, logical + 7]).to(torch.int32).contiguous()
        coords = ids.t()
        angles = torch.rand(max(pos) + 16, core.rotary_dim // 2, generator=g, device=device) * 6.28
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
        weights = [
            (0.3 * torch.randn(n, generator=g, device=device)).to(torch.bfloat16)
            for n in (dim, dim, core.index_head_dim, core.index_head_dim)
        ]
        proj = (2.0 * torch.randn(rows, core.attn_in_width, generator=g, device=device)).to(
            torch.bfloat16
        )
        pool = torch.randn(pages, 2, hk, tpb, dim, generator=g, device=device).to(
            torch.float8_e4m3fn
        )
        index_cache = torch.randn(pages, tpb, 1, core.index_head_dim, generator=g, device=device)
        index_cache = index_cache.to(torch.bfloat16)
        position_cache = torch.randint(
            0, max(pos), (pages, tpb, 3), generator=g, device=device, dtype=torch.int32
        )
        runs = []
        for fused in (False, True):
            pool_r, index_r, position_r = pool.clone(), index_cache.clone(), position_cache.clone()
            proj_r = proj.clone()
            index_q = torch.empty(
                rows, core.index_heads, core.index_head_dim, dtype=torch.bfloat16, device=device
            )
            if fused:
                one_rank_attn_decode_prologue(
                    proj_r, index_q, ids, coords, cos_sin, *weights, pool_r[:, 0], pool_r[:, 1],
                    index_r, position_r, request, logical, block_table, tpb, core.compress_ratio,
                    num_heads_q=hq, num_heads_k=hk, head_dim=dim, index_heads=core.index_heads,
                    index_head_dim=core.index_head_dim, index_q_col=core.index_col,
                    rotary_dim=core.rotary_dim, eps=core.eps, mrope_section1=core.mrope_section1,
                    mrope_section2=core.mrope_section2, index_mrope_section1=0,
                    index_mrope_section2=0, compress_mrope_section=_QSA_MROPE_SECTION,
                    rows_per_request=rows_per_request,
                )  # fmt: skip
            else:
                qwen4_exp_qk_norm_rope(
                    proj_r, index_q, ids, cos_sin, weights[0], weights[1], weights[2],
                    num_heads_q=hq, num_heads_k=hk, head_dim=dim, index_heads=core.index_heads,
                    index_head_dim=core.index_head_dim, index_q_col=core.index_col,
                    rotary_dim=core.rotary_dim, eps=core.eps, mrope_section1=core.mrope_section1,
                    mrope_section2=core.mrope_section2, index_mrope_section1=0,
                    index_mrope_section2=0,
                )  # fmt: skip
                token_k = torch.split(
                    torch.reshape(
                        proj_r[:, core.index_col :],
                        [
                            rows,
                            core.index_scratch_width // core.index_head_dim,
                            core.index_head_dim,
                        ],
                    ),
                    [core.index_heads, core.index_kv_heads, 1],
                    1,
                )[1]
                qsa_paged_index_store(
                    token_k, coords, index_r, position_r, request, logical, block_table, tpb
                )
                qsa_prefill_compress(
                    logical_positions=logical, request_indices=request, block_table=block_table,
                    index_cache=index_r, position_cache=position_r, k_norm_weight=weights[3],
                    cos_sin=cos_sin, eps=core.eps, tokens_per_block=tpb,
                    compress_ratio=core.compress_ratio, mrope_section=_QSA_MROPE_SECTION,
                )  # fmt: skip
                _, k_dst, v_dst = _attn_in_heads(core, proj_r)
                qsa_paged_kv_store(
                    k_dst, v_dst, pool_r[:, 0], pool_r[:, 1], request, logical, block_table, tpb
                )
            runs.append((pool_r, index_r, position_r, index_q, proj_r[:, : (hq + hk) * dim]))
        torch.cuda.synchronize(device)
        names = ("K/V pool", "index plane", "position cache", "index query", "q/k heads")
        for name, a, b in zip(names, *runs):
            if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                bad.append(f"{name} at {rows} rows, {rows_per_request} a request")
    return bad


def _tep_gqa_e4m3(core, device) -> bool:
    """`TEP_GQA_E4M3_ENV` at this core: whether a tep4 rank with its fast forms on SM103 sends its
    decode GQA to the e4m3 entry. Set where it cannot apply, one warning."""
    if not _switch(TEP_GQA_E4M3_ENV):
        return False
    if (
        core.tep_fast
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == archs.SM_103
    ):
        return True
    logger.warning(
        f"{TEP_GQA_E4M3_ENV} is set but ignored: it serves a tep4 rank with "
        f"{TEP_FAST_DECODE_ENV} on, on SM103"
    )
    return False


def _tep_attn_inplace_build(core, device) -> bool:
    """`TEP_ATTN_INPLACE_ENV`, read once at build: True at a tep4 rank with its fast forms on SM103
    whose self-check passes (`_tep_attn_inplace_check`); False elsewhere, silently, and where the
    check fails or does not run (one warning; the rank keeps the packed prologue, which needs no
    other rank to agree: the two forms write the same bytes)."""
    if not (
        _switch(TEP_ATTN_INPLACE_ENV, default=True)
        and core.tep_fast
        and torch.cuda.get_device_capability(device) == archs.SM_103
    ):
        return False
    try:
        bad = _tep_attn_inplace_check(core, device)
    except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
        bad = [f"{type(e).__name__}: {e}"]
    if bad:
        logger.warning(
            f"{TEP_ATTN_INPLACE_ENV}: the in-place prologue is not bitwise the packed one at "
            f"tp rank {core.mapping.tp_rank} ({'; '.join(bad[:4])}); the rank keeps the packed prologue"
        )
        return False
    logger.info(
        f"tep4 in-place attention prologue: tp rank {core.mapping.tp_rank}, sparse trunk steps up to "
        f"{_TEP_ATTN_INPLACE_MAX_ROWS} rows over an e4m3 pool"
    )
    return True


def _tep_attn_inplace_check(core, device) -> list[str]:
    """Run both prologues on one synthetic input of this rank's geometry (an e4m3 K/V pool as the
    engine lays it out, a bf16 index plane): an MTP-3 verify request whose rows end a group with
    its three earlier members in the launch, and four decode rows, one a group end. Returns what
    differs: the K/V pool, the index plane, the position cache or the index query bytes, or the
    in-place q / k heads against the packed copy (v is not written in place)."""
    tpb, width, pages = 32, 4, 40
    hq, hk, dim = core.heads_q, core.heads_kv, core.head_dim
    g = torch.Generator(device=device).manual_seed(2311)
    bad = []
    for rows_per_request, pos in ((4, (60, 61, 62, 63)), (1, (95, 33, 127, 70))):
        rows = len(pos)
        request = torch.tensor(
            [r // rows_per_request for r in range(rows)], dtype=torch.int32, device=device
        )
        logical = torch.tensor(pos, dtype=torch.int64, device=device)
        requests = rows // rows_per_request
        block_table = torch.randperm(pages, generator=g, device=device).to(torch.int32)
        block_table = block_table[: requests * width].reshape(requests, width).contiguous()
        ids = torch.stack([logical, logical + 3, logical + 7]).to(torch.int32).contiguous()
        coords = ids.t()
        angles = torch.rand(max(pos) + 16, core.rotary_dim // 2, generator=g, device=device) * 6.28
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
        weights = [
            (0.3 * torch.randn(n, generator=g, device=device)).to(torch.bfloat16)
            for n in (dim, dim, core.index_head_dim, core.index_head_dim)
        ]
        proj = (2.0 * torch.randn(rows, core.attn_in_width, generator=g, device=device)).to(
            torch.bfloat16
        )
        pool = torch.randn(pages, 2, hk, tpb, dim, generator=g, device=device).to(
            torch.float8_e4m3fn
        )
        index_cache = torch.randn(pages, tpb, 1, core.index_head_dim, generator=g, device=device)
        index_cache = index_cache.to(torch.bfloat16)
        position_cache = torch.randint(
            0, max(pos), (pages, tpb, 3), generator=g, device=device, dtype=torch.int32
        )
        runs = []
        for inplace in (False, True):
            pool_r, index_r, position_r = pool.clone(), index_cache.clone(), position_cache.clone()
            proj_r = proj.clone()
            index_q = torch.empty(
                rows, core.index_heads, core.index_head_dim, dtype=torch.bfloat16, device=device
            )
            shape = (block_table, tpb, core.compress_ratio)
            common = (ids, coords, cos_sin, *weights, pool_r[:, 0], pool_r[:, 1], index_r)
            if inplace:
                one_rank_attn_decode_prologue(
                    proj_r, index_q, *common, position_r, request, logical, *shape,
                    num_heads_q=hq, num_heads_k=hk, head_dim=dim, index_heads=core.index_heads,
                    index_head_dim=core.index_head_dim, index_q_col=core.index_col,
                    rotary_dim=core.rotary_dim, eps=core.eps, mrope_section1=core.mrope_section1,
                    mrope_section2=core.mrope_section2, index_mrope_section1=0,
                    index_mrope_section2=0, compress_mrope_section=_QSA_MROPE_SECTION,
                    rows_per_request=rows_per_request,
                )  # fmt: skip
                heads = proj_r[:, : core.qkv_width]
            else:
                qkv = torch.empty(rows, core.qkv_width, dtype=torch.bfloat16, device=device)
                qwen4_exp_attn_decode_prologue(
                    proj_r, qkv, index_q, *common, position_r, request, logical, *shape,
                    hq, hk, dim, core.index_heads, core.index_head_dim, core.index_col,
                    core.rotary_dim, core.eps, core.mrope_section1, core.mrope_section2,
                    rows_per_request=rows_per_request,
                )  # fmt: skip
                heads = qkv
            runs.append((pool_r, index_r, position_r, index_q, heads[:, : (hq + hk) * dim]))
        torch.cuda.synchronize(device)
        names = ("K/V pool", "index plane", "position cache", "index query", "q/k heads")
        for name, a, b in zip(names, *runs):
            if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                bad.append(f"{name} at {rows} rows, {rows_per_request} a request")
    return bad


def _tep_index_prefetch(core) -> bool:
    """`TEP_INDEX_PREFETCH_ENV` at this core: whether a tep4 rank with its fast forms prefetches
    its verify steps' index keys. Set where it cannot apply, one warning."""
    if not _switch(TEP_INDEX_PREFETCH_ENV):
        return False
    if core.tep_fast and _INDEX_SCORES_PDL:
        return True
    logger.warning(
        f"{TEP_INDEX_PREFETCH_ENV} is set but ignored: it serves a tep4 rank with "
        f"{TEP_FAST_DECODE_ENV} on and PDL on (TRTLLM_ENABLE_PDL)"
    )
    return False


def _tep_proj(core) -> bool:
    """`TEP_PROJ_ENV` at this core: whether a tep4 rank with its fast forms takes the wide rows.
    Set where it cannot apply, one warning."""
    if not _switch(TEP_PROJ_ENV):
        return False
    if core.tep_fast:
        return True
    logger.warning(
        f"{TEP_PROJ_ENV} is set but ignored: it serves a tep4 rank with {TEP_FAST_DECODE_ENV} on"
    )
    return False


#: Off by default (a Lamport transport: race-risk, opt-in). At a tep4 rank whose trunk MoE takes
#: the route front (`TEP_FAST_DECODE_ENV`), each captured block of up to
#: `tep_finalize_allreduce.MAX_ROWS` rows enters the runner without its finalize, and one launch
#: (`ops/moe/tep_finalize_allreduce`) forms the finalize and the TP all-reduce over symmetric
#: memory instead of the runner's finalize and `core.tp_sum`: the finalize's slot-order fp32
#: expression, then the one-shot sum in fp32 in rank order, one bf16 rounding each; a -0.0 lane
#: of a rank's partial travels as +0.0. Its slot is the block's layer index mod 3, baked into
#: the captured graph. `1` releases the launch's dependents after its publish, `early` right
#: after its PDL wait. The build checks every row count against the reference on every rank and
#: raises on every rank if one fails. Read once, when the target is built; set where it cannot
#: apply (another layout, no route front, a group across nodes) it warns and does nothing.
TEP_FINAR_ENV = "TRTLLM_QWEN4_EXP_TEP_FINAR"


#: Off by default (opt-in: a same-precision reorder of the shared expert, on the Lamport transport
#: of `TEP_FINAR_ENV`, which it needs). At the fused finalize all-reduce's row counts in
#: `_TEP_FINAR_SHARED_ROWS`, a tep4 rank runs its QUARTER of the trunk shared expert instead of the
#: whole replicated one: the gate|up rows and the gate-logit row of a quarter of the intermediate
#: (one GEMM), SiLU-and-mul, and the matching quarter of the down projection with an fp32 output,
#: on the side stream as before. The fused launch adds sigmoid(gate logit) x that quarter to the
#: rank's fp32 partial before its one rounding, so the four quarters summed over the group are the
#: layer's gated shared expert, and the block hands the next site a zero shared expert and gate
#: (`routed + sigmoid(0) x 0` is `routed`, bitwise). Another work split (the down projection's K
#: over the ranks, fp32 partials, one rounding a rank, an `ex2.approx` sigmoid) where the stock
#: chain rounds the shared expert to bf16 first. About 2.5 MiB a MoE layer for the quarters. Read
#: once, when the target is built; set where it cannot apply it warns and does nothing.
TEP_FINAR_SHARED_ENV = "TRTLLM_QWEN4_EXP_TEP_FINAR_SHARED"
#: Row counts that take the quarter (3 and 4 requests of four verify rows): GB300 tep4 against the
#: whole shared expert on the side stream, -0.014 / -0.038 / -0.015 ms a step at 3 / 4 / 5
#: requests; measured slower at 8 rows and at 20-32.
_TEP_FINAR_SHARED_ROWS = (12, 16)  # capacity: fixed: a dispatch table
#: The quarters a shared expert's intermediate splits into: the group's ranks.
_TEP_FINAR_QUARTERS = 4  # capacity: fixed: the tep4 group


def _tep_finar_mode() -> str:
    """`TEP_FINAR_ENV`: "" (unset, empty, 0), "publish" (1) or "early"; ValueError otherwise."""
    raw = os.environ.get(TEP_FINAR_ENV, "")
    value = raw.strip().lower()
    if value in ("", "0"):
        return ""
    if value in ("1", "early"):
        return "publish" if value == "1" else "early"
    raise ValueError(f"{TEP_FINAR_ENV}={raw!r}: expected 0, 1 or early")


#: Switch, on by default: at a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) the hyper-connection
#: mixers of a decode or short prefill step take row-count plans of their two tcgen05 kernels
#: that sum the same K slices in the same order as the 128-row plans, so every output is bitwise:
#: - `qwen4_exp_hc_decode_tc_first` on a 64-row M tile at `_HC_TILE64_ROWS` rows (half the
#:   normed bytes a CTA streams), and for a mixer with the injection the rows-on-N kernel (the
#:   weight rows on a 64-row MMA M, the normed rows on the MMA N in 16-row steps) at
#:   `_HC_ROWSN_ROWS` rows, where tcgen05 has no M tile between 64 and 128;
#: - `qwen4_exp_hc_up_gate_mean` on one 64-row M tile at `_HC_TILE64_ROWS` rows, and on two at
#:   `_HC_UP2_ROWS` rows where it does not release its dependents early (`DP_INPROJ_RELEASE_ENV`:
#:   two tiles are slower there).
#: GB300 micros, a site: the first kernel 5.1-5.6 -> 4.4-4.9 us at 8-64 rows, the rows-on-N
#: kernel -0.55..-0.60 us at 65..80 rows and -0.14..-0.23 at 81..96, the up kernel on two tiles
#: -0.45..-0.48 us at 105..128 rows; dep4 steps at 17..24 requests a rank -0.02..-0.06 ms.
#: `0` keeps the 128-row plans. Read once, when the target is built.
DP_HC_TILES_ENV = "TRTLLM_QWEN4_EXP_DP_HC_TILES"
#: Rows at which both kernels take one 64-row tile: up to 4 rows the first kernel keeps its
#: 10-way split.
_HC_TILE64_ROWS = (5, _HC_TILE64_MAX_ROWS)  # capacity: fixed: dispatch thresholds
#: Rows at which an up kernel that does not release early takes two 64-row tiles.
_HC_UP2_ROWS = (65, 128)  # capacity: fixed: dispatch thresholds
#: Off by default (class S: another K split). With `DP_HC_TILES_ENV` engaged, a mixer with the
#: injection at `_HC_FIRST_X2_ROWS` rows runs its first kernel on two 64-row M tiles with an
#: 8-way K split (`qwen4_exp_hc_decode_tc_first(..., tm64x2=True)`) instead of the rows-on-N
#: kernel, which is about 1 us slower than a 64-row tile at equal rows. `low` moves within one
#: bf16 step; `inject` measured bitwise. GB300 micro, us a site: -0.20..-0.51 at 68..96 rows;
#: dep8 server step at 20 / 24 requests a rank: -0.038 / -0.028 ms. Read once, when the target
#: is built.
DP_HC_FIRST_X2_ENV = "TRTLLM_QWEN4_EXP_DP_HC_FIRST_X2"
#: Rows the switch above serves: the rows-on-N kernel's.
_HC_FIRST_X2_ROWS = _HC_ROWSN_ROWS  # capacity: fixed: dispatch thresholds
#: Switch, on by default: at a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) with PDL, trunk site
#: A of a decode step (the mixer in front of a GDN or attention block) releases its dependents
#: before its own wait (`qwen4_exp_hc_up_gate_mean(release_early=True)`), and the block's bf16
#: input projection runs on `mid_rows_mm`'s bf16 tcgen05 GEMM over the checkpoint's `[N, K]`
#: weight (a view, no copy) at the plan `_INPROJ_RELEASE_PLANS` names for the weight geometry and row count. That
#: GEMM is launched with PDL, streams its weight before its wait and reads `mixed` after it, so
#: it is the launch that runs beside the released up kernel (the stock entry, `low_m_bf16_mm`,
#: runs cuBLAS without PDL at these rows, and a release in front of it moves nothing). It
#: engages at the row counts of the CUDA-graph batch sizes inside `_INPROJ_RELEASE_ROWS`, for
#: which the GEMM is compiled at build. The build compares the GEMM with the stock entry on this
#: rank's weights at every compiled row count and keeps the switch only when every element is
#: equal (fp32 accumulation of K in order and one rounding on both sides at these shapes; GB300:
#: 0 of 2.1e7..1.7e8 elements a row count differ at 28..224 rows on the checkpoint's weights).
#: GB300 dep4 trunk-loop micro, with the GDN forms of `DP_GDN_LEAN_ENV`: -0.47..-0.49 ms a step at
#: 7..56 requests a rank. `0` keeps the stock entry and the up kernel's end-of-kernel release.
#: Read once, when the target is built.
DP_INPROJ_RELEASE_ENV = "TRTLLM_QWEN4_EXP_DP_INPROJ_RELEASE"
#: Rows a rank at which the switch above can engage (`mid_rows_mm`'s rows past the GEMV's).
_INPROJ_RELEASE_ROWS = (17, 256)  # capacity: fixed: dispatch thresholds
#: Off by default. `<r>` (2..4 requests a rank): the bitwise attention-DP decode forms below also
#: serve r..4 requests (4 r..16 rows; from 5 rows the hc first kernel's default plan has the K
#: split the rows-on-N kernel keeps): the input projections behind a releasing site A
#: (`DP_INPROJ_RELEASE_ENV`, its first plan; its build self-check still gates every row count),
#: the rows-on-N hc first kernel for a mixer with the injection (`DP_HC_TILES_ENV`, in place of
#: the 64-row tile; the same K slices in the same order), the lean verify conv
#: (`DP_GDN_CONV_LEAN_ENV`) and the conv as the input projection's epilogue
#: (`DP_GDN_INPROJ_CONV_BF16_ENV`) and the norm in the replay (`DP_GDN_REPLAY_NORM_BF16_ENV`, where
#: the launch mapping gives a CTA whole v heads): the same kernels at fewer rows. Measured on dep8
#: at 3 requests a rank with r = 2, as part of a -0.33 ms step. Read once, when the target is
#: built.
DP_LOWROWS_ENV = "TRTLLM_QWEN4_EXP_DP_LOWROWS"
#: The most rows the lowered floors reach below their own ranges: 4 requests a rank.
_DP_LOWROWS_MAX_ROWS = 16  # capacity: fixed: a dispatch threshold


def _dp_lowrows() -> int | None:
    """`DP_LOWROWS_ENV`: the lowest request count a rank, or None (unset, empty, off)."""
    raw = os.environ.get(DP_LOWROWS_ENV, "")
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return None
    if value.isdigit() and 2 <= int(value) <= 4:
        return int(value)
    raise ValueError(f"{DP_LOWROWS_ENV}={raw!r}: expected 0 or 2..4 requests a rank")


def _dp_lowrows_floor(base: int) -> int:
    """`base` lowered to `DP_LOWROWS_ENV`'s rows (4 a request), or `base` where it is off."""
    r = _dp_lowrows()
    return base if r is None else min(base, 4 * r)


#: Measured plans of the bf16 GEMM behind a releasing site A (GB300 dep4 trunk-loop micro at 28 /
#: 48 / 96 / 120 / 224 rows, 13 plan sets): weight geometry `(N, K)` -> `((largest rows, MMA N
#: rule, Plan(ks, kpb, stages, l2, epi)), ...)`. MMA N rule "op": `mid_rows_mm`'s (32 / 64 / 128 /
#: 256); "fine": `max(32, rows rounded up to 16)`. Up to 64 rows 8 k-blocks a stage and 2
#: k-tiles of L2 prefetch; 65..96 rows the fine N (80 / 96 instead of the op's 128, which pads
#: 25-47% of every MMA and ring stage with zero rows) with an 8-deep ring; 97..128 rows the op's
#: ring with 4 k-tiles of prefetch; above, the fine N with the deepest ring. The activation rows
#: sit on the MMA N, so the fine N only changes how many zero rows ride along. No plan splits K.
#: GB300 dep8 server step at 18 / 24 requests a rank (72 / 96 rows), fine N against N 128:
#: -0.063 / -0.052 ms. A geometry not listed does not engage.
_INPROJ_RELEASE_PLANS: dict = {
    # GDN in_proj (q|k|v|z|a|b)
    (16480, 2560): (
        (64, "op", MidPlan(1, 8, 5, 2, 16)),
        (96, "fine", MidPlan(1, 4, 8, 4, 16)),
        (128, "op", MidPlan(1, 4, 6, 4, 16)),
        (256, "fine", MidPlan(1, 4, 5, 1, 16)),
    ),
    # attention in-proj (q|gate|k|v|index)
    (14080, 2560): (
        (64, "op", MidPlan(1, 8, 5, 2, 16)),
        (96, "fine", MidPlan(1, 4, 8, 4, 16)),
        (128, "op", MidPlan(1, 4, 6, 4, 16)),
        (256, "fine", MidPlan(1, 4, 5, 1, 16)),
    ),
}


#: Switch, on by default: at a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) whose verify steps
#: run the CuTe replay over a bf16 recurrent pool (no FlashInfer ring) with 4 rows a request and
#: work items from 16 requests (the layout the table was timed at), a verify step of
#: `_GDN_REPLAY_MAP_REQUESTS` requests launches the replay on the mapping `_gdn_replay_map`
#: names instead of the kernel's own (`default_config`): its own overflows one wave of resident
#: CTAs (608 = 152 SMs x 4) at 7-8 requests (96 CTAs a request) and at 13-15 (48 a request).
#: Every launch mapping of the kernel computes each state tile with the same statements, so the
#: output, the history and the commits are bitwise. GB300 micro, ms a step saved over the
#: kernel's own mapping: 0.020-0.028 at 5-6 requests, 0.064-0.071 at 7-10, 0.025-0.035 at
#: 11-12, 0.065-0.072 at 13-15, 0.003-0.024 at 17-32, 0.017-0.028 at 33-48. `0` keeps the
#: kernel's own mapping. Read once, at each cache-manager bind.
DP_GDN_REPLAY_MAP_ENV = "TRTLLM_QWEN4_EXP_DP_GDN_REPLAY_MAP"
#: Requests a rank the table names a mapping for (the largest timed: 48).
_GDN_REPLAY_MAP_REQUESTS = (5, 48)  # capacity: fixed: dispatch thresholds
#: 8 m-tiles a CTA on 8 warps, every state tile by bulk copy as soon as the slot is known.
_GDN_REPLAY_MAP_A = dict(tpc=8, warps=8, pf=0, minb=4, early_acc=True)
#: At 13..15 requests (48 CTAs a request): 4 warps, five CTAs resident an SM.
_GDN_REPLAY_MAP_B = dict(tpc=8, warps=4, pf=1, minb=5)
_GDN_REPLAY_MAP_B_REQUESTS = (13, 15)  # capacity: fixed: dispatch thresholds
#: The partition threshold and rows a request the table was timed at; 16 requests (the first
#: partitioned count) keep the kernel's own mapping, the best there.
_GDN_REPLAY_MAP_PARTITION, _GDN_REPLAY_MAP_ROWS = 16, 4  # capacity: fixed: the measured layout


def _gdn_replay_map(n: int) -> dict | None:
    """`DP_GDN_REPLAY_MAP_ENV`'s launch mapping of a verify step of `n` requests, or None (the
    kernel's own)."""
    lo, hi = _GDN_REPLAY_MAP_REQUESTS
    if not lo <= n <= hi or n == _GDN_REPLAY_MAP_PARTITION:
        return None
    lo_b, hi_b = _GDN_REPLAY_MAP_B_REQUESTS
    return dict(_GDN_REPLAY_MAP_B if lo_b <= n <= hi_b else _GDN_REPLAY_MAP_A)


def _gdn_replay_map_on(core, cm, device, ring: bool) -> bool:
    """`DP_GDN_REPLAY_MAP_ENV` against a bound manager: an attention-DP rank on SM103, the CuTe
    replay over a bf16 pool (not FlashInfer's ring), the measured partition threshold and rows
    a request."""
    return (
        _switch(DP_GDN_REPLAY_MAP_ENV, default=True)
        and _dp_decode_rank(core, device)
        and not ring
        and (CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, core.max_verify_rows)
        == (_GDN_REPLAY_MAP_PARTITION, _GDN_REPLAY_MAP_ROWS)
    )


#: Switch, on by default: at a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) with MTP and PDL,
#: whose bound manager runs the CuTe replay over a bf16 pool (no FlashInfer ring) with the stock
#: `GDNReplayState` promoting linear draft chains from a stacked conv pool, a generation-only
#: verify step of `_GDN_LEAN_ROWS` rows runs every GDN layer's conv on the lean conv
#: (`gdn_conv_lean`: the conv output and the raw q|k|v rows of every verify row into `xsave`, no
#: window snapshots, no pool advance; a PDL dependent of the input projection), promotes the
#: windows from those rows (`gdn_promote_x`, in place of the stock promotion, same bookkeeping
#: arithmetic over the promotion's replay slots) and launches the gated norm as a PDL dependent
#: of the replay (`rms_norm_gated_token_major(pdl=True)`; the registered op drops PDL once its
#: grid covers the SMs, which every dep verify step's does). Values: the conv output, the
#: promoted windows, the bookkeeping and the norm output are bitwise the stock launches'.
#:
#: Decided in `DecodeTarget.forward` from host ints that a decode graph's key fixes (rows,
#: rows a request, requests), consumed by the wrapped `GDNReplayState` promotion, which the spec
#: worker runs inside the same forward (`update_mamba_states`): a captured graph records the lean
#: conv and `promote_x` together, a replay re-runs both. A step is lean for all its GDN layers or
#: for none; mixed and context steps keep the stock pair. The bind compiles both kernels and runs
#: them against the stock pair on private buffers with this rank's taps (rank-local; a
#: difference or an error leaves the switch off at this rank with one warning). GB300 micro
#: 0.05-0.10 ms a step; measured on every dep4 / dep8 front point. `0` keeps the stock pair.
#: Read once, when the target is built.
DP_GDN_CONV_LEAN_ENV = "TRTLLM_QWEN4_EXP_DP_GDN_CONV_LEAN"
#: Rows a rank at which a generation-only verify step runs lean: the input projection's
#: `mid_rows_mm` rows (`DP_LOWROWS_ENV` lowers the floor).
_GDN_LEAN_ROWS = (17, 256)  # capacity: fixed: dispatch thresholds
#: The attribute marking a `GDNReplayState` whose promotion a core wrapped (binds repeat; a
#: state is wrapped once).
_GDN_LEAN_WRAPPED = "_qwen4_exp_gdn_lean_core"
#: `(what, count, capturing)` shapes a lean form has reported in this process (one line each).
_GDN_LEAN_LOGGED: set = set()


class _GdnStep:
    """What the lean GDN forms decided for the trunk step in flight, host side: `lean` (every
    GDN layer runs `gdn_conv_lean`; the promotion must be `gdn_promote_x`) and the generation
    requests it carried. Both trunk targets clear it before they decide; the wrapped promotion,
    the last launch of the step's state update, clears `lean` again."""

    __slots__ = ("lean", "gens", "inconv", "rnorm")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.lean = False
        self.gens = 0
        # `DP_GDN_INPROJ_CONV_BF16_ENV`: the step's GDN input projections run the lean conv as
        # their epilogue.
        self.inconv = False
        # `DP_GDN_REPLAY_NORM_BF16_ENV`: the step's replays write the gated norm's output.
        self.rnorm = False


def _gdn_lean_now(core) -> bool:
    """Whether the trunk step in flight runs lean (`DP_GDN_CONV_LEAN_ENV`)."""
    step = getattr(core, "_gdn_step", None)
    return step is not None and step.lean


def _gdn_inconv_now(core) -> bool:
    """Whether the trunk step in flight runs the lean conv in the GDN input projections
    (`DP_GDN_INPROJ_CONV_BF16_ENV`)."""
    step = getattr(core, "_gdn_step", None)
    return step is not None and step.lean and step.inconv


def _gdn_rnorm_now(core) -> bool:
    """Whether the trunk step in flight runs the gated norm in its replays
    (`DP_GDN_REPLAY_NORM_BF16_ENV`)."""
    step = getattr(core, "_gdn_step", None)
    return step is not None and step.lean and step.rnorm


def _gdn_step_clear(core) -> None:
    """A trunk target's first word on the step: the previous step's lean decision is not this
    step's."""
    step = getattr(core, "_gdn_step", None)
    if step is not None:
        step.clear()


def _gdn_lean_seen(core, what: str, count: int, unit: str) -> None:
    """One log line the first time a lean form engages at a shape, eager and under capture
    apart. Host side: it runs in the warmup and the capture of a graph, never on a replay."""
    seen = (what, count, bool(torch.cuda.is_current_stream_capturing()))
    if seen not in _GDN_LEAN_LOGGED:
        _GDN_LEAN_LOGGED.add(seen)
        logger.info(f"DP GDN {what} engaged: rank {core.dp_rank}, {count} {unit}, graph={seen[2]}")


def _gdn_lean_build(core, device) -> bool:
    """`DP_GDN_CONV_LEAN_ENV`, read once at build: the wish `_gdn_lean_bind` settles against
    each bound manager. True at a dep4 / dep8 rank on SM103 with MTP, PDL and a width-4 conv
    and the switch on (default); False elsewhere (set explicitly where it cannot apply: one
    warning)."""
    if not _switch(DP_GDN_CONV_LEAN_ENV, default=True):
        return False
    ok = (
        _dp_decode_rank(core, device)
        and core.mtp_enabled
        and bool(_GDN_REPLAY_PDL)
        and core.conv_kernel == 4
    )
    if not ok and os.environ.get(DP_GDN_CONV_LEAN_ENV, "").strip():
        logger.warning(
            f"{DP_GDN_CONV_LEAN_ENV} is set but ignored: it takes effect only at a dep4 / dep8 "
            "rank on SM103 with MTP and PDL on, over a width-4 conv; every verify step keeps the "
            "stock conv, promotion and gated norm"
        )
    return ok


def _gdn_lean_state(core, cm) -> tuple:
    """`(state, why)`: the bound manager's speculative state when the lean forms can promote
    through it, else `(None, the reason)`."""
    from tensorrt_llm._torch.modules.fla.cache_manager import GDNReplayState

    bound = core._bound
    state = getattr(cm, "_speculative_state", None)
    g0 = core.gdn_layers[0]
    if bound.replay is None or bound.int_conv is None or bound.int_idx is None:
        return None, "the manager bound no verify-step buffers"
    if bound.replay.ring is not None:
        return None, "the manager runs FlashInfer's ring replay"
    if bound.pools.ssm[g0].dtype != torch.bfloat16:
        return None, f"the recurrent pool is {bound.pools.ssm[g0].dtype}, not bf16 (no CuTe replay)"
    if type(state) is not GDNReplayState:
        return None, f"the speculative state is {type(state).__name__}, not GDNReplayState"
    if not state._linear_steps or state._stacked_conv is None:
        return None, "the speculative state does not promote linear draft chains in one launch"
    if getattr(state, _GDN_LEAN_WRAPPED, None) not in (None, core):
        return None, "the speculative state is already wrapped by another model"
    stack, snaps, c = state._stacked_conv, state.intermediate_conv, core.conv_dim
    layers = list(core.gdn_layers)
    if snaps is None or snaps.dim() != 5 or tuple(snaps.shape[3:]) != (c, 3):
        return None, "the snapshot buffer is not [layers, max_batch, T, conv_dim, 3]"
    if tuple(stack.shape[0:1] + stack.shape[2:]) != (len(layers), c, 3):
        return None, f"the stacked conv pool is {tuple(stack.shape)}"
    if stack.dtype != torch.bfloat16 or stack.stride(2) != 3 or stack.stride(3) != 1:
        return None, f"the conv pool is {stack.dtype} with strides {tuple(stack.stride())}"
    for j, i in enumerate(layers):
        view = bound.pools.conv[i]
        if (
            view.data_ptr() != stack[j].data_ptr()
            or view.stride() != stack[j].stride()
            or bound.int_conv[i].data_ptr() != snaps[j].data_ptr()
        ):
            return None, f"GDN layer {i} is not layer {j} of the manager's stacked buffers"
        taps = core._decode._layers[i].conv_w
        if tuple(taps.shape) != (c, 4) or not taps.is_contiguous() or taps.dtype != torch.bfloat16:
            return None, f"GDN layer {i}'s conv taps are {tuple(taps.shape)} {taps.dtype}"
    meta = state.get_replay_metadata()
    if meta is None or meta.replay_step_width != snaps.shape[2]:
        return None, "the replay step width is not the snapshot buffer's"
    return state, ""


def _gdn_lean_selfcheck(core, conv_w, c: int, t: int, history: int, stride: int, device) -> str:
    """Run the lean pair and the stock pair (`causal_conv1d_triton_update` with snapshots, then
    `promote_conv_and_advance_replay_triton`) on private buffers -- this rank's taps, random
    rows, two layers, four requests on separate replay slots, every accepted count, a dummy
    request -- and compare the conv output, the windows and the bookkeeping bitwise. '' when
    they agree, else what differs. Rank-local: no collective."""
    from tensorrt_llm._torch.modules.fla.replay_promotion import (
        promote_conv_and_advance_replay_triton,
    )

    gen = torch.Generator(device=device).manual_seed(20261125)
    n, layers, slots_n = 4, 2, 8
    page = torch.randn(slots_n, layers, c, 3, generator=gen, device=device).to(torch.bfloat16)
    x = torch.randn(n * t, stride, generator=gen, device=device).to(torch.bfloat16)[:, :c]
    slots = torch.tensor([7, 0, 3, 5], dtype=torch.int32, device=device)
    rows = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    replay = torch.tensor([9, 4, 0, 6], dtype=torch.int32, device=device)
    acc = torch.tensor([(g % t) + 1 for g in range(n)], dtype=torch.int32, device=device)
    dummy = torch.tensor([False, False, False, True], device=device)
    meta = torch.randint(0, history + 1, (12, 2), generator=gen, device=device).int()
    meta[:, 1] %= 2
    lean_page, lean_meta = page.clone(), meta.clone()
    xsave = torch.zeros(layers, n, t, c, dtype=torch.bfloat16, device=device)
    snaps = torch.zeros(layers, n, t, c, 3, dtype=torch.bfloat16, device=device)
    lean_out, stock_out = [], []
    for j in range(layers):
        lean_out.append(
            gdn_conv_lean(x, conv_w, lean_page[:, j], slots, rows, xsave[j], t, pad_slot_id=-1)
        )
        o = causal_conv1d_triton_update(
            x=torch.transpose(torch.reshape(x, [n, t, c]), 1, 2), conv_state=page[:, j],
            weight=conv_w, bias=None, activation="silu", conv_state_indices=slots,
            pad_slot_id=-1, num_accepted_tokens=None, intermediate_conv_window=snaps[j],
            intermediate_state_indices=rows,
        )  # fmt: skip
        stock_out.append(torch.reshape(torch.transpose(o, 1, 2), [n * t, c]))
    gdn_promote_x(
        lean_page.permute(1, 0, 2, 3), xsave, rows, slots, acc, dummy, lean_meta[:, 0],
        lean_meta[:, 1], t, history, replay_slots=replay, pad_slot_id=-1,
    )  # fmt: skip
    promote_conv_and_advance_replay_triton(
        page.permute(1, 0, 2, 3), snaps, rows, acc, slots, num_accepted_tokens=acc,
        replay_pnat=meta[:, 0], replay_cache_buf_idx=meta[:, 1], dummy_request_mask=dummy,
        replay_step_width=t, replay_history_size=history, replay_indices=replay,
        position_source_is_token_count=True,
    )  # fmt: skip
    torch.cuda.synchronize(device)
    bad = []
    if not all(torch.equal(a, b) for a, b in zip(lean_out, stock_out)):
        bad.append("the conv output")
    if not torch.equal(lean_page, page):
        bad.append("the promoted windows")
    if not torch.equal(lean_meta, meta):
        bad.append("the replay bookkeeping")
    return " and ".join(bad)


def _gdn_lean_bind(core, cm, max_requests: int) -> None:
    """Settle `DP_GDN_CONV_LEAN_ENV` against one bound cache manager (behind `_bound`): the
    manager's state must be the stock `GDNReplayState` laid out as `_gdn_lean_state` checks; then
    `xsave` (`[GDN layers, max_batch, T, conv_dim]` bf16) is allocated, both kernels and the PDL
    gated norm are compiled, the self-check runs, and the state's promotion is wrapped once (it
    runs the stock method unless the step in flight ran lean). Otherwise one warning and the
    manager's verify steps keep the stock pair. Rank-local."""
    if not getattr(core, "_gdn_lean_want", False):
        return
    core._gdn_lean_on = False
    core._gdn_lean_state_ref = None
    core._gdn_xsave, core._gdn_xsave_all = {}, None
    core._gdn_inconv = {}
    core._gdn_rnorm = frozenset()
    core._gdn_step.clear()
    state, why = _gdn_lean_state(core, cm)
    layers = list(core.gdn_layers)
    if state is not None:
        snaps = state.intermediate_conv
        max_batch, t, c = snaps.shape[1], snaps.shape[2], core.conv_dim
        history = state.get_replay_metadata().replay_history_size
        device = snaps.device
        conv_w = core._decode._layers[layers[0]].conv_w
        try:
            gdn_conv_lean_precompile(c, t, history, sum(core.in_proj_split), device)
            # The PDL gated norm a lean step launches, on a private two-token layer layout.
            hv, n = core.num_v_heads, core.head_v_dim
            proj = torch.zeros(2, sum(core.in_proj_split), dtype=torch.bfloat16, device=device)
            z3 = torch.reshape(torch.split(proj, core.in_proj_split, 1)[1], [2, hv, n])
            rms_norm_gated_token_major(
                torch.zeros(2 * hv, n, dtype=torch.bfloat16, device=device), z3,
                weight=core._decode._layers[layers[0]].gnorm, eps=core.eps, fp8_scale=None,
                gate_activation="sigmoid", pdl=True,
            )  # fmt: skip
            why = _gdn_lean_selfcheck(core, conv_w, c, t, history, sum(core.in_proj_split), device)
            if why:
                why = f"the lean pair differs from the stock pair in {why}"
        except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
            why = f"the lean kernels raised {type(e).__name__}: {e}"
    if why:
        logger.warning(
            f"{DP_GDN_CONV_LEAN_ENV} set but not engaged for {type(cm).__name__} at rank "
            f"{core.dp_rank}: {why}; its verify steps keep the stock conv, promotion and gated "
            "norm"
        )
        return
    xsave = torch.zeros([len(layers), max_batch, t, c], dtype=torch.bfloat16, device=device)
    core._gdn_xsave_all = xsave
    core._gdn_xsave = {i: xsave[j] for j, i in enumerate(layers)}
    if getattr(state, _GDN_LEAN_WRAPPED, None) is None:
        _gdn_lean_wrap(core, state)
    core._gdn_lean_state_ref = state
    core._gdn_lean_on = True
    # `DP_GDN_INPROJ_CONV_BF16_ENV` and `DP_GDN_REPLAY_NORM_BF16_ENV`, on top.
    _gdn_inconv_bind(core, cm, state)
    _gdn_rnorm_bind(core, cm, max_requests)
    logger.info(
        f"DP GDN conv lean bound ({DP_GDN_CONV_LEAN_ENV}): rank {core.dp_rank} of {core.dp_size}, "
        f"{type(cm).__name__}, verify steps of {_dp_lowrows_floor(_GDN_LEAN_ROWS[0])}.."
        f"{_GDN_LEAN_ROWS[1]} rows, xsave {tuple(xsave.shape)} = "
        f"{xsave.numel() * 2 / (1 << 20):.1f} MiB (bind {core._binds})"
    )


def _gdn_lean_wrap(core, state) -> None:
    """Wrap `state._promote_conv_states_and_advance_replay` once: the stock method unless the
    step in flight ran lean on this core's bound state; then `gdn_promote_x` builds the same
    windows from the raw rows and advances the same bookkeeping of the promotion's replay
    slots. The promotion is the last launch of a state update, so the step's record ends here."""
    stock = state._promote_conv_states_and_advance_replay

    def promote(batch, metadata, replay_indices):
        step = core._gdn_step
        lean, step.lean = step.lean, False
        if not lean or core._gdn_lean_state_ref is not state:
            return stock(batch, metadata, replay_indices)
        xsave = core._gdn_xsave_all
        if batch.num_generations != step.gens or metadata.replay_step_width != xsave.shape[2]:
            raise RuntimeError(
                f"{DP_GDN_CONV_LEAN_ENV}: the lean step carried {step.gens} generation requests "
                f"of {xsave.shape[2]} rows and the state update {batch.num_generations} of "
                f"{metadata.replay_step_width}"
            )
        assert batch.is_dummy_request is not None, "the promotion needs the dummy mask"
        gdn_promote_x(
            state._stacked_conv,
            xsave,
            batch.source_state_indices,
            batch.destination_state_indices,
            batch.num_accepted_tokens,
            batch.is_dummy_request,
            metadata.prev_num_accepted_tokens,
            metadata.cache_buf_idx,
            metadata.replay_step_width,
            metadata.replay_history_size,
            replay_slots=replay_indices,
            pad_slot_id=_PAD_SLOT_ID,
        )

    state._promote_conv_states_and_advance_replay = promote
    setattr(state, _GDN_LEAN_WRAPPED, core)


def _gdn_lean_step(core, num_gens: int, rows: int, p: int) -> None:
    """`DP_GDN_CONV_LEAN_ENV`: decide, once for a generation-only verify step of `num_gens`
    requests x `p` rows, whether all its GDN layers run lean. Host ints of the step, fixed by a
    decode graph's key."""
    xsave = core._gdn_xsave_all
    lo, hi = _dp_lowrows_floor(_GDN_LEAN_ROWS[0]), _GDN_LEAN_ROWS[1]
    if lo <= rows <= hi and p == xsave.shape[2] and num_gens <= xsave.shape[1]:
        step = core._gdn_step
        step.lean, step.gens = True, num_gens
        _gdn_lean_seen(
            core, "conv lean", rows, "rows (gdn_conv_lean, gdn_promote_x, PDL gated norm)"
        )
        if num_gens in getattr(core, "_gdn_rnorm", ()):
            step.rnorm = True
            _gdn_lean_seen(
                core, "replay norm bf16", num_gens, "requests (the gated norm in the replay)"
            )
        if rows in getattr(core, "_gdn_inconv", {}):
            step.inconv = True
            _gdn_lean_seen(
                core,
                "inproj conv bf16",
                rows,
                "rows (the lean conv in the input projection's GEMM)",
            )


#: Switch, on by default, riding on `DP_GDN_CONV_LEAN_ENV`: the CuTe replay of a lean verify step
#: of `_GDN_REPLAY_NORM_REQUESTS` requests a rank writes the gated RMS norm of its output itself
#: (the verify op's norm mode, `cached_replay_norm_update`: the instructions of the registered
#: norm's 4-row kernel on the bf16-rounded replay output, stored right after their bf16
#: rounding), so the gated norm launch leaves the chain (36 a step) and the output projection
#: becomes the replay's PDL dependent. The tensor is the norm launch's bit for bit, and state,
#: history and commits are the replay's. It engages at request counts whose launch mapping gives
#: a CTA whole v heads (tpc 8 or 24: every count of `DP_GDN_REPLAY_MAP_ENV`'s table, and 16) and
#: an even number of rows a request. Values: unset or `1` -- `_GDN_REPLAY_NORM_REQUESTS`;
#: `<lo>:<hi>` -- that request range, taken as written (inside it; with `DP_LOWROWS_ENV` = r its
#: floor reaches r, so `2:64` starts at 2 requests under `DP_LOWROWS_ENV=2`); `0` -- the norm
#: launch.
#: `DP_LOWROWS_ENV` lowers the floor of the default range (counts whose mapping splits a v head
#: still keep the launch). With the in-proj
#: release and its conv epilogue, part of the -0.47..-0.49 ms dep4 trunk-loop micro; dep8 left
#: 8-CTX share 68.3 -> 70.6% with the bf16 output projections. Read once, at each bind.
DP_GDN_REPLAY_NORM_BF16_ENV = "TRTLLM_QWEN4_EXP_DP_GDN_REPLAY_NORM_BF16"
#: Requests a rank the switch above serves (`5:64`, the range measured at dep4 / dep8).
_GDN_REPLAY_NORM_REQUESTS = (5, 64)  # capacity: fixed: dispatch thresholds


def _gdn_replay_norm_range() -> tuple | None:
    """`DP_GDN_REPLAY_NORM_BF16_ENV`'s request range, or None (off); any other spelling raises.
    A named range may start at `DP_LOWROWS_ENV`'s request count (the measured run's `2:64`)."""
    raw = os.environ.get(DP_GDN_REPLAY_NORM_BF16_ENV, "")
    value = raw.strip().lower()
    lo, hi = _GDN_REPLAY_NORM_REQUESTS
    if not value or value in _SWITCH_ON:
        return lo, hi
    if value in _SWITCH_OFF:
        return None
    lo = min(lo, _dp_lowrows() or lo)
    a, sep, b = value.partition(":")
    if sep and a.isdigit() and b.isdigit() and lo <= int(a) <= int(b) <= hi:
        return int(a), int(b)
    raise ValueError(
        f"{DP_GDN_REPLAY_NORM_BF16_ENV}={raw!r}: expected 1 / 0 or <lo>:<hi> inside {lo}:{hi}"
    )


def _gdn_rnorm_bind(core, cm, max_requests: int) -> None:
    """`DP_GDN_REPLAY_NORM_BF16_ENV` against a manager the lean conv engaged for (called at the
    end of `_gdn_lean_bind`): `core._gdn_rnorm`, the request counts whose lean steps take the
    norm mode, after compiling their kernels. Empty where it cannot take effect (one warning when
    it was set, or when a set range keeps the launch at some counts)."""
    core._gdn_rnorm = frozenset()
    rng = _gdn_replay_norm_range()
    if rng is None:
        return
    explicit = bool(os.environ.get(DP_GDN_REPLAY_NORM_BF16_ENV, "").strip())
    # `DP_LOWROWS_ENV` lowers the default range's floor; a named range is served as written.
    named = ":" in os.environ.get(DP_GDN_REPLAY_NORM_BF16_ENV, "")
    xsave = core._gdn_xsave_all
    p = xsave.shape[2]
    bad = ""
    if p % 2:
        bad = f"{p} rows a request: the norm mode splits an even count over two lane halves"
    elif core.value_dim % 8 or sum(core.in_proj_split) % 8 or core.head_v_dim != 128:
        bad = "the input projection's z columns are not 16-byte aligned 128-wide heads"
    elif any(core._decode._layers[i].gnorm.data_ptr() % 16 for i in core.gdn_layers):
        bad = "a gated-norm weight is not 16-byte aligned"
    lo = rng[0] if named else min(rng[0], _dp_lowrows() or rng[0])
    hi = min(rng[1], max_requests, xsave.shape[1])
    lean_lo = _dp_lowrows_floor(_GDN_LEAN_ROWS[0])
    took, left = [], []
    if not bad:
        for n in range(lo, hi + 1):
            if not lean_lo <= n * p <= _GDN_LEAN_ROWS[1]:
                continue
            commit = n < CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            config = _gdn_replay_map(n) if core.gdn_replay_map else None
            (took if gdn_replay_norm_ok(n, p, commit, config) else left).append(n)
        if not took:
            bad = f"no request count in {lo}..{hi} launches the replay with whole v heads a CTA"
    if not bad:
        mm = core._bound.cm
        try:
            gdn_replay_precompile(
                p,
                core.num_k_heads,
                core.num_v_heads,
                0,
                pdl=_GDN_REPLAY_PDL,
                partition_from=CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
                work_item_width=core._gdn_wi_width,
                separate_replay_slots=_has_replay_slots(mm),
                configs={n: _gdn_replay_map(n) for n in took} if core.gdn_replay_map else None,
                norm_counts=tuple(took),
            )
        except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
            bad = f"the norm kernels raised {type(e).__name__}: {e}"
    if bad:
        if explicit or bad.startswith("the norm kernels"):
            logger.warning(
                f"{DP_GDN_REPLAY_NORM_BF16_ENV} is set but not engaged for {type(cm).__name__} at "
                f"rank {core.dp_rank}: {bad}; every lean step keeps the gated norm launch"
            )
        return
    if left and explicit:
        logger.warning(
            f"{DP_GDN_REPLAY_NORM_BF16_ENV}: at {left} requests the launch mapping splits a v head "
            "over CTAs; those lean steps keep the gated norm launch"
        )
    core._gdn_rnorm = frozenset(took)
    logger.info(
        f"DP GDN replay norm bf16 ({DP_GDN_REPLAY_NORM_BF16_ENV}): rank {core.dp_rank} of "
        f"{core.dp_size}, the gated norm in the replay at {min(took)}..{max(took)} requests"
        + (f" ({len(left)} counts keep the launch: {left})" if left else "")
        + f" (bind {core._binds})"
    )


#: Switch, on by default, riding on `DP_GDN_CONV_LEAN_ENV` and `DP_INPROJ_RELEASE_ENV`: a lean
#: verify step of `_GDN_INPROJ_CONV_REQUESTS` requests a rank runs each GDN layer's lean conv as
#: the EPILOGUE of its bf16 input projection (`mid_rows_mm`'s conv mode, `mid_mm_conv_bf16`: the
#: release's GEMM at the same plan and MMA N; the weight tiles of q|k|v apply the 4-tap conv +
#: SiLU to the bf16 tile they stage and store the conv output and the raw rows into `xsave`),
#: so the `gdn_conv_lean` launch leaves the chain (36 a step) and the replay becomes the GEMM's
#: PDL dependent. The conv keeps `gdn_conv_lean`'s instructions on the same bf16 columns: the
#: conv output, `xsave` and z | a | b are bitwise the two launches'. From 17 requests an
#: epilogue thread of a conv tile holds two items and the epilogue costs more than the launch it
#: replaces (GB300 micro), hence the range; `DP_LOWROWS_ENV` lowers its floor. The bind runs the
#: conv GEMM against the two launches on private buffers with this rank's weights (rank-local; a
#: difference or an error leaves the lean steps on `gdn_conv_lean`). Part of the -0.47..-0.49 ms
#: dep4 trunk-loop micro of the input-projection release. `0` keeps `gdn_conv_lean`. Read once,
#: when the target is built.
DP_GDN_INPROJ_CONV_BF16_ENV = "TRTLLM_QWEN4_EXP_DP_GDN_INPROJ_CONV_BF16"
#: Requests a rank the switch above serves.
_GDN_INPROJ_CONV_REQUESTS = (5, 16)  # capacity: fixed: dispatch thresholds


def _gdn_inconv_bind(core, cm, state) -> None:
    """`DP_GDN_INPROJ_CONV_BF16_ENV` against a manager the lean conv engaged for (called at the
    end of `_gdn_lean_bind`): `core._gdn_inconv = {rows: _InProj of the first GDN layer}` for
    the CUDA-graph row counts of the request range whose release entry has a K-unsplit plan,
    after compiling each conv GEMM and checking it bitwise against `mid_mm_raw` + `gdn_conv_lean`
    on private buffers. Empty where it cannot take effect (one warning when it was set)."""
    core._gdn_inconv = {}
    if not _switch(DP_GDN_INPROJ_CONV_BF16_ENV, default=True):
        return
    explicit = bool(os.environ.get(DP_GDN_INPROJ_CONV_BF16_ENV, "").strip())
    release = core._inproj_release or {}
    layers = list(core.gdn_layers)
    xsave = core._gdn_xsave_all
    p = xsave.shape[2]
    bad = ""
    if any(i not in release for i in layers):
        bad = f"{DP_INPROJ_RELEASE_ENV} does not serve every GDN layer"
    elif p != _MID_CONV_ROWS:
        bad = f"{p} rows a request; the conv epilogue is written for {_MID_CONV_ROWS}"
    elif core._bound.int_idx is None or core._bound.int_idx.dtype != torch.int32:
        bad = "the manager's snapshot-row indices are not int32"
    elif core.conv_dim % 128 or core.conv_dim > sum(core.in_proj_split):
        bad = f"a conv of {core.conv_dim} columns is not whole weight tiles"
    else:
        for i in layers:
            why = conv_ready(
                core._decode._layers[i].conv_w, core._bound.pools.conv[i], core._gdn_xsave[i]
            )
            if why:
                bad = f"GDN layer {i}: {why}"
                break
    lo = min(_GDN_INPROJ_CONV_REQUESTS[0], _dp_lowrows() or _GDN_INPROJ_CONV_REQUESTS[0])
    hi = min(_GDN_INPROJ_CONV_REQUESTS[1], xsave.shape[1])
    ents = {}
    if not bad:
        first = release[layers[0]]
        for r in range(lo, hi + 1):
            ent = first.get(r * p)
            if ent is not None and ent.plan.ks == 1:
                ents[r * p] = ent
        bad = "" if ents else f"no CUDA-graph row count of {lo}..{hi} requests has a release entry"
    if not bad:
        bad = _gdn_inconv_selfcheck(core, ents, p)
    if bad:
        if explicit or bad.startswith("the conv GEMM"):
            logger.warning(
                f"{DP_GDN_INPROJ_CONV_BF16_ENV} is set but not engaged for {type(cm).__name__} "
                f"at rank {core.dp_rank}: {bad}; every lean step keeps gdn_conv_lean"
            )
        return
    core._gdn_inconv = ents
    logger.info(
        f"DP GDN inproj conv bf16 ({DP_GDN_INPROJ_CONV_BF16_ENV}): rank {core.dp_rank} of "
        f"{core.dp_size}, the lean conv in the input projection's GEMM at rows {sorted(ents)} "
        f"(bind {core._binds})"
    )


def _gdn_inconv_selfcheck(core, ents: dict, p: int) -> str:
    """Every row count of `ents` (one compile a distinct (MMA N, plan); the request count only
    moves the epilogue's item predicates, which a check at one row count of a plan would leave
    unchecked at the others), run on private buffers with the first GDN layer's weight and taps
    against `mid_mm_raw` + `gdn_conv_lean`: '' when the conv output, `xsave` and z | a | b are
    equal, else what differs. Rank-local."""
    i = core.gdn_layers[0]
    L = core._decode._layers[i]
    c = core.conv_dim
    try:
        device = L.conv_w.device
        gen = torch.Generator(device=device).manual_seed(20261125)
        page = torch.randn(8, 1, c, 3, generator=gen, device=device).to(torch.bfloat16)
        for rows, ent in sorted(ents.items()):
            n = rows // p
            x = torch.randn(rows, ent.w.shape[1], generator=gen, device=device)
            x = x.to(torch.bfloat16)
            slots = torch.randperm(8, generator=gen, device=device)[: min(n, 8)].int()
            slots = slots.repeat(-(-n // slots.shape[0]))[:n].contiguous()
            xrows = torch.arange(n, dtype=torch.int32, device=device)
            xs_a = torch.zeros(n, p, c, dtype=torch.bfloat16, device=device)
            xs_b = torch.zeros_like(xs_a)
            conv_a, out_a = mid_mm_conv_bf16(
                x, ent.w, ent.plan, L.conv_w, page[:, 0], slots, xrows, xs_a, p,
                mpad=ent.mpad, pad_slot_id=_PAD_SLOT_ID,
            )  # fmt: skip
            out_b = _inproj_mm(x, ent)
            conv_b = gdn_conv_lean(out_b[:, :c], L.conv_w, page[:, 0], slots, xrows, xs_b, p)
            if not (
                torch.equal(conv_a, conv_b)
                and torch.equal(xs_a, xs_b)
                and torch.equal(out_a[:, c:], out_b[:, c:])
            ):
                return f"the conv GEMM differs from the two launches at {rows} rows"
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
        return f"the conv GEMM raised {type(e).__name__}: {e}"
    return ""


#: Off by default (class S: another K split than the bf16 entry). At an attention-DP rank on
#: SM103, the output projections of a decode step -- the GDN out projection and the attention
#: o projection, `[rows, 6144] x [6144, 2560]` -- run `mid_rows_mm`'s bf16 tcgen05 GEMM over the
#: checkpoint's weight (no copy) with PDL, at the MMA N and plan `_OUT_PROJ_TABLE` names for the
#: row count, instead of the step's bf16 entry (`low_m_bf16_mm`: FlashInfer's split-K kernel up
#: to 32 rows, cuBLASLt without PDL above). The GEMM streams its weight before its wait, so it
#: overlaps the gated norm (which releases its dependents early) or the attention's last
#: launch. Same bf16 operands and fp32 accumulation; the K split differs, so the output moves
#: at bf16 rounding level (GB300 dep4: none of 18 greedy texts stayed identical). Values: `1` --
#: the GDN out projection at every row count of the table, the attention o projection at
#: `_OUT_PROJ_ATTN_ROWS` (where its producer does not release early and the micro measured a
#: gain); `gdn` -- the GDN site only; `all` -- both sites at every row count of the table. Only
#: the row counts of the CUDA-graph batch sizes engage (their kernels are compiled at build).
#: GB300 dep4 at 12 requests a rank: -0.14 ms a step; the GDN site saves 36..111 us a
#: step of 36 launches at 20..128 rows. Read once, when the target is built.
DP_OUT_PROJ_BF16_ENV = "TRTLLM_QWEN4_EXP_DP_OUT_PROJ_BF16"
#: `(first rows, last rows, MMA N, plan)`: the GEMM of a row count, measured on GB300 (the
#: model's GDN chain of 36 layers captured with each candidate; ks 3..12 x 9 (kpb, stages, l2)
#: at 28 / 48 / 96 / 120 / 224 rows and the best plans at 15 more counts). The MMA N rounded to
#: 16 rows wins over the op's 32 / 64 / 128 / 256 at every count. Past 144 rows cuBLASLt is
#: faster; those rows keep the bf16 entry.
_OUT_PROJ_TABLE = (
    (17, 28, 32, MidPlan(12, 2, 8, 2)),
    (29, 32, 32, MidPlan(6, 4, 10, 1)),
    (33, 36, 48, MidPlan(6, 4, 10, 1)),
    (37, 40, 48, MidPlan(6, 4, 8, 2)),
    (41, 48, 48, MidPlan(6, 4, 10, 1)),
    (49, 64, 64, MidPlan(6, 4, 8, 2)),
    (65, 80, 80, MidPlan(6, 4, 4, 2)),
    (81, 96, 96, MidPlan(6, 4, 8, 2)),
    (97, 112, 112, MidPlan(6, 4, 8, 2)),
    (113, 128, 128, MidPlan(6, 4, 6, 2)),
    (129, 144, 144, MidPlan(6, 4, 6, 2)),
)
#: The weight geometry `(N, K)` the table was measured for: the trunk's output projections.
_OUT_PROJ_GEOM = (2560, 6144)
#: Rows at which the attention o projection takes the GEMM under the value `1` (GB300 micro,
#: 12 launches a step: 0.004..0.017 ms saved at 32..96 rows, -0.003..-0.005 at 20..28 against
#: FlashInfer's split-K, -0.003..-0.008 from 120 rows).
_OUT_PROJ_ATTN_ROWS = (29, 96)  # capacity: fixed: dispatch thresholds


#: Off by default (class S, as `DP_OUT_PROJ_BF16_ENV`). `<lo>` (1..16) with the output
#: projections engaged: `lo`..16 rows a rank also take the table's first entry (MMA N 32, plan
#: (12, 2, 8, 2)), at the GDN site and, under `all`, the attention site. Measured on dep8 with
#: `lo` 5 at 3 requests a rank (8 / 12 rows), where it was part of a -0.33 ms step together with
#: forms not ported here; it was never run at the other caps' lower batches, hence its own
#: switch. Read once, when the target is built.
DP_OUT_PROJ_BF16_ROWSLO_ENV = "TRTLLM_QWEN4_EXP_DP_OUT_PROJ_BF16_ROWSLO"


def _out_proj_rowslo() -> int | None:
    """`DP_OUT_PROJ_BF16_ROWSLO_ENV`: the lowest row count, or None (unset, empty, off)."""
    raw = os.environ.get(DP_OUT_PROJ_BF16_ROWSLO_ENV, "")
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return None
    if value.isdigit() and 1 <= int(value) < _OUT_PROJ_TABLE[0][0]:
        return int(value)
    raise ValueError(
        f"{DP_OUT_PROJ_BF16_ROWSLO_ENV}={raw!r}: expected 0 or 1..{_OUT_PROJ_TABLE[0][0] - 1}"
    )


def _out_proj_mode() -> str:
    """`DP_OUT_PROJ_BF16_ENV`: "" (unset, empty, an off spelling), "1", "gdn" or "all"."""
    raw = os.environ.get(DP_OUT_PROJ_BF16_ENV, "")
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return ""
    if value in _SWITCH_ON:
        return "1"
    if value in ("gdn", "all"):
        return value
    raise ValueError(f"{DP_OUT_PROJ_BF16_ENV}={raw!r}: expected 0, 1, gdn or all")


def _out_proj_shape(rows: int, rowslo: int | None = None) -> tuple | None:
    """`(MMA N, plan)` of `_OUT_PROJ_TABLE` at `rows`, None outside it; with `rowslo`
    (`DP_OUT_PROJ_BF16_ROWSLO_ENV`) the first entry also from `rowslo` rows."""
    first = _OUT_PROJ_TABLE[0]
    if rowslo is not None and rowslo <= rows < first[0]:
        return first[2], first[3]
    for lo, hi, mpad, plan in _OUT_PROJ_TABLE:
        if lo <= rows <= hi:
            return mpad, plan
    return None


def _out_proj_build(core, device) -> dict | None:
    """`DP_OUT_PROJ_BF16_ENV`, read once at build: `{"gdn_out" | "attn_o": {rows: (MMA N,
    plan)}}` at an attention-DP rank on SM103 whose output projections are contiguous bf16
    `_OUT_PROJ_GEOM` weights with no other decode copy, every entry's GEMM compiled here; None
    where it is off. Set where it cannot apply: one warning, None."""
    mode = _out_proj_mode()
    if not mode:
        return None
    names = [f"l{i}_out_w" for i in core.gdn_layers] + [f"l{i}_o_w" for i in core.attn_layers]
    bad = [
        n
        for n in names
        if n not in core.w
        or n in core._mid
        or core.w[n].dtype != torch.bfloat16
        or tuple(core.w[n].shape) != _OUT_PROJ_GEOM
        or not core.w[n].is_contiguous()
        or core.w[n].data_ptr() % 16
    ]
    if not _dp_decode_rank(core, device) or bad:
        logger.warning(
            f"{DP_OUT_PROJ_BF16_ENV} is set but ignored: it serves a dep4 / dep8 rank on SM103 "
            f"whose output projections are bf16 {_OUT_PROJ_GEOM[0]} x {_OUT_PROJ_GEOM[1]} "
            f"weights without another decode copy ({len(bad)} are not)"
        )
        return None
    rowslo = _out_proj_rowslo()
    gdn = {}
    first = _OUT_PROJ_TABLE[0][0] if rowslo is None else rowslo
    for rows in _graph_rows(core, first, _OUT_PROJ_TABLE[-1][1]):
        shape = _out_proj_shape(rows, rowslo)
        if shape is not None:
            gdn[rows] = shape
    lo, hi = _OUT_PROJ_ATTN_ROWS
    attn = (
        {}
        if mode == "gdn"
        else dict(gdn)
        if mode == "all"
        else {r: e for r, e in gdn.items() if lo <= r <= hi}
    )
    w = core.w[names[0]]
    for mpad, plan in sorted(set(gdn.values()), key=str):
        # The first call compiles the kernel (`mid_mm_raw` caches it per shape and plan).
        mid_mm_raw(torch.zeros(1, w.shape[1], dtype=w.dtype, device=device), w, plan, mpad=mpad)
    logger.info(
        f"output projections on the bf16 mid-rows GEMM ({DP_OUT_PROJ_BF16_ENV}={mode}): rank "
        f"{core.dp_rank} of {core.dp_size}, GDN rows {sorted(gdn)}, attention rows {sorted(attn)}"
    )
    return {"gdn_out": gdn, "attn_o": attn}


#: Off by default (class S, as `DP_OUT_PROJ_BF16_ENV`, which it needs engaged). The draft
#: layer's attention output projection (`_mtp_tail`, one row a request) runs the output
#: projections' bf16 mid-rows GEMM at the table's first entry (MMA N 32, plan (12, 2, 8, 2))
#: at the CUDA-graph batch sizes up to 32 rows, instead of cuBLAS. dep8 at 3 requests a rank:
#: -0.019 ms a step. Read once, when the target is built.
DP_DRAFT_O_MID_ENV = "TRTLLM_QWEN4_EXP_DP_DRAFT_O_MID"


def _draft_o_build(core, device) -> dict | None:
    """`DP_DRAFT_O_MID_ENV`, read once at build after `_out_proj_build`: `{rows: (MMA N, plan)}`
    of the draft output projection, its GEMM compiled here; None where it is off. Set where it
    cannot apply: one warning, None."""
    if not _switch(DP_DRAFT_O_MID_ENV):
        return None
    name = f"l{getattr(core, 'mtp_layer_idx', -1)}_mtp_o_w"
    w = core.w[name] if getattr(core, "mtp_enabled", False) and name in core.w else None
    why = None
    if core._out_proj is None:
        why = f"it rides on {DP_OUT_PROJ_BF16_ENV}, which is not engaged"
    elif w is None:
        why = "there is no draft layer"
    elif (
        w.dtype != torch.bfloat16
        or tuple(w.shape) != _OUT_PROJ_GEOM
        or not w.is_contiguous()
        or w.data_ptr() % 16
    ):
        why = f"{name} is {tuple(w.shape)} {w.dtype}, not a contiguous bf16 {_OUT_PROJ_GEOM}"
    _, _, mpad, plan = _OUT_PROJ_TABLE[0]
    sizes = sorted(
        {int(b) for b in (getattr(core.model_config, "cuda_graph_batch_sizes", None) or ())}
    )
    rows = {b: (mpad, plan) for b in sizes if 1 <= b <= mpad}
    if why is None and not rows:
        why = f"no CUDA-graph batch size is in 1..{mpad}"
    if why is not None:
        logger.warning(f"{DP_DRAFT_O_MID_ENV} is set but ignored: {why}")
        return None
    mid_mm_raw(torch.zeros(1, w.shape[1], dtype=w.dtype, device=device), w, plan, mpad=mpad)
    logger.info(
        f"draft output projection on the bf16 mid-rows GEMM ({DP_DRAFT_O_MID_ENV}=1): rank "
        f"{core.dp_rank} of {core.dp_size}, rows {sorted(rows)}"
    )
    return rows


def _draft_o(core, attn, weight_t):
    """The draft layer's attention output projection: `DP_DRAFT_O_MID_ENV`'s GEMM at a row count
    it serves (`attn` contiguous and 16-byte aligned), else cuBLAS."""
    table = getattr(core, "_draft_o", None)
    ent = None if table is None else table.get(attn.shape[0])
    if ent is None or not attn.is_contiguous() or attn.data_ptr() % 16:
        return cublas_mm(attn, weight_t)
    return mid_mm_raw(attn, weight_t.t(), ent[1], pdl=True, mpad=ent[0])


def _warn_dp_row_floors_ignored(core) -> None:
    """After the attention-DP decode forms are built: one warning for each row-floor switch that is
    set but rides on a form that is not engaged (`DP_OUT_PROJ_BF16_ROWSLO_ENV` on
    `DP_OUT_PROJ_BF16_ENV`; `DP_LOWROWS_ENV` on `DP_HC_TILES_ENV` or `DP_INPROJ_RELEASE_ENV`), so
    the engine log shows it as the other declines do. Validates both values."""
    if _out_proj_rowslo() is not None and core._out_proj is None:
        logger.warning(
            f"{DP_OUT_PROJ_BF16_ROWSLO_ENV} is set but ignored: it rides on "
            f"{DP_OUT_PROJ_BF16_ENV}, which is not engaged"
        )
    if _dp_lowrows() is not None and not (
        core.hc_tiles or core._inproj_release or getattr(core, "_gdn_lean_want", False)
    ):
        logger.warning(
            f"{DP_LOWROWS_ENV} is set but ignored: it lowers {DP_HC_TILES_ENV}, "
            f"{DP_INPROJ_RELEASE_ENV} and {DP_GDN_CONV_LEAN_ENV}, none of which is engaged"
        )


def _out_proj(core, x, weight_t, site: str, dense):
    """An output projection of a decode step: `DP_OUT_PROJ_BF16_ENV`'s GEMM at a row count its
    table serves for `site` (`weight_t` the `.t()` view of the checkpoint's `[N, K]` weight,
    `x` contiguous and 16-byte aligned), else `dense()` (the step's `_dense_mm`)."""
    table = core._out_proj
    ent = None if table is None else table[site].get(x.shape[0])
    if ent is None or not x.is_contiguous() or x.data_ptr() % 16:
        return dense()
    arm = getattr(core, "_siteb_pending", None)
    if arm is not None and arm.site == site and x.shape[0] == arm.hc.shape[0]:
        # `DP_OUT_PROJ_INJECT_ENV`: the trunk loop armed this projection with its site B.
        core._siteb_pending = None
        mpad, plan = core._siteb.gemm[x.shape[0]]
        return _SiteBFolded(
            *mid_mm_inject(x, weight_t.t(), plan, arm.hc, arm.inject, arm.norm, mpad=mpad)
        )
    return mid_mm_raw(x, weight_t.t(), ent[1], pdl=True, mpad=ent[0])


#: Off by default (class S). With `DP_OUT_PROJ_BF16_ENV` engaged, the MoE-side hyper-connection
#: site (site B) of a trunk layer launches no `qwen4_exp_hc_inject_norm`: the block's output
#: projection runs `mid_rows_mm`'s bf16 GEMM in inject mode (`mid_mm_inject`: same MMA N, plan and
#: operands as `_out_proj`), whose split-K store writes the injected residual (bitwise the inject
#: kernel's), the stream before its rstd `bf16(hc * (1 + norm))` and the partial sums of squares;
#: the site's first projection and up / gate / mean run their deferred-rstd forms
#: (`qwen4_exp_hc_decode_tc_first_ss`, `qwen4_exp_hc_up_gate_mean(rstd=)`). Three launches a site
#: B where there were four. The activation is rounded to bf16 once on both paths, before the rstd
#: factor here and after it there, and the rstd is summed in another order: `mixed` and the
#: injection move at rounding level. Values: `1` -- `_SITEB_ROWS`; `all` -- every row count of
#: the output projections' table; `<lo>:<hi>` -- that range of requests a rank (4 rows a
#: request). Only the CUDA-graph row counts of `_out_proj`'s table engage, at each site where
#: `_out_proj` takes the GEMM (the GDN out projection; the attention o projection where its rows
#: are served). The layer in front of the PLE layer keeps the inject kernel. GB300 dep4 server
#: A/B, both arm orders: -0.059 / -0.060 / -0.052 ms a step at 10 / 12 / 16 requests a rank.
#: Read once, when the target is built.
DP_OUT_PROJ_INJECT_ENV = "TRTLLM_QWEN4_EXP_DP_OUT_PROJ_INJECT"
#: Rows the value `1` serves: 5..20 requests a rank. GB300 dep4 trunk-loop micro, ms a step saved
#: at `all`: 0.045..0.073 at 5..16 requests a rank, 0.019..0.027 at 17..20, within +-0.009 from 22
#: (a warp of the GEMM then owns two or three units, and the fold's store loop eats the launch it
#: saves).
_SITEB_ROWS = (17, 80)  # capacity: fixed: dispatch thresholds
#: Off by default (class S, as `DP_OUT_PROJ_INJECT_ENV`, which it needs engaged). The injecting
#: GEMM at the row counts whose `_OUT_PROJ_TABLE` entry is `_SITEB_PLAN32[0]` (MMA N 32 on the
#: 12-way split: up to 28 rows) runs `_SITEB_PLAN32[1]`, the plan of 29..32 rows, instead: the
#: 12-CTA cluster's split-K store of the fold is slower than the 6-CTA one. Every row of an MMA
#: N 32 tile is computed on its own, so each row is bitwise the row 32 rows would compute; against
#: the table's plan it is another K split. GB300 dep8, with `DP_OUT_PROJ_BF16=all`: -0.133..-0.169
#: ms a step at 5..7 requests a rank. Read once, when the target is built.
DP_SITEB_PLAN32_ENV = "TRTLLM_QWEN4_EXP_DP_SITEB_PLAN32"
_SITEB_PLAN32 = ((32, MidPlan(12, 2, 8, 2)), (32, MidPlan(6, 4, 10, 1)))
#: The norm weight the fold reads is the site's `[C * H]` bf16 vector; its stream width is the
#: output projections' N.
_SITEB_STREAMS = 4  # capacity: fixed: the fold's compiled stream count
#: The most rows the fold serves: the output projections' table's last.
_SITEB_MAX_ROWS = 144  # capacity: fixed: dispatch threshold


class _SiteB(NamedTuple):
    """`DP_OUT_PROJ_INJECT_ENV` engaged at this rank (`core._siteb`): `gemm` = `{rows: (MMA N,
    plan)}` of the injecting GEMM, `attn_rows` the row counts the attention site also takes,
    `first` = `{rows: plan arguments of the first projection}`, `up` = `{rows: the up kernel's
    M tile}`, `skip` the layer that keeps the inject kernel (in front of the PLE layer),
    `plan32` the row counts `DP_SITEB_PLAN32_ENV` moved."""

    gemm: dict
    attn_rows: frozenset
    first: dict
    up: dict
    skip: int
    plan32: tuple


class _SiteBArm(NamedTuple):
    """The operands `_siteb_arm` hands the layer's output projection: the site it applies to
    (`gdn_out` / `attn_o`), the residual, the injection and the site's norm weight."""

    site: str
    hc: torch.Tensor
    inject: torch.Tensor
    norm: torch.Tensor


class _SiteBFolded(NamedTuple):
    """What a folded output projection returns in place of its output: the injected residual,
    `y = bf16(hc * (1 + norm))` and the partial sums of squares of `hc` (fp32 `[rows, C, N /
    32]`)."""

    hc: torch.Tensor
    y: torch.Tensor
    ss: torch.Tensor


def _siteb_wanted(rows_a_request: int) -> tuple | None:
    """`DP_OUT_PROJ_INJECT_ENV` as a `(first rows, last rows)` range, None unset / off."""
    raw = os.environ.get(DP_OUT_PROJ_INJECT_ENV, "")
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return None
    if value in _SWITCH_ON:
        return _SITEB_ROWS
    if value == "all":
        return (1, _SITEB_MAX_ROWS)
    lo, sep, hi = value.partition(":")
    if sep and lo.isdigit() and hi.isdigit() and 1 <= int(lo) <= int(hi):
        return (int(lo) * rows_a_request, int(hi) * rows_a_request)
    raise ValueError(
        f"{DP_OUT_PROJ_INJECT_ENV}={raw!r}: expected 0, 1, all or <lo>:<hi> requests a rank"
    )


def _hc_first_form(core, rows: int, with_inject: bool) -> dict:
    """The plan arguments `_hc_first` gives `qwen4_exp_hc_decode_tc_first` at `rows` rows (up to
    `_HC_TC_FIRST_MAX_ROWS`): `{}` is the default plan."""
    if core.hc_tiles:
        rowsn_lo = getattr(core, "hc_rowsn_lo", _HC_ROWSN_ROWS[0])
        rowsn = _HC_ROWSN_ROWS[0] <= rows <= _HC_ROWSN_ROWS[1] or (
            rowsn_lo <= rows <= _DP_LOWROWS_MAX_ROWS
        )
        if with_inject and rowsn:
            if getattr(core, "hc_first_x2", False) and rows >= _HC_ROWSN_ROWS[0]:
                return {"tm64x2": True}
            return {"rows_on_n": True}
        if _HC_TILE64_ROWS[0] <= rows <= _HC_TILE64_ROWS[1]:
            return {"tm": 64}
    return {}


def _hc_up_tile(core, rows: int, release: bool) -> int:
    """The M tile `_hc_mixer` gives the up kernel at `rows` rows."""
    if core.hc_tiles and (
        _HC_TILE64_ROWS[0] <= rows <= _HC_TILE64_ROWS[1]
        or (not release and _HC_UP2_ROWS[0] <= rows <= _HC_UP2_ROWS[1])
    ):
        return 64
    return 128


def _siteb_selfcheck(core, st: _SiteB, L, rows: int, device) -> str | None:
    """One fold at `rows` rows on this rank's weights of a GDN layer against the stock site
    (`mid_mm_raw` at the fold's plan, `qwen4_exp_hc_inject_norm`, `_hc_mixer`): the residual
    must be bitwise, `mixed` / `low` / `inject` within 1/64 of their scale (rounding level). None
    when it holds, else what differed."""
    gen = torch.Generator(device=device).manual_seed(rows)
    w = L.out.t()
    x = torch.randn(rows, w.shape[1], generator=gen, device=device).to(torch.bfloat16)
    hc = torch.randn(rows, core.stream, generator=gen, device=device).to(torch.bfloat16)
    inject = (torch.rand(rows, core.hc_count, generator=gen, device=device) * 2).to(torch.bfloat16)
    mpad, plan = st.gemm[rows]
    b = mid_mm_raw(x, w, plan, pdl=True, mpad=mpad)
    hc_ref, normed = qwen4_exp_hc_inject_norm(hc, b, inject, L.mhc_norm, core.eps, core.hc_count)
    mixed_ref, inj_ref = _hc_mixer(core, normed, L.mhc_first, L.mhc_up, True)
    folded = _SiteBFolded(*mid_mm_inject(x, w, plan, hc, inject, L.mhc_norm, mpad=mpad))
    hc2, mixed, inj = _siteb_site(core, folded, L, st)[:3]
    if not torch.equal(hc2, hc_ref):
        return f"the injected residual differs from qwen4_exp_hc_inject_norm's at {rows} rows"
    for name, got, ref in (("mixed", mixed, mixed_ref), ("inject", inj, inj_ref)):
        scale = ref.float().abs().max().clamp_min(1e-6)
        if not bool(((got.float() - ref.float()).abs().max() <= scale / 64).item()):
            return f"{name} is past rounding level of the stock site's at {rows} rows"
    return None


def _siteb_build(core, device) -> _SiteB | None:
    """`DP_OUT_PROJ_INJECT_ENV` (and `DP_SITEB_PLAN32_ENV`), read once at build after
    `_out_proj_build` and the hc plan switches whose launches it mirrors: the `_SiteB` of this
    rank, every kernel compiled and one fold checked against the stock site at each distinct
    launch shape (GEMM plan, first-projection plan, up tile); None where it is off. Set where it
    cannot apply (`DP_OUT_PROJ_BF16_ENV` not engaged, another stream count): one warning, None.
    A kernel that does not compile: one warning, that row count keeps the inject kernel. A
    failed check, or one that raises: one warning, the whole switch keeps it (the exception does
    not leave the build, so no rank dies). Nothing here is collective: a rank that declines runs
    the stock site, which no other rank depends on."""
    want = _siteb_wanted(core.max_verify_rows)
    plan32 = _switch(DP_SITEB_PLAN32_ENV)
    if want is None:
        if plan32:
            logger.warning(
                f"{DP_SITEB_PLAN32_ENV} is set but ignored: it rides on {DP_OUT_PROJ_INJECT_ENV}, "
                "which is not set"
            )
        return None
    out = core._out_proj
    why = None
    if out is None:
        why = f"it rides on {DP_OUT_PROJ_BF16_ENV}, which is not engaged at this rank"
    elif core.hc_count != _SITEB_STREAMS or core.stream != _SITEB_STREAMS * _OUT_PROJ_GEOM[0]:
        why = (
            f"the fold is compiled for {_SITEB_STREAMS} streams of the output projections' "
            f"width (this model: {core.hc_count} streams of {core.stream // core.hc_count})"
        )
    if why is not None:
        logger.warning(f"{DP_OUT_PROJ_INJECT_ENV} is set but ignored: {why}")
        return None
    layers = core._decode._layers
    skip = core.ple_layer - 1
    gdn = next((i for i in core.gdn_layers if i != skip), None)
    n, k = _OUT_PROJ_GEOM
    w = core.w[f"l{gdn}_out_w"]
    gemm, first, up, moved, missed = {}, {}, {}, [], []
    shape = {}  # rows -> the launch shape the self-check keys on
    for rows, (mpad, plan) in sorted(out["gdn_out"].items()):
        if not want[0] <= rows <= want[1]:
            continue
        if plan32 and (mpad, plan) == _SITEB_PLAN32[0]:
            mpad, plan = _SITEB_PLAN32[1]
            moved.append(rows)
        form = _hc_first_form(core, rows, True)
        fplan = hc_tc_first_ss_plan(
            rows, form.get("tm", 128), "rows_on_n" in form, "tm64x2" in form
        )
        if rows % 4 or rows > min(_SITEB_MAX_ROWS, _HC_TC_FIRST_MAX_ROWS) or fplan[2] % 4:
            missed.append(rows)
            continue
        tm_up = _hc_up_tile(core, rows, False)
        try:
            x = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
            mid_mm_inject(x, w, plan, x, x[:, :4], x, mpad=mpad, compile_only=True)
            y = torch.empty(rows, core.stream, dtype=torch.bfloat16, device=device)
            ss = torch.empty(rows, 4, n // _INJECT_UNIT_COLS, dtype=torch.float32, device=device)
            hc_tc_first_ss(
                y, ss, layers[gdn].mhc_first, core.hc_lowrank, core.hc_count, core.eps, fplan,
                compile_only=True,
            )  # fmt: skip
            hc_up_precompile(
                core.hc_count, core.hidden, core.hc_lowrank, device, [(False, tm_up)], ss=True
            )
        except Exception as e:  # noqa: BLE001 -- an opt-in form declines a row count it cannot compile
            missed.append(rows)
            logger.warning(
                f"{DP_OUT_PROJ_INJECT_ENV}: a kernel of {rows} rows did not compile at rank "
                f"{core.dp_rank} ({type(e).__name__}: {str(e)[:300]}); it keeps the inject kernel"
            )
            continue
        gemm[rows], first[rows], up[rows] = (mpad, plan), form, tm_up
        shape[rows] = ((mpad, plan), fplan, tm_up)
    attn = frozenset(r for r in out["attn_o"] if r in gemm)
    st = _SiteB(gemm, attn, first, up, skip, tuple(moved))
    if not gemm:
        logger.warning(
            f"{DP_OUT_PROJ_INJECT_ENV} is set but ignored at rank {core.dp_rank}: no row count of "
            f"the output projections' table ({sorted(out['gdn_out'])}) in {want[0]}..{want[1]} "
            f"rows takes the fold (not served: {missed})"
        )
        return None
    # One check a distinct launch shape (each first-projection kind and up tile, not only each
    # GEMM plan), at its smallest row count.
    for key in sorted(set(shape.values()), key=str):
        rows = min(r for r, e in shape.items() if e == key)
        try:
            bad = _siteb_selfcheck(core, st, layers[gdn], rows, device)
        except Exception as e:  # noqa: BLE001 -- an opt-in form declines; the rank must not die
            bad = f"the check at {rows} rows raised {type(e).__name__}: {str(e)[:300]}"
        if bad is not None:
            logger.warning(
                f"{DP_OUT_PROJ_INJECT_ENV} is set but ignored at rank {core.dp_rank}: {bad}; every "
                "MoE-side site keeps qwen4_exp_hc_inject_norm"
            )
            return None
    logger.info(
        f"site-B fold into the output projection ({DP_OUT_PROJ_INJECT_ENV}="
        f"{os.environ.get(DP_OUT_PROJ_INJECT_ENV, '').strip()}): rank {core.dp_rank} of "
        f"{core.dp_size}, GDN rows {sorted(gemm)}, attention rows {sorted(attn)}, every trunk layer "
        f"but {skip}; not served {missed or 'none'}"
        + (f"; {DP_SITEB_PLAN32_ENV}=1 at {moved}" if moved else "")
    )
    return st


def _siteb_arm(core, L, i: int, rows: int, hc, inject) -> bool:
    """`DP_OUT_PROJ_INJECT_ENV`: called by the trunk loop in front of layer `i`'s block. Where the
    layer's site B takes the fold it hands the site's operands to the block's output projection
    (`core._siteb_pending`, read and cleared by `_out_proj`) and returns True."""
    st = getattr(core, "_siteb", None)
    if st is None or i == st.skip:
        return False
    attn = bool(core.is_attn[i])
    if rows not in (st.attn_rows if attn else st.gemm):
        return False
    norm = L.mhc_norm
    if not (
        inject is not None
        and hc.dtype == inject.dtype == norm.dtype == torch.bfloat16
        and tuple(hc.shape) == (rows, core.stream)
        and tuple(inject.shape) == (rows, core.hc_count)
        and norm.numel() == core.stream
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (hc, inject, norm))
    ):
        return False
    core._siteb_pending = _SiteBArm("attn_o" if attn else "gdn_out", hc, inject, norm)
    return True


def _siteb_disarm(core) -> None:
    """Drops a fold `_siteb_arm` armed that the block's output projection did not take."""
    core._siteb_pending = None


def _siteb_site(core, folded: _SiteBFolded, L, st: _SiteB | None = None) -> tuple:
    """`DecodeTarget._hc_moe_site` behind a folded output projection: the site's mixer on the
    deferred-rstd forms. Returns `(hc, mixed, inject, None)`."""
    st = core._siteb if st is None else st
    rows = folded.hc.shape[0]
    low, inject, rstd = qwen4_exp_hc_decode_tc_first_ss(
        folded.y, folded.ss, L.mhc_first, core.hc_lowrank, core.hc_count, core.eps, **st.first[rows]
    )
    mixed = qwen4_exp_hc_up_gate_mean(
        low, L.mhc_up, folded.y, core.hc_count, tm=st.up[rows], rstd=rstd
    )
    return folded.hc, mixed, inject, None


#: Switch, on by default: at a dep4 / dep8 rank on SM103 (`_dp_decode_rank`) two launch orders
#: of a MoE block's round trip. (1) A trunk block launches its shared expert on the side stream
#: from inside the round trip, right after the gather, instead of before it: the gather (one
#: Lamport launch) then reaches the SMs ahead of the shared expert's first GEMM, and the side
#: stream still runs it beside the routed experts. (2) The draft layer's shared expert runs on the
#: side stream beside its round trip, launched the same way, instead of after it on the main
#: stream. The same kernels on the same inputs, so bitwise. Measured on GB300 dep4 inside a step
#: bundle with the draft attention input pack (0.08-0.2 ms a step derived); the index scores'
#: PDL launch that bundle also had is already the decode path's. `0` keeps the order before.
#: Read once, when the target is built.
DP_LAUNCH_ORDER_ENV = "TRTLLM_QWEN4_EXP_DP_LAUNCH_ORDER"


#: Switch, on by default: at a dep8 rank on SM103 (`_dp_decode_rank`, the draft layer's FP8
#: experts in 64-wide windows) the draft MoE's trtllm-gen runner is entered at the tactic
#: `_DRAFT_MOE_TACTICS` names for the call's gathered row count instead of the AutoTuner's cold
#: fallback, which takes the widest valid token tile ([64, 0] at 96 rows, [128, 0] from 112):
#: with about 2 tokens an expert those tiles are mostly padding. Every tactic of the runner
#: computes the same sums in the same order, so the draft tokens are bitwise. GB300 micro (dep8
#: window, us a call): 108.8 -> 72.0 at 96 rows, 171.3 -> 73.2 at 112, 186.8 -> 78.2 at 192;
#: dep8 server step -0.083 ms at 14 requests a rank, -0.055 at 20. `0` keeps the AutoTuner's
#: tactic. Read once, when the target is built.
DP_DRAFT_MOE_TILE_ENV = "TRTLLM_QWEN4_EXP_DP_DRAFT_MOE_TILE"
#: Local draft experts a rank -> `((first rows, last rows, (token tile, config)), ...)`: the
#: gathered row counts of one runner call that take a tactic. Measured at dep8 only; 64..95 rows
#: (10..11 requests a rank) measured no gain ([8, 8]: within noise) and keep the fallback.
_DRAFT_MOE_TACTICS = {64: ((96, 192, (16, 8)),)}  # capacity: fixed: a dispatch table


def _draft_moe_tactics_build(core, device) -> dict | None:
    """`DP_DRAFT_MOE_TILE_ENV`, read once at build: `{}` (filled per row count on first use by
    `_draft_moe_tactic`) at a dep8 rank on SM103 whose draft experts are the FP8 runner's, None
    elsewhere. No collective, nothing compiled (the runner's kernels are precompiled cubins)."""
    if not (
        _switch(DP_DRAFT_MOE_TILE_ENV, default=True)
        and core.mtp_enabled
        and _dp_decode_rank(core, device)
        and core.mtp_local_experts in _DRAFT_MOE_TACTICS
        and getattr(core, "_mtp", None) is not None
        and core._mtp.moe_gscale is None
    ):
        return None
    rules = _DRAFT_MOE_TACTICS[core.mtp_local_experts]
    logger.info(
        f"draft MoE tactics: rank {core.dp_rank} of {core.dp_size}, "
        + ", ".join(f"{list(t)} at {lo}..{hi} gathered rows" for lo, hi, t in rules)
    )
    return {}


def _draft_moe_tactic(core, rows: int) -> tuple | None:
    """`DP_DRAFT_MOE_TILE_ENV`'s tactic for a draft MoE runner call of `rows` gathered rows, or
    None (the AutoTuner's). A table tactic the runner does not list valid at `rows` is not
    taken. Host only; settled once per row count."""
    memo = core._draft_moe_tactics
    if memo is None:
        return None
    if rows not in memo:
        tactic = next(
            (t for lo, hi, t in _DRAFT_MOE_TACTICS[core.mtp_local_experts] if lo <= rows <= hi),
            None,
        )
        if tactic is not None and list(tactic) not in fp8_moe_valid_tactics(
            core.topk, core.hidden, core.moe_inter, core.mtp_local_experts, rows
        ):
            tactic = None
        memo[rows] = tactic
    return memo[rows]


#: Switch, on by default: at a dep4 / dep8 rank on SM103 whose MoE round trip is the Lamport
#: transport (`DP_MOE_COMM_ENV`), a trunk MoE block of a captured generation-only step ends with ONE
#: launch where it had two: the Lamport reduce-scatter also forms the block output (`routed +
#: sigmoid(gate) * shared`), injects it into the hyper-connection streams and takes the next site's
#: grouped norm (`ops/moe/dep_lamport_rs_inject`), so the next site launches no
#: `qwen4_exp_hc_inject_norm` (site A of the next layer, or the stack-final site). Never the layer
#: before the PLE layer (its pending block is added explicitly), an eager or a padded step, or a row
#: count outside `_DP_RS_INJECT_ROWS`. The same bytes travel as with the plain reduce-scatter, and
#: `hc_out` / `normed` are bitwise what the two launches write: the launch runs the reduce-scatter's
#: and the inject-norm kernel's own instructions. The build compiles every layout, then checks every
#: row count on every rank against the two launches (three reduce-scatters a row count, collective)
#: and keeps the row counts that hold on every rank. GB300: -0.10..-0.14 ms a step at dep4 and dep8
#: (the launch and its HBM round trip, 47 sites a step). `0` keeps the two launches. Read once,
#: after load.
DP_RS_INJECT_ENV = "TRTLLM_QWEN4_EXP_DP_RS_INJECT"
#: With `DP_RS_INJECT_ENV`: the `(streams a CTA, CTA limit)` table the fused launch picks its grid
#: from (`dep_lamport_rs_inject.streams_per_cta`), as `1:304,2:608`. Unset: the measured table of the
#: group size (`dep_lamport_rs_inject.LAYOUTS`: dep8 keeps four CTAs a row up to 76 rows, dep4 up
#: to 48). Any table reads the same bytes and computes the same bits. For A/Bs.
DP_RS_INJECT_LAYOUTS_ENV = "TRTLLM_QWEN4_EXP_DP_RS_INJECT_LAYOUTS"
#: Rows a rank the fused launch serves (4 rows a request under MTP-3): the micro's range.
_DP_RS_INJECT_ROWS = (4, 256)  # capacity: fixed: dispatch thresholds
#: Switch, on by default, with `DP_RS_INJECT_ENV`: where the decode route front serves the
#: gathered rows (`DP_ROUTE_FRONT_ENV`), the trtllm-gen runner is entered without its finalize and
#: the fused launch forms each partial itself from the permuted FC2 rows, the routing weights and
#: the expanded -> permuted map, with the finalize's statements (slot-order fp32 fma from +0,
#: skipping -1, one rounding): the finalize launch and the `[W * R, H]` store and reload go. Same
#: bytes exchanged, bitwise (checked at build with the fused launch). Only at row counts whose
#: finalize grid is one resident wave (`dep_lamport_rs_inject.fin_resident`: up to 152 rows a rank
#: on GB300 at dep4 and dep8), since every CTA of its rows publishes. GB300 dep8: -0.103 ms a step
#: at 12 requests a rank. `0` keeps the finalize. Read once, after load.
DP_MOE_FIN_RS_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_FIN_RS"
#: Switch, on by default (and only with PDL on, `TRTLLM_ENABLE_PDL`): the finalize form launches
#: with programmatic dependent launch. Its `griddepcontrol.wait` is its first statement and every
#: load of the runner's output is coherent, so nothing is read before the runner completes; the
#: launch and its CTA ramp overlap the runner's tail. `0`: a plain launch.
DP_MOE_FIN_RS_PDL_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_FIN_RS_PDL"
#: The draft layer's MoE under attention DP, by parts, on a dep4 / dep8 rank's captured draft
#: passes (with `DP_LAUNCH_ORDER_ENV`, the side-stream form of the draft MoE). A comma list of:
#: - `combine`: the draft MoE's Lamport reduce-scatter is the fused launch of `DP_RS_INJECT_ENV`
#:   over the draft residual with the draft head's grouped norm: it forms `routed + sigmoid(gate) *
#:   shared`, injects it and normalizes, in place of the reduce-scatter and
#:   `mtp_moe_combine_inject` (bitwise: the same combine and injection instructions). Needs
#:   `DP_RS_INJECT_ENV` built; the pass's row count must be one the build checked.
#: - `headnorm` (with `combine`): the draft head (`_mtp_head_final`) takes that norm instead of
#:   launching `qwen4_exp_hc_inject_norm` again, when its rows are the combine's output tensor.
#: - `route`: the FP8 draft runner routes from the router logits itself (its Renormalize routing)
#:   instead of `renorm_moe_routing_op` + the pre-routed entry: one launch less. The runner's
#:   in-kernel routing is not certified by the op (measured bitwise on dep4), so opt-in.
#: Unset: `combine,headnorm`; `1` or `all`: every part; `0`: none. GB300 dep4 at 12 requests a rank:
#: the three with the draft glue -0.17 ms a step. Read once, after load.
DP_DRAFTPACK_ENV = "TRTLLM_QWEN4_EXP_DP_DRAFTPACK"
_DP_DRAFTPACK_PARTS = ("combine", "headnorm", "route")
_DP_DRAFTPACK_DEFAULT = frozenset({"combine", "headnorm"})


class _RsInjected(NamedTuple):
    """What a MoE block hands the next site under `DP_RS_INJECT_ENV` instead of its routed rows:
    the residual with the block injected, and that site's grouped norm of it."""

    hc: torch.Tensor
    normed: torch.Tensor


class _Unfinalized(NamedTuple):
    """The runner's output without its finalize over the group's gathered rows
    (`DP_MOE_FIN_RS_ENV`): permuted FC2 rows, routing weights, the expanded -> permuted map."""

    g2: torch.Tensor
    w: torch.Tensor
    e2p: torch.Tensor


class _DpRsInject:
    """What `DP_RS_INJECT_ENV` built (`_dp_rs_inject_build`): the compiled launches (`rs`), the row
    counts a rank its trunk blocks take them at (`rows`) and of those the finalize form's
    (`fin_rows`), the draft passes' row counts (`draft_rows`), and the draft head's stash."""

    def __init__(self, rs, rows: frozenset, fin_rows: frozenset, draft_rows: frozenset) -> None:
        self.rs = rs
        self.rows = rows
        self.fin_rows = fin_rows
        self.draft_rows = draft_rows
        #: The draft combine's `_RsInjected` of the pass in flight (`headnorm`), host state.
        self.stash: _RsInjected | None = None


def _dp_draftpack_parts(core, rsi, device) -> frozenset:
    """`DP_DRAFTPACK_ENV`, read once at build: the parts this rank's captured draft passes take."""
    raw = os.environ.get(DP_DRAFTPACK_ENV, "").strip().lower()
    if not raw:
        parts, explicit = set(_DP_DRAFTPACK_DEFAULT), False
    elif raw in _SWITCH_OFF:
        return frozenset()
    elif raw in _SWITCH_ON or raw == "all":
        parts, explicit = set(_DP_DRAFTPACK_PARTS), True
    else:
        parts, explicit = {p.strip() for p in raw.split(",") if p.strip()}, True
        if not parts <= set(_DP_DRAFTPACK_PARTS) or (
            "headnorm" in parts and "combine" not in parts
        ):
            raise ValueError(
                f"{DP_DRAFTPACK_ENV}={raw!r}: expected 0, 1, all or a comma list of "
                f"{', '.join(_DP_DRAFTPACK_PARTS)} (headnorm only with combine)"
            )
    why = ""
    if not (core.mtp_enabled and getattr(core, "dp_launch_order", False)):
        why = f"it serves the draft MoE's side-stream form ({DP_LAUNCH_ORDER_ENV}) with MTP on"
    elif getattr(core, "_mtp", None) is None or core._mtp.moe_gscale is not None:
        why = "the draft experts are not the FP8 block-scale runner's"
    if why:
        if explicit:
            logger.warning(f"{DP_DRAFTPACK_ENV}={raw} is set but ignored: {why}")
        return frozenset()
    if rsi is None and parts & {"combine", "headnorm"}:
        if explicit:
            logger.warning(
                f"{DP_DRAFTPACK_ENV}: parts combine / headnorm are set but ignored: the fused "
                f"reduce-scatter ({DP_RS_INJECT_ENV}) is not built"
            )
        parts -= {"combine", "headnorm"}
    return frozenset(parts)


def _dp_rs_inject_operands(core, rows: int, gen, device) -> tuple:
    """Random site operands of `rows` rows in the layouts the decode step hands the launch:
    `(hc, inject, norm weight, shared, gate logits)`, the gate a column of a wider buffer."""
    bf = torch.bfloat16
    width = core.hc_count * core.hidden
    hc = torch.randn([rows, width], generator=gen, device=device).to(bf)
    inject = torch.randn([rows, core.hc_count], generator=gen, device=device).to(bf)
    norm = (0.1 * torch.randn([width], generator=gen, device=device)).to(bf)
    shared = torch.randn([rows, core.hidden], generator=gen, device=device).to(bf)
    gate = torch.randn([rows, 3], generator=gen, device=device).to(bf)[:, 1:2]
    return hc, inject, norm, shared, gate


def _bitwise(got: torch.Tensor, want: torch.Tensor) -> bool:
    return got.shape == want.shape and torch.equal(got.view(torch.int16), want.view(torch.int16))


def _dp_rs_inject_check(core, rs, rows: int, fin: bool, gen, device, draft: bool = False) -> tuple:
    """At `rows` rows a rank, eagerly: three reduce-scatters of the workspace, each a collective
    every rank of the group launches in the same order -- the plain one (then
    `qwen4_exp_hc_inject_norm`: the reference), the fused launch on the same partials, and with
    `fin` the finalize form over the runner's output of the gathered rows those partials are the
    finalized form of (else the fused launch again: a peer may run its finalize form, and this
    rank's result then shows whether that peer's partials were the finalized ones). A launch that
    raises on this rank is replaced by the plain reduce-scatter, which exchanges the same bytes, so
    no peer waits; so is the finalize form when forming the runner's output raised here (before
    any launch: the partials are then random rows). With `draft` (a draft pass's row count,
    `DP_DRAFTPACK_ENV`) the fused launch is also held, locally, to the draft path it replaces:
    `mtp_moe_combine_inject` and the draft head's `qwen4_exp_hc_inject_norm` without injection.
    Returns `(fused differs, third launch differs, draft path differs)`, each "" when bitwise."""
    trip = core._lamport_trip
    g = trip.world * rows
    hc, inject, norm, shared, gate = _dp_rs_inject_operands(core, rows, gen, device)
    unfin, prep = None, ""
    if fin:
        try:
            st, L = core._route_front, core._decode._layers[0]
            x = torch.randn([g, core.hidden], generator=gen, device=device).to(torch.bfloat16)
            logits = _decode_router(x, L.router)
            partial = _route_front_run(core, st, x, L, 0, logits)[0]
            fc2, _, _, _, wts = _route_front_run(core, st, x, L, 0, logits, finalize=False)
            unfin = _Unfinalized(fc2, wts, st.views[g][0][0].view(g, core.topk))
        except Exception as e:  # noqa: BLE001 -- before any launch; the peers still need 3
            prep = f"finalize: the runner's output: {type(e).__name__}: {e}"
            unfin = None
    if unfin is None:
        partial = torch.randn([g, core.hidden], generator=gen, device=device).to(torch.bfloat16)
        partial[:, ::97] = -0.0
    routed = dep_lamport_round_trip(partial, trip, True)
    ref = qwen4_exp_hc_inject_norm(hc, routed, inject, norm, core.eps, core.hc_count, shared, gate)
    out, fused_out = [], None
    for form in ("fused", "finalize"):
        try:
            if form == "finalize" and unfin is not None:
                why = rs.fin_operands_refuse(rows, *unfin)
                if why:
                    raise RuntimeError(f"the runner's output: {why}")
                got = rs.run_fin(*unfin, hc, inject, norm, core.eps, shared, gate)
            else:
                got = rs.run(partial, hc, inject, norm, core.eps, shared, gate)
        except Exception as e:  # noqa: BLE001 -- the peers' launch still needs this rank's bytes
            dep_lamport_round_trip(partial, trip, True)
            out.append(f"{form}: {type(e).__name__}: {e}")
            continue
        if form == "fused":
            fused_out = got
        bad = [
            n
            for n, a, b in (("hc_out", got[0], ref[0]), ("normed", got[1], ref[1]))
            if not _bitwise(a, b)
        ]
        out.append(f"{form}: {', '.join(bad)} differ" if bad else "")
    out[1] = out[1] or prep
    drafted = ""
    if draft:
        try:
            if fused_out is None:
                raise RuntimeError("the fused launch raised")
            want_hc = mtp_moe_combine_inject(hc, routed, gate, shared, inject)
            _, want_n = qwen4_exp_hc_inject_norm(want_hc, None, None, norm, core.eps, core.hc_count)
            bad = [
                n
                for n, a, b in (("hc", fused_out[0], want_hc), ("head norm", fused_out[1], want_n))
                if not _bitwise(a, b)
            ]
            drafted = f"draft: {', '.join(bad)} differ" if bad else ""
        except Exception as e:  # noqa: BLE001 -- local; the collectives above all ran
            drafted = f"draft: {type(e).__name__}: {e}"
    torch.cuda.synchronize(device)
    return out[0], out[1], drafted


def _dp_rs_inject_build(core, device) -> "_DpRsInject | None":
    """`DP_RS_INJECT_ENV` (+ `DP_MOE_FIN_RS_ENV`, `DP_MOE_FIN_RS_PDL_ENV`,
    `DP_RS_INJECT_LAYOUTS_ENV`), read once at build (`post_load_weights`, after the route and
    router fronts). None where it is off or cannot take
    effect: not a dep4 / dep8 rank on SM103 on the Lamport transport, another geometry, no
    CUDA-graph row count in range. Otherwise every layout's launch is compiled now (no serving step
    or capture meets a compile), the group agrees the compiles, and every row count is checked
    (`_dp_rs_inject_check`, collective) and agreed: a row count that differs on any rank is dropped
    on every rank, with one warning. Collective: every rank of the group runs it."""
    if not _switch(DP_RS_INJECT_ENV, default=True):
        return None
    trip = core._lamport_trip
    if trip is None or not _dp_decode_rank(core, device):
        if os.environ.get(DP_RS_INJECT_ENV, "").strip():
            logger.warning(
                f"{DP_RS_INJECT_ENV} is set but ignored: it serves a dep4 / dep8 rank on SM103 on "
                f"the Lamport transport ({DP_MOE_COMM_ENV})"
            )
        return None
    if not _rs_inject_op.supported(core.hidden, core.hc_count):
        logger.warning(
            f"{DP_RS_INJECT_ENV} is set but ignored: the fused launch is written for a hidden in "
            f"(2048, 4096] and up to 8 streams (this model: {core.hidden} x {core.hc_count})"
        )
        return None
    raw = os.environ.get(DP_RS_INJECT_LAYOUTS_ENV, "").strip()
    layouts = _rs_inject_op.parse_layouts(raw) if raw else _rs_inject_op.LAYOUTS[core.dp_size]
    if not raw and "fin" in _dp_hicap(core):
        # `DP_HICAP_ENV`'s `fin`: every layout ends at one resident wave, so past it a row's
        # streams share one CTA (the same bits) and the fused finalize stays resident.
        layouts = tuple((spp, min(limit, _DP_HICAP_FIN_CTAS)) for spp, limit in layouts)
    lo, hi = _DP_RS_INJECT_ROWS
    hi = min(hi, trip.max_rows, _rs_inject_op.MAX_ROWS)
    trunk = set(_graph_rows(core, lo, hi))
    sizes = getattr(core.model_config, "cuda_graph_batch_sizes", None) or ()
    draft = {int(b) for b in sizes if lo <= int(b) <= hi} if core.mtp_enabled else set()
    if not trunk | draft:
        return None
    st = core._route_front
    fin = (
        _switch(DP_MOE_FIN_RS_ENV, default=True)
        and st is not None
        and any(trip.world * r in st.tactics for r in trunk)
    )
    sms = torch.cuda.get_device_properties(device).multi_processor_count if fin else 0

    def fin_at(r: int) -> bool:
        # The route front serves the gathered rows and the finalize grid is one resident wave
        # (`fin_resident`: dep8 up to 152 rows a rank, dep4 up to 152; the measured
        # dep4 caps 160..208 keep the runner's finalize, as they ran).
        return (
            fin
            and trip.world * r in st.tactics
            and _rs_inject_op.fin_resident(r, core.hc_count, layouts, sms)
        )

    fin = fin and any(fin_at(r) for r in trunk)
    pdl = fin and _switch(DP_MOE_FIN_RS_PDL_ENV, default=True) and bool(get_env_enable_pdl())
    from tensorrt_llm._torch.distributed.communicator import Distributed

    dist = Distributed.get(core.mapping)
    rs, error = None, ""
    try:
        with torch.cuda.device(device):
            rs = _rs_inject_op.build(trip, core.hc_count, layouts, fin, core.topk, pdl)
    except Exception as e:  # noqa: BLE001 -- every rank decides together below
        error = f"{type(e).__name__}: {e}"
    errors = [e for e in dist.tp_allgather(error) if e]
    if errors:
        logger.warning(
            f"{DP_RS_INJECT_ENV}: not engaged at rank {core.dp_rank}: the fused launch did not "
            f"compile on {len(errors)} rank(s): {errors[0]}"
        )
        return None
    gen = torch.Generator(device=device).manual_seed(20261126)
    bad = []
    with torch.cuda.device(device):
        for rows in sorted(trunk | draft):
            fin_here = rows in trunk and fin_at(rows)
            bad.append(
                (rows, *_dp_rs_inject_check(core, rs, rows, fin_here, gen, device, rows in draft))
            )
    every = dist.tp_allgather(bad)
    off = {r for rank_bad in every for r, fused, _, _ in rank_bad if fused}
    fin_off = {r for rank_bad in every for r, fused, final, _ in rank_bad if fused or final}
    draft_off = {r for rank_bad in every for r, fused, _, drafted in rank_bad if fused or drafted}
    for r, fused, final, drafted in (x for rank_bad in every for x in rank_bad):
        if fused or final:
            logger.warning(f"{DP_RS_INJECT_ENV}: {r} rows a rank stay unfused: {fused or final}")
        if drafted and not fused:
            logger.warning(f"{DP_DRAFTPACK_ENV}: {r} draft rows keep the draft combine: {drafted}")
    rows = frozenset(trunk - off)
    fin_rows = frozenset(r for r in rows if fin_at(r) and r not in fin_off)
    built = _DpRsInject(rs, rows, fin_rows, frozenset(draft - draft_off))
    logger.info(
        f"DP rs inject: rank {core.dp_rank} of {core.dp_size}, the trunk MoE reduce-scatter injects "
        f"and normalizes for the next site at rows {sorted(rows)} a rank, the finalize folded in at "
        f"{sorted(fin_rows)} (PDL {pdl}), draft rows {sorted(built.draft_rows)}; layouts "
        f"{list(layouts)}"
    )
    return built


def _dp_rs_inject_next(
    core, captured: bool, pad_rows: int, rows: int, i: int, layers, final_norm, hc, inject
):
    """`DP_RS_INJECT_ENV`: what trunk layer `i`'s MoE block injects and normalizes for the site
    behind it -- `(hc, inject, norm weight)`, the weight of layer `i + 1`'s site A or, behind the
    last layer, the stack-final site's -- or None where the block keeps its plain reduce-scatter:
    not built, an eager or padded step, a row count the build did not keep, and the layer before
    the PLE layer (its pending block is added explicitly). Host values, the same on every rank."""
    rsi = getattr(core, "_rs_inject", None)
    if rsi is None or not captured or pad_rows or rows not in rsi.rows or i + 1 == core.ple_layer:
        return None
    return hc, inject, layers[i + 1].ahc_norm if i + 1 < len(layers) else final_norm


#: `(rows, why)` the fused launch refused in this process (one warning each).
_RS_INJECT_REFUSED: set = set()


def _rs_inject_tail(core, rsi: _DpRsInject, site, operands, main, side):
    """The launch the round trip makes in place of its reduce-scatter (`tail(partial)`): it joins
    the side stream first (the fused kernel reads the shared expert's output and gate logit
    before it polls) and returns `_RsInjected`, or None where this call's operands are not in the
    kernel's layout (the plain reduce-scatter then runs: the same bytes travel). `site` is the
    next site's `(hc, inject, norm weight)`, `operands()` the shared expert's `(shared, gate
    logits)` once the window has launched it. With an `_Unfinalized` partial (`DP_MOE_FIN_RS_ENV`,
    chosen by the window after `refuses_fin`) it runs the finalize form."""
    hc, inject, norm = site

    def tail(partial):
        shared, gate = operands()
        if isinstance(partial, _Unfinalized):
            why = rsi.rs.fin_operands_refuse(hc.shape[0], *partial)
            if why is not None:
                raise RuntimeError(f"{DP_MOE_FIN_RS_ENV}: the runner's output: {why}")
            main.wait_stream(side)
            return _RsInjected(*rsi.rs.run_fin(*partial, hc, inject, norm, core.eps, shared, gate))
        why = rsi.rs.refuses(partial, hc, inject, norm, shared, gate)
        if why is not None:
            if (hc.shape[0], why) not in _RS_INJECT_REFUSED:
                _RS_INJECT_REFUSED.add((hc.shape[0], why))
                logger.warning(
                    f"{DP_RS_INJECT_ENV}: {hc.shape[0]} rows keep the plain reduce-scatter: {why}"
                )
            return None
        main.wait_stream(side)
        return _RsInjected(*rsi.rs.run(partial, hc, inject, norm, core.eps, shared, gate))

    return tail


def _route_front_unfinalized(core, rows, L: _LayerWeights, router_mm) -> "_Unfinalized | None":
    """`_route_front_experts` without the runner's finalize (`DP_MOE_FIN_RS_ENV`): the gathered
    `rows`' route front and the runner's permuted FC2 rows, weights and index map, or None where
    the route front does not serve them."""
    st = core._route_front
    g = rows.shape[0]
    layer = st.layers.get(L.fc1_w.data_ptr()) if st is not None else None
    if layer is None or g not in st.tactics or rows.stride(1) != 1 or rows.stride(0) != core.hidden:
        return None
    logits = _tep_front_logits(st, rows, L, router_mm)
    fc2, _, _, _, wts = _route_front_run(core, st, rows, L, layer, logits, None, finalize=False)
    return _Unfinalized(fc2, wts, st.views[g][layer][0].view(g, core.topk))


#: Opt-in (class S). At a dep8 rank on SM103 on the host-index Lamport transport
#: (`DP_MOE_COMM_ENV=lamport_step`) with the route front, a captured decode step's trunk MoE block
#: gathers each row's NVFP4 form and its router logits (1,232 lanes a row) instead of its bf16 form
#: (2,560): the router runs over the rank's OWN rows, the packed gather
#: (`ops/moe/dep_lamport_pack_gather`) quantizes them inside the launch, and the route front's
#: routing cluster alone (`route_front_only`) routes the gathered logits before the same pre-routed
#: runner call. The codes and scales are the route front's bits over the same rows; the logits come
#: from the router GEMM over R rows instead of the group's W R, which may pick another kernel and
#: move a logit's last bit (a near-tie of the top-10 can flip; every rank routes the owner's bits,
#: so the group agrees on every row's experts). GB300 dep8: -0.015 / -0.028 / -0.044 / -0.047 ms a
#: step at 8 / 10 / 11 / 12 requests a rank. Read once, after load.
DP_MOE_PACK_GATHER_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_PACK_GATHER"
#: With `DP_MOE_PACK_GATHER_ENV`: `mid` runs the owner's router on `mid_rows_mm`'s bf16 kernel (a
#: PDL launch that streams the router weight before its wait; checked at build against the fp32
#: product, one bf16 ulp), else the decode router. Opt-in (class S, as the router front).
DP_MOE_PACK_ROUTER_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_PACK_ROUTER"
#: Default on where `DP_MOE_PACK_GATHER_ENV` is engaged (bitwise; at `_DP_MOE_OWNK_ROWS`, and only
#: at row counts the build checked on every rank). Each row's owner routes it inside the packed
#: gather (`PackGather(ownk=True)`: one warp a row runs the route front's own `route_one_token`
#: over the owner's sanitized logits), and the row carries its 10 experts and weights (3 vectors)
#: instead of its 512 logits (64). The routing-only front then builds the tables from them
#: (`route_front_only(ids_in=True)`) instead of every rank routing all `W R` gathered rows on one
#: 16-CTA cluster. The experts, weights and tables are the bits the packed gather's routing-only
#: front writes. GB300 dep8 server A/B (two nodes of one rack, packed gather on both sides):
#: -0.091 / -0.121 / -0.137 ms a step at 18 / 20 / 24 requests a rank; -0.026 at 12 (noise).
#: Values: unset or `1` -- `_DP_MOE_OWNK_ROWS`; `all` -- every gathered row count the packed
#: gather serves past the lean routing front (`LEAN_MAX_ROWS`); `0` -- off. Read once, after load.
DP_MOE_OWNK_ENV = "TRTLLM_QWEN4_EXP_DP_MOE_OWNK"
#: Gathered rows the default serves: the routing front's two-tokens-a-warp launch up to the
#: largest measured cap (17..28 requests a rank at dep8 MTP-3; caps 18 / 20 / 24 by A/B, 26 / 28
#: in the dep8 deployments that ran it). At a token a warp it measured within noise; caps 29..32
#: (928..1024 rows) were never run and take `all`.
_DP_MOE_OWNK_ROWS = (513, 896)  # capacity: fixed: dispatch thresholds
#: The most gathered rows a routing-only front takes (`moe_route_front__decode.MAX_ROWS`).
_DP_PACK_MAX_GATHERED = 1024  # capacity: fixed: the route front's launch bound


class _DpPack:
    """What `DP_MOE_PACK_GATHER_ENV` built (`_dp_pack_build`): the packed gather, the gathered row
    counts whose routing-only front compiled and checked (`rows`), and with `mid` each layer's
    router weight (`mids`, by its pointer) and the own row counts it serves (`mid_rows`); with
    `DP_MOE_OWNK_ENV` the owner top-k gather (`ownk`) and the gathered rows it serves
    (`ownk_rows`, `_dp_ownk_build`)."""

    def __init__(self, gather, rows: frozenset, mids: dict, mid_rows: frozenset) -> None:
        self.gather = gather
        self.rows = rows
        self.mids = mids
        self.mid_rows = mid_rows
        self.ownk = None
        self.ownk_rows: frozenset = frozenset()


def _dp_ownk_wanted(core=None) -> tuple | None:
    """`DP_MOE_OWNK_ENV` as a `(first, last)` range of gathered rows at `core`'s rank, None off."""
    raw = os.environ.get(DP_MOE_OWNK_ENV, "")
    value = raw.strip().lower()
    if value in _SWITCH_OFF:
        return None
    if not value or value in _SWITCH_ON:
        return _DP_MOE_OWNK_ROWS
    if value == "all":
        # `DP_HICAP_ENV`'s `route`: every row count of the wider route front too.
        hicap = _DP_HICAP_ROUTE_MAX_ROWS if "route" in _dp_hicap(core) else 0
        return (1, max(_DP_PACK_MAX_GATHERED, hicap))
    raise ValueError(f"{DP_MOE_OWNK_ENV}={raw!r}: expected unset, 0, 1 or all")


def _dp_ownk_logits(rows: int, experts: int, offset: int, local: int, gen, device, salt: int):
    """`[rows, experts]` bf16 router logits for the owner top-k check, row `r` of kind
    `(r + salt) % 5`: plain; lifted on the expert window at `offset` (the caller names it by the
    owning rank, the same on every rank); coarse (bf16 ties inside the
    top-10 and at its edge); one lane's 16 experts on top (the fast path runs dry: the full sort);
    +-0.0 ties (the sanitized -0.0 lanes)."""
    lg = torch.randn([rows, experts], generator=gen, device=device)
    kinds = (torch.arange(rows, device=device) + salt) % 5
    lifted = lg.clone()
    lifted[:, offset : offset + local] += 3.0
    lg = torch.where((kinds == 1)[:, None], lifted, lg)
    lg = torch.where((kinds == 2)[:, None], (lg * 2).round() / 2, lg)
    lane = (torch.arange(experts, device=device) // 16)[None, :]
    owner = (torch.arange(rows, device=device) % (experts // 16))[:, None]
    lg = torch.where((kinds == 3)[:, None] & (lane == owner), lg + 8.0, lg)
    zeros = torch.zeros_like(lg)
    zeros[:, ::3] = -0.0
    zeros[:, ::37] = 1.0
    lg = torch.where((kinds == 4)[:, None], zeros, lg)
    return lg.to(torch.bfloat16)


def _dp_pack_check(core, pk: _DpPack, st, rows: int, device, ownk=None) -> str:
    """At `rows` rows a rank, eagerly: one packed gather (a collective every rank launches) of rows
    and logits every rank draws from the same per-rank seeds, against `fp4_quantize` of the
    group's sanitized rows (its codes as the transport delivers them: a 16-bit half equal to the
    marker 0x8000, an E2M1 -0 after three +0, arrives as 0x0000, the same zeros) and the sanitized
    logits, and the routing-only front against the route front over the gathered logits (experts,
    weights, tables). "" when bitwise. With `ownk` (the owner top-k gather, `_dp_ownk_build`) that
    gather runs over logits of every kind `_dp_ownk_logits` draws; its codes are held the same
    way, its routing rows to `routed_rows` of the routing-only front over the group's sanitized
    logits, and the front reading them (`ids_in`) to that front's experts, weights and tables.

    The gather is the check's only collective, so every rank launches it: a rank whose inputs
    fail to build (an allocation that raises on that rank alone) launches it over zeros and
    reports the failure, instead of leaving its peers polling for its rows."""
    trip = core._lamport_trip
    g = trip.world * rows
    L = core._decode._layers[0]
    me = core.dp_rank
    xs, lgs, failed = [], [], ""
    try:
        for p in range(trip.world):
            gen = torch.Generator(device=device).manual_seed(20261127 + 1009 * p + rows)
            x = torch.randn([rows, core.hidden], generator=gen, device=device).to(torch.bfloat16)
            x[:, ::131] = -0.0
            xs.append(x)
            if ownk is None:
                lg = torch.randn([rows, core.num_experts], generator=gen, device=device)
                lgs.append(lg.to(torch.bfloat16))
            else:
                # Rank p's rows are lifted on a window named by p, not by this rank's own
                # `expert_offset`: every rank must draw the same group's logits.
                lgs.append(
                    _dp_ownk_logits(
                        rows,
                        core.num_experts,
                        (p * core.local_experts) % core.num_experts,
                        core.local_experts,
                        gen,
                        device,
                        p,
                    )
                )
        mine = (xs[me], lgs[me].contiguous())
    except Exception as e:  # noqa: BLE001 -- reported below, after the collective
        failed = f"inputs raised {type(e).__name__}: {e}"
        mine = (
            torch.zeros([rows, core.hidden], dtype=torch.bfloat16, device=device),
            torch.zeros([rows, core.num_experts], dtype=torch.bfloat16, device=device),
        )
    xq, xsf, lg = (pk.gather if ownk is None else ownk)(*mine, L.moe_gscale)
    if failed:
        torch.cuda.synchronize(device)
        return failed
    gathered = torch.cat([dep_lamport_sanitized(x) for x in xs])
    rxq, rxsf = fp4_quantize(gathered, L.moe_gscale, _SF_VEC, False, False)
    logits = torch.cat([dep_lamport_sanitized(t) for t in lgs])
    checked = [
        ("codes", xq, dep_lamport_transported_codes(rxq)),
        ("scales", xsf, rxsf.reshape(g, -1)),
    ]
    if ownk is None:
        checked.append(("logits", lg, logits))
    bad = [
        name
        for name, a, b in checked
        if not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    ]
    tile = int(st.tactics[g][0])
    flat = st.flats[g][0].clone()
    window = (tile, core.expert_offset, core.local_experts, core.hidden, st.fast2.get(g, 0))
    early = _route_front_early_kw(st, g)
    if ownk is None:
        ids, wts = _route_front_only(lg, flat, *window, **early)
        _, rxq2, _, rids, rwts = _route_front_run(core, st, gathered, L, 0, lg)
        want = st.flats[g][0]
    else:
        want = st.flats[g][0].clone()
        rids, rwts = _route_front_only(logits, want, *window, **early)
        if not torch.equal(lg.view(torch.int16), _routed_rows(rids, rwts)):
            bad.append("owner routing")
        ids, wts = _route_front_only(lg, flat, *window, True, **early)
    if not torch.equal(ids, rids) or not torch.equal(wts.view(torch.int16), rwts.view(torch.int16)):
        bad.append("routing")
    if not torch.equal(flat, want):
        bad.append("tables")
    torch.cuda.synchronize(device)
    return ", ".join(bad)


def _dp_pack_build(core, device) -> "_DpPack | None":
    """`DP_MOE_PACK_GATHER_ENV` (+ `DP_MOE_PACK_ROUTER_ENV`), read once at build (after the route
    front and the fused reduce-scatter). None where it is off or cannot take effect (one warning):
    not a dep8 rank on SM103 on the host-index Lamport transport with the route front. Otherwise
    the packed gather, each routing-only front and with `mid` each own-row router are compiled, the
    group agrees the builds, every row count is checked (`_dp_pack_check`, one collective each)
    and a row count that differs on any rank is dropped on every rank. Collective."""
    if not _switch(DP_MOE_PACK_GATHER_ENV):
        return None
    trip, st = core._lamport_trip, core._route_front
    mode = os.environ.get(DP_MOE_PACK_ROUTER_ENV, "").strip().lower()
    if mode not in ("", "mid"):
        raise ValueError(f"{DP_MOE_PACK_ROUTER_ENV}={mode!r}: expected unset or mid")
    why = ""
    if core.dp_size != 8 or trip is None or trip.steps is None or not _dp_decode_rank(core, device):
        why = (
            f"it serves a dep8 rank on SM103 on the host-index Lamport transport "
            f"({DP_MOE_COMM_ENV}=lamport_step)"
        )
    elif st is None or not st.tactics:
        why = f"it needs the dep8 route front ({DP_ROUTE_FRONT_ENV})"
    if why:
        logger.warning(f"{DP_MOE_PACK_GATHER_ENV} is set but ignored: {why}")
        return None
    from tensorrt_llm._torch.distributed.communicator import Distributed

    dist = Distributed.get(core.mapping)
    pk, error = None, ""
    try:
        with torch.cuda.device(device):
            gather = _PackGather(trip, core.hidden, core.num_experts)
            rows = set()
            for g in sorted(st.tactics):
                tile = int(st.tactics[g][0])
                if _route_front_compile(
                    g,
                    core.hidden,
                    core.hidden,
                    core.local_experts,
                    tile,
                    st.fast2.get(g, 0),
                    True,
                    **_route_front_early_kw(st, g),
                ):
                    rows.add(g)
            mids, mid_rows = {}, set()
            if mode == "mid":
                layers = core._decode._layers
                for L in layers:
                    mids[L.router.data_ptr()] = MidWeight(L.router.t())
                own = sorted({g // trip.world for g in rows if g // trip.world <= _MID_ROWS_MAX})
                if own:
                    mw = mids[layers[0].router.data_ptr()]
                    mid_rows_precompile(mw, min(own), max(own))
                    gen = torch.Generator(device=device).manual_seed(20261127)
                    for r in own:
                        if _dp_router_front_check(mw, r, gen, device) == 0:
                            mid_rows.add(r)
            pk = _DpPack(gather, frozenset(rows), mids, frozenset(mid_rows))
    except Exception as e:  # noqa: BLE001 -- every rank decides together below
        error = f"{type(e).__name__}: {e}"
    errors = [e for e in dist.tp_allgather(error) if e]
    if errors:
        logger.warning(
            f"{DP_MOE_PACK_GATHER_ENV} is set but not engaged at rank {core.dp_rank}: "
            f"{len(errors)} rank(s) failed the build: {errors[0]}"
        )
        return None
    every = dist.tp_allgather(sorted(pk.rows))
    rows = sorted(set.intersection(*(set(r) for r in every)))
    bad = []
    with torch.cuda.device(device):
        for g in rows:
            try:
                what = _dp_pack_check(core, pk, st, g // trip.world, device)
            except Exception as e:  # noqa: BLE001 -- recorded; the launch itself ran or raised early
                what = f"{type(e).__name__}: {e}"
            bad.append((g, what))
    off = {g for rank_bad in dist.tp_allgather(bad) for g, what in rank_bad if what}
    every_mid = dist.tp_allgather(sorted(pk.mid_rows))
    pk.rows = frozenset(g for g in rows if g not in off)
    pk.mid_rows = frozenset(set.intersection(*(set(r) for r in every_mid))) if mode else frozenset()
    for g, what in bad:
        if what:
            logger.warning(
                f"{DP_MOE_PACK_GATHER_ENV}: {g} gathered rows keep the bf16 gather: {what}"
            )
    _dp_ownk_build(core, pk, st, device, dist)
    logger.info(
        f"DP packed MoE gather: rank {core.dp_rank} of {core.dp_size}, gathered rows "
        f"{sorted(pk.rows)}, owner router "
        + (f"mid_rows_mm at own rows {sorted(pk.mid_rows)}" if pk.mid_rows else "the decode router")
        + f", owner top-k at gathered rows {sorted(pk.ownk_rows)}"
    )
    return pk if pk.rows else None


def _dp_ownk_build(core, pk: _DpPack, st, device, dist) -> None:
    """`DP_MOE_OWNK_ENV`, at the end of `_dp_pack_build` (collective; `pk.rows` is the group's):
    the owner top-k gather and each `ids_in` routing-only front are compiled at the agreed row
    counts in the value's range (the front declines its lean rows), the group agrees the builds,
    every row count is checked (`_dp_pack_check(ownk=)`, one collective each) and a row count that
    differs on any rank is dropped on every rank: those keep the packed gather of the logits. Sets
    `pk.ownk` / `pk.ownk_rows`."""
    want = _dp_ownk_wanted(core)
    if want is None:
        return
    trip = core._lamport_trip
    wanted = sorted(g for g in pk.rows if want[0] <= g <= want[1])
    gather, built, error = None, set(), ""
    try:
        with torch.cuda.device(device):
            if wanted:
                gather = _PackGather(trip, core.hidden, core.num_experts, ownk=True)
            for g in wanted:
                tile = int(st.tactics[g][0])
                if _route_front_compile(
                    g,
                    core.hidden,
                    core.hidden,
                    core.local_experts,
                    tile,
                    st.fast2.get(g, 0),
                    True,
                    True,
                    **_route_front_early_kw(st, g),
                ):
                    built.add(g)
    except Exception as e:  # noqa: BLE001 -- every rank decides together below
        error = f"{type(e).__name__}: {e}"
    errors = [e for e in dist.tp_allgather(error) if e]
    if errors:
        logger.warning(
            f"{DP_MOE_OWNK_ENV}: not engaged at rank {core.dp_rank}: {len(errors)} rank(s) failed "
            f"the build: {errors[0]}; the packed gather keeps the logits"
        )
        return
    every = dist.tp_allgather(sorted(built))
    rows = sorted(set.intersection(*(set(r) for r in every)))
    bad = []
    with torch.cuda.device(device):
        for g in rows:
            try:
                what = _dp_pack_check(core, pk, st, g // trip.world, device, ownk=gather)
            except Exception as e:  # noqa: BLE001 -- recorded; the launch itself ran or raised early
                what = f"{type(e).__name__}: {e}"
            bad.append((g, what))
    off = {g for rank_bad in dist.tp_allgather(bad) for g, what in rank_bad if what}
    for g, what in bad:
        if what:
            logger.warning(
                f"{DP_MOE_OWNK_ENV}: {g} gathered rows keep the packed gather of the logits: {what}"
            )
    pk.ownk_rows = frozenset(g for g in rows if g not in off)
    pk.ownk = gather if pk.ownk_rows else None


def _dp_pack_block(core, pk: _DpPack, x, L, rs_next, shared_expert, forked, main, side):
    """`_moe_block` over the packed gather (`DP_MOE_PACK_GATHER_ENV`): the owner's router, the
    packed gather, the shared expert on the side stream, the routing-only front, the pre-routed
    runner (without its finalize at the fused reduce-scatter's finalize rows) and the
    reduce-scatter (the fused one with `rs_next`). Returns `(routed or _RsInjected, joined)` or
    None where this block keeps the bf16 gather. Host values, the same on every rank."""
    st = core._route_front
    rows = x.shape[0]
    g = rows * core.dp_size
    layer = st.layers.get(L.fc1_w.data_ptr())
    # `DP_MOE_OWNK_ENV`: the rows travel routed by their owners
    ownk = g in pk.ownk_rows
    gather = pk.ownk if ownk else pk.gather
    if g not in pk.rows or layer is None or not gather.takes(rows, x):
        return None
    if rows in pk.mid_rows:
        logits = mid_rows_mm_op(x, pk.mids[L.router.data_ptr()])
    else:
        logits = _decode_router(x, L.router)
    xq, xsf, lg = gather(x, logits.contiguous(), L.moe_gscale)
    if not forked:
        shared_expert()
    tile = int(st.tactics[g][0])
    ids, wts = _route_front_only(
        lg,
        st.flats[g][layer],
        tile,
        core.expert_offset,
        core.local_experts,
        core.hidden,
        st.fast2.get(g, 0),
        ownk,
        **_route_front_early_kw(st, g),
    )
    rsi = getattr(core, "_rs_inject", None) if rs_next is not None else None
    fin = (
        rsi is not None
        and rows in rsi.fin_rows
        and rsi.rs.refuses_fin(*rs_next, *forked[0]) is None
    )
    out = _route_front_run(core, st, None, L, layer, None, None, not fin, (xq, xsf, ids, wts))
    partial = _Unfinalized(out[0], wts, st.views[g][layer][0].view(g, core.topk)) if fin else out[0]
    if rsi is not None:
        fused = _rs_inject_tail(core, rsi, rs_next, lambda: forked[0], main, side)(partial)
        if fused is not None:
            return fused, True
    return dep_lamport_round_trip(partial, core._lamport_trip, True), False


#: Attention-DP group sizes the decode forms below (`DP_HC_TILES_ENV`, `DP_INPROJ_RELEASE_ENV`,
#: `DP_GDN_REPLAY_MAP_ENV`, `DP_OUT_PROJ_BF16_ENV`) were measured at: dep4 and dep8 (GB300).
#: dep2 was never run with them; it keeps the stock forms.
_DP_DECODE_SIZES = (4, 8)  # capacity: fixed: the measured layouts, not a buffer size


def _dp_decode_rank(core, device) -> bool:
    """Whether this core is an attention-DP rank (no attention TP) of a measured group size
    (`_DP_DECODE_SIZES`: dep4, dep8) on an SM103 GPU: where the attention-DP decode forms
    (`DP_HC_TILES_ENV`, `DP_INPROJ_RELEASE_ENV`, ...) are measured."""
    return (
        not core.one_rank
        and core.dp_size in _DP_DECODE_SIZES
        and core.attn_tp_size == 1
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == archs.SM_103
    )


def _graph_rows(core, lo: int, hi: int) -> list[int]:
    """This rank's decode row counts (`max_verify_rows` x a CUDA-graph batch size) in lo..hi."""
    p = core.max_verify_rows
    sizes = getattr(core.model_config, "cuda_graph_batch_sizes", None) or ()
    return sorted({p * int(b) for b in sizes if lo <= p * int(b) <= hi})


def _hc_tiles_build(core, device) -> bool:
    """`DP_HC_TILES_ENV`, read once at build (`post_load_weights`): True at an attention-DP rank
    on SM103 with the switch on (default), after compiling every plan `_hc_first` and `_hc_mixer`
    can then launch; False elsewhere, and where a plan does not compile (one warning: the rank
    keeps the 128-row plans, which it compiled before; no other rank needs to agree)."""
    if not (_switch(DP_HC_TILES_ENV, default=True) and _dp_decode_rank(core, device)):
        return False
    try:
        hc_tc_first_precompile(core.stream, core.hc_lowrank, core.hc_count, device, tile64=True)
        core.hc_rowsn_lo = _dp_lowrows_floor(_HC_ROWSN_ROWS[0])
        rowsn = set(range(_HC_ROWSN_ROWS[0], _HC_ROWSN_ROWS[1] + 1))
        rowsn |= set(range(core.hc_rowsn_lo, _DP_LOWROWS_MAX_ROWS + 1))
        bns = sorted({-(-r // 16) * 16 for r in rowsn})
        hc_tc_first_precompile_rowsn(core.stream, core.hc_lowrank, core.hc_count, device, bns)
        variants = [(False, 64), (True, 64), (True, 128)]
        hc_up_precompile(core.hc_count, core.hidden, core.hc_lowrank, device, variants)
    except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
        logger.warning(
            f"{DP_HC_TILES_ENV}: a plan did not compile at rank {core.dp_rank} "
            f"({type(e).__name__}: {e}); the hc mixers keep the 128-row plans"
        )
        return False
    logger.info(
        f"hc mixer plans: rank {core.dp_rank} of {core.dp_size}, 64-row tiles at "
        f"{_HC_TILE64_ROWS[0]}..{_HC_TILE64_ROWS[1]} rows, rows on N at {_HC_ROWSN_ROWS[0]}.."
        f"{_HC_ROWSN_ROWS[1]}"
        + (
            f" and {core.hc_rowsn_lo}..{_DP_LOWROWS_MAX_ROWS}"
            if core.hc_rowsn_lo <= _DP_LOWROWS_MAX_ROWS
            else ""
        )
        + f", two up tiles at {_HC_UP2_ROWS[0]}..{_HC_UP2_ROWS[1]}"
    )
    return True


def _hc_first_x2_build(core, device) -> bool:
    """`DP_HC_FIRST_X2_ENV`, read once at build after `_hc_tiles_build`: True where the switch is
    set and the mixer plans are engaged (the first kernel's two-tile variants compiled here);
    set where they are not: one warning, False."""
    if not _switch(DP_HC_FIRST_X2_ENV):
        return False
    if not core.hc_tiles:
        logger.warning(
            f"{DP_HC_FIRST_X2_ENV} is set but ignored: it rides on the attention-DP mixer plans "
            f"({DP_HC_TILES_ENV}), which serve a dep4 / dep8 rank on SM103 and are not engaged"
        )
        return False
    hc_tc_first_precompile(core.stream, core.hc_lowrank, core.hc_count, device, tile64x2=True)
    lo, hi = _HC_FIRST_X2_ROWS
    logger.info(
        f"hc first kernel on two 64-row tiles ({DP_HC_FIRST_X2_ENV}=1): rank {core.dp_rank} of "
        f"{core.dp_size}, mixers with the injection at {lo}..{hi} rows"
    )
    return True


class _InProj(NamedTuple):
    """`DP_INPROJ_RELEASE_ENV`: one trunk layer's input projection at one row count, as
    `DecodeTarget.forward` hands it to the layer's block (`q_in`)."""

    layer: int
    #: The checkpoint's bf16 `[N, K]` weight (contiguous).
    w: torch.Tensor
    #: The MMA N and the plan the GEMM runs at.
    mpad: int
    plan: MidPlan


def _inproj_plan(n: int, k: int, rows: int) -> tuple | None:
    """`_INPROJ_RELEASE_PLANS`: `(MMA N, plan)` of geometry `(n, k)` at `rows` rows, or None."""
    for bound, rule, plan in _INPROJ_RELEASE_PLANS.get((n, k), ()):
        if rows <= bound:
            if rule == "fine":
                return max(32, -(-rows // 16) * 16), plan
            return next(b for b in (32, 64, 128, 256) if rows <= b), plan
    return None


def _inproj_mm(x: torch.Tensor, ent: _InProj) -> torch.Tensor:
    """`DP_INPROJ_RELEASE_ENV`: `x [rows, K]` bf16 @ `ent.w`^T on `mid_mm_raw` (bf16 operands,
    PDL, the entry's MMA N and plan). Returns a fresh bf16 `[rows, N]`."""
    return mid_mm_raw(x, ent.w, ent.plan, pdl=True, mpad=ent.mpad)


def _inproj_name(core, i: int) -> str:
    return f"l{i}_attn_in_w" if core.is_attn[i] else f"l{i}_in_proj_w"


def _inproj_release_build(core, device) -> dict | None:
    """`DP_INPROJ_RELEASE_ENV`, read once at build after the decode target: `{layer: {rows:
    _InProj}}` at an attention-DP rank on SM103 with PDL and the switch on (default), for every
    trunk layer whose input projection is a bf16 `[N, K]` weight with a plan at the CUDA-graph
    row counts and no other decode copy (`_mid`, packed FP8 / NVFP4: tp1 only today). Every
    entry's GEMM is compiled here and compared with the stock entry on this rank's weight over
    one random activation a row count; a difference at a row count only `DP_LOWROWS_ENV` adds
    drops that row count (one warning); a difference anywhere else, or a GEMM that does not
    compile or run, leaves the switch off at this rank (one warning; the release is rank-local,
    so no other rank needs to agree). None where it is off or cannot take effect."""
    if not (
        _switch(DP_INPROJ_RELEASE_ENV, default=True)
        and _dp_decode_rank(core, device)
        and _DECODE_PROJ_PDL
    ):
        return None
    rows_list = _graph_rows(
        core, _dp_lowrows_floor(_INPROJ_RELEASE_ROWS[0]), _INPROJ_RELEASE_ROWS[1]
    )
    table: dict = {}
    gen = torch.Generator(device=device).manual_seed(20261125)
    checked = {}
    for i in range(core.num_layers):
        name = _inproj_name(core, i)
        w = core.w[name] if name in core.w else None
        if (
            w is None
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or not w.is_contiguous()
            or w.data_ptr() % 16
            or name in core._mid
            or name in core._gemv8
            or name in core._gemv_nv
        ):
            continue
        geom = (int(w.shape[0]), int(w.shape[1]))
        ents = {}
        for rows in rows_list:
            plan = _inproj_plan(*geom, rows)
            if plan is not None:
                ents[rows] = _InProj(i, w, plan[0], plan[1])
        if not ents:
            continue
        if geom not in checked:
            # The stock entry the block runs without the switch (`_dense_mm`'s bf16 branch).
            stock = core.decode_mm if core.is_attn[i] else core.decode_in_mm
            try:
                x = torch.randn(max(ents), geom[1], generator=gen, device=device)
                x = x.to(torch.bfloat16)
                low_bad, n_bad = [], 0
                for rows, ent in ents.items():
                    got = _inproj_mm(x[:rows], ent)
                    ref = stock(x[:rows], w.t())
                    n = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
                    if n and rows < _INPROJ_RELEASE_ROWS[0]:
                        # A row count only `DP_LOWROWS_ENV` adds: it alone keeps the stock forms,
                        # so that opt-in never turns the default rows off.
                        low_bad.append(rows)
                    else:
                        n_bad += n
                bad = f"differs from the stock entry in {n_bad} elements" if n_bad else ""
            except Exception as e:  # noqa: BLE001 -- declined like a GEMM that differs
                bad, low_bad = f"raised {type(e).__name__}: {e}", []
            checked[geom] = (bad, low_bad)
            if low_bad:
                logger.warning(
                    f"{DP_LOWROWS_ENV}: the bf16 mid-rows GEMM of a {geom[0]} x {geom[1]} input "
                    f"projection differs from the stock entry at rows {low_bad} on rank "
                    f"{core.dp_rank}; those row counts keep the stock forms"
                )
        if checked[geom][0]:
            logger.warning(
                f"{DP_INPROJ_RELEASE_ENV}: the bf16 mid-rows GEMM of a {geom[0]} x {geom[1]} "
                f"input projection {checked[geom][0]} at rank {core.dp_rank}; every site and input "
                "projection keeps the stock forms"
            )
            return None
        for rows in checked[geom][1]:
            ents.pop(rows)
        if ents:
            table[i] = ents
    if not table:
        return None
    try:
        torch.cuda.synchronize(device)
        hc_up_precompile(core.hc_count, core.hidden, core.hc_lowrank, device, [(True, 128)])
    except Exception as e:  # noqa: BLE001 -- a default-on form declines rather than stop the load
        logger.warning(
            f"{DP_INPROJ_RELEASE_ENV}: the releasing up kernel did not compile at rank "
            f"{core.dp_rank} ({type(e).__name__}: {e}); every site and input projection keeps "
            "the stock forms"
        )
        return None
    logger.info(
        f"input projections behind a releasing hc site: rank {core.dp_rank} of {core.dp_size}, "
        f"{len(table)} layers, rows {sorted({r for t in table.values() for r in t})}"
    )
    return table


def _inproj_release_at(core, i: int, rows: int) -> _InProj | None:
    """`DP_INPROJ_RELEASE_ENV`: trunk layer `i`'s input projection at `rows` rows, or None where
    its site A and projection stay the stock forms. Host side, a per-capture constant."""
    table = core._inproj_release
    return None if table is None else table.get(i, {}).get(rows)


def _decode_forms(core) -> str:
    """The line a multi-rank core logs once its decode forms are built (`post_load_weights`):
    what each `TRTLLM_QWEN4_EXP_*` decode form of its layout engaged at on this rank, row counts
    where they decide. A form set where it cannot take effect also warns on its own; the GDN
    replay map, settled at each cache-manager bind, logs there."""

    def rows(table) -> list:
        return sorted(table) if table else []

    route = core._route_front
    head = f"decode forms: rank {core.mapping.tp_rank} of {core.mapping.tp_size}"
    if core.dp_size > 1:
        router, inproj, out = core._router_front, core._inproj_release, core._out_proj
        rsi = getattr(core, "_rs_inject", None)
        forms = {
            "moe comm": _dp_moe_comm() if core._lamport_trip is not None else "nccl",
            "route front gathered rows": rows(route.tactics if route else None),
            "router front gathered rows": rows(router.rows if router else None),
            "releasing gather rows": rows(router.release_rows if router else None),
            "hc tiles": core.hc_tiles,
            "in-proj release rows": rows({r for t in (inproj or {}).values() for r in t}),
            "gdn conv lean": getattr(core, "_gdn_lean_want", False),
            "gdn inproj conv": _switch(DP_GDN_INPROJ_CONV_BF16_ENV, default=True)
            and getattr(core, "_gdn_lean_want", False),
            "gdn replay norm": _gdn_replay_norm_range() is not None
            and getattr(core, "_gdn_lean_want", False),
            "out-proj gdn rows": rows(out["gdn_out"] if out else None),
            "out-proj attn rows": rows(out["attn_o"] if out else None),
            "out-proj inject rows": rows(getattr(getattr(core, "_siteb", None), "gemm", None)),
            "siteb plan32 rows": list(getattr(getattr(core, "_siteb", None), "plan32", ())),
            "grouped gqa from requests": core.dp_gqa_min_requests,
            "cluster top-k up to rows": core.dp_topk_cluster_rows,
            "bitmap gqa from requests": getattr(core, "dp_gqa_bitmap", 0),
            "draft gqa grouped1": getattr(core, "dp_draft_gqa_g1", False),
            "mtp attn pack": getattr(core, "dp_mtp_attn_pack", False),
            "hc first x2": getattr(core, "hc_first_x2", False),
            "route fast2 rows": rows(getattr(route, "fast2", None) if route else None),
            "route fast2 preload rows": [
                r for r, f in sorted(getattr(route, "fast2", {}).items()) if f == 2
            ]
            if route
            else [],
            "route front early rows": rows(getattr(route, "early", None) if route else None),
            "draft moe tile": getattr(core, "_draft_moe_tactics", None) is not None,
            "launch order": getattr(core, "dp_launch_order", False),
            "index key fp8": getattr(core, "index_key_fp8", False),
            "rs inject rows": rows(getattr(rsi, "rows", None)),
            "fin rs rows": rows(getattr(rsi, "fin_rows", None)),
            "fin rs pdl": bool(rsi is not None and rsi.rs.pdl),
            "draftpack": sorted(getattr(core, "dp_draftpack", ())),
            "pack gather rows": rows(getattr(getattr(core, "_dp_pack", None), "rows", None)),
            "pack ownk rows": rows(getattr(getattr(core, "_dp_pack", None), "ownk_rows", None)),
            "scores early": getattr(core, "dp_scores_early", False),
            "steady prepare": getattr(core, "host_step_forms", {}).get("dp_steady", False),
            "qsa steady": getattr(core, "host_step_forms", {}).get("qsa_steady", False),
            "replay gap": getattr(core, "host_step_forms", {}).get("qsa_replay_gap", False),
            "topk cowrite": getattr(core, "topk_cowrite", False),
        }
    else:
        forms = {
            "tep fast": core.tep_fast,
            "route front rows": rows(route.tactics if route else None),
            "decode front rows": rows(route.decode if route else None),
            "finar": core._tep_finar is not None,
            "draft layer": core.tep_draft_layer,
            "proj split-k": getattr(core, "tep_proj", False),
            "finar shared": getattr(core._tep_finar, "shared", None) is not None,
            "index prefetch": getattr(core, "tep_index_prefetch", False),
            "gqa e4m3": getattr(core, "tep_gqa_e4m3", False),
            "attn inplace": getattr(core, "tep_attn_inplace", False),
            "index key fp8": getattr(core, "index_key_fp8", False),
            "qsa steady": getattr(core, "host_step_forms", {}).get("qsa_steady", False),
            "topk cowrite": getattr(core, "topk_cowrite", False),
            "tp sum lamport": getattr(core, "_tep_tpsum", None) is not None,
            "index split": getattr(core, "tep_index_split", 0),
            "draft prologue": getattr(core, "tep_draft_prologue", False),
        }
    return f"{head}: " + ", ".join(f"{k} {v}" for k, v in forms.items())


def _moe_prefill_copy(one_rank: bool, trunk_moe_backend: str, max_num_tokens: int) -> bool:
    """Whether the trunk keeps the CuteDSL-layout expert copy: at one rank only (every expert
    is local, the layout the CuteDSL MoE is certified at), only under trtllm-gen, only when a
    step's rows (at most `max_num_tokens`) can reach the dispatch threshold, and not when the
    env var turns it off."""
    wanted = _switch(_MOE_PREFILL_COPY_ENV, default=True)
    return (
        one_rank
        and trunk_moe_backend == "TRTLLM"
        and max_num_tokens >= _CUTE_DSL_MOE_MIN_ROWS
        and wanted
    )


def _ple_window(core):
    """The n-gram gather's `row_window` for this rank's table (None: the whole table)."""
    host = core._ple_table_host
    return None if host is None else host.window


def _ple_needs_sum(core) -> bool:
    """Whether the gathered n-gram rows are this rank's share, summed over attention TP."""
    host = core._ple_table_host
    return host is not None and host.needs_sum


def _ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is declared in host memory (the default)."""
    return _switch(_PLE_HOST_OFFLOAD_ENV, default=True)


def _moe_layer_scalars(
    w, i: int, start: int, count: int, prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Layer `i`'s routed-expert scalars for the trtllm-gen runner over this rank's window
    `[start, start + count)`, built from the checkpoint's `input_scale` / `weight_scale_2`
    rows (declared over the whole routing space on every rank).

    The checkpoint stores modelopt reciprocals (`input_scale = 1 / g_act`, `weight_scale_2 =
    1 / g_w`), so a quantizer's global scale is `1 / input_scale` and the runner's scalars
    are products of the two."""
    parameter_prefix = f"l{i}_{prefix}"
    isc1, isc1_up = w[f"{parameter_prefix}e_isc1"], w[f"{parameter_prefix}e_isc1_up"]
    assert torch.equal(isc1, isc1_up), (
        f"layer {i}: expert gate/up input_scale differ; the fused FC1 reads one activation "
        "quantization for both halves"
    )
    assert torch.equal(w[f"{parameter_prefix}e_ws2_1"], w[f"{parameter_prefix}e_ws2_1_up"]), (
        f"layer {i}: expert gate/up weight_scale_2 differ; the fused FC1 carries one alpha "
        "per expert for both halves"
    )
    # One activation quantization feeds all experts, on every rank.
    assert torch.equal(isc1.min(), isc1.max()), (
        f"layer {i}: the routed experts do not share one gate_proj.input_scale, so a single "
        "activation quantization cannot feed them all"
    )
    e_isc2, ws2_2 = w[f"{parameter_prefix}e_isc2"], w[f"{parameter_prefix}e_ws2_2"]
    gate1 = isc1 * w[f"{parameter_prefix}e_ws2_1"]
    window = slice(start, start + count)
    return dict(
        # The routed FC1 activation global scale (one per layer).
        global_scale=(1.0 / isc1[:1]).contiguous(),
        # [count] each. Swapping the first two is finite and silent.
        output1_scale_scalar=(gate1 / e_isc2)[window].contiguous(),
        output1_scale_gate_scalar=gate1[window].contiguous(),
        output2_scale_scalar=(e_isc2 * ws2_2)[window].contiguous(),
    )


def _moe_layer_scalars_cd(w, i: int) -> dict[str, torch.Tensor]:
    """Layer `i`'s routed-expert scalars for the CuteDSL MoE over the second expert copy (one
    rank, every expert; `_moe_layer_scalars` checks the checkpoint facts they rest on). FC1's
    per-expert alpha is `input_scale * weight_scale_2`, its NVFP4 input the same activation
    quantization as the trtllm-gen runner's; FC2's input is quantized at one global scale,
    `1 / max(input_scale)` over the layer's experts, and its alpha carries that maximum."""
    isc2_max = w[f"l{i}_e_isc2"].max()
    return dict(
        alpha1=(w[f"l{i}_e_isc1"] * w[f"l{i}_e_ws2_1"]).contiguous(),
        alpha2=(isc2_max * w[f"l{i}_e_ws2_2"]).contiguous(),
        global_sf=(1.0 / isc2_max).reshape(1).contiguous(),
    )


class _Replay(NamedTuple):
    """The GDN cached replay's history buffers and bookkeeping of one manager: two history
    halves a slot (`old`, the Triton / CuTe replay), or, under FlashInfer's ring replay
    (`TRTLLM_USE_GDN_FLASHINFER_REPLAY`), a 32-row ring a slot (`ring`, `cache_base`)."""

    old: dict  # GDN layer -> (old_x, old_B, old_dt, old_dA_cumsum); empty under the ring
    cache_buf_idx: torch.Tensor  # [slots] int32: the history half each slot writes next
    prev_num_accepted: torch.Tensor  # [slots] int32: tokens pending since the checkpoint
    history: int  # rows a history half holds
    width: int  # the one verify width a step may carry per generation request
    ring: dict | None = None  # GDN layer -> (k_ring, u_ring, g_ring), FlashInfer's layout
    cache_base: torch.Tensor | None = None  # [slots] int32: each slot's ring start


def _check_replay_ring(g, layer: int, slots: int, k_ring, u_ring, g_ring) -> None:
    """One GDN layer's FlashInfer ring buffers at this rank's heads: `[slots, H, 32, K]`,
    `[slots, HV, 32, V]` in the state's bf16 and `[slots, HV, 32]` fp32 cumulative gates."""
    ring = FLASHINFER_REPLAY_RING_SIZE
    shapes = (
        (k_ring, (slots, g.num_k_heads, ring, g.head_k_dim), torch.bfloat16),
        (u_ring, (slots, g.num_v_heads, ring, g.head_v_dim), torch.bfloat16),
        (g_ring, (slots, g.num_v_heads, ring), torch.float32),
    )
    for t, shape, dtype in shapes:
        assert t is not None and tuple(t.shape) == shape and t.dtype == dtype, (
            f"layer {layer}: a FlashInfer replay ring of {None if t is None else tuple(t.shape)} "
            f"{None if t is None else t.dtype}; this rank's heads need {shape} {dtype}"
        )


class _Bound(NamedTuple):
    """What the core reads off one KV cache manager: the manager, this rank's pool views
    (`pool.RankPools`) and, under MTP, the verify step's runtime-owned buffers: the per-step
    conv windows (`int_conv`, by GDN layer), the rows `update_mamba_states` promotes them
    from (`int_idx`) and the cached replay. Valid only for that manager."""

    cm: object
    pools: _pool.RankPools
    int_idx: torch.Tensor | None = None
    int_conv: dict | None = None
    replay: _Replay | None = None


class _DraftBound(NamedTuple):
    """The draft layer's planes in the manager its draft step is handed: its QSA index plane,
    that manager's position plane, its paged K/V and, under IndexShare, its row of the
    metadata's shared selection buffers."""

    cm: object
    index_cache: torch.Tensor
    position_cache: torch.Tensor
    kv: _pool.KvPlanes
    share_slot: int | None


class Qwen4ExpDefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` is passed in because the text sub-config adopted as
        `model_config.pretrained_config` carries neither the vision half nor the media
        token ids."""
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        # One rank holds the whole model (tp1), which takes forms of its own (module docstring).
        self.one_rank = mapping.world_size == 1
        assert archs.served(torch.cuda.get_device_capability()), (
            f"target serves {archs.served_names()}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        assert getattr(cfg, "model_type", None) == "qwen4_exp_text", (
            f"the core must be built on the normalized text config; got model_type "
            f"{getattr(cfg, 'model_type', None)!r}"
        )
        self.mapping = mapping
        # The ranks attention heads and the recurrent heads are split over: the TP group,
        # or one rank under attention DP.
        self.attn_tp_size = _pool.attention_tp_size(mapping)
        # Under attention DP each rank serves its own requests and the MoE blocks run over the
        # rows the whole group holds (`_expert_sum_op`). The group is the TP group, which spans
        # the DP ranks; its size is the world's, never a constant.
        self.dp_size = mapping.tp_size if mapping.enable_attention_dp else 1
        self.dp_rank = mapping.tp_rank
        self.dp_group = list(mapping.tp_group)
        # One-model MTP-Eagle (routing admits no other speculation): the checkpoint's one MTP
        # module, built only when the engine asked for speculation.
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None
        # The draft layer's FP8 block-scale experts are whole-expert windows over the TP group
        # at every layout, tp4 included (`weights.rank_dims`, `mtp_local_experts`).
        # Rows per generation request of the widest verify step (`max_draft_len + 1`); 1
        # without MTP.
        self.max_verify_rows = spec_config.max_draft_len + 1 if self.mtp_enabled else 1
        # The most rows one forward carries (a draft step's included): sizes the draft
        # layer's unit injection weight (`_MtpWeights.unit_inject`).
        self.max_num_tokens = model_config.max_num_tokens

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps
        layer_types = list(cfg.layer_types)
        self.is_attn = [t == "full_attention" for t in layer_types]
        self.attn_layers = [i for i, a in enumerate(self.is_attn) if a]
        self.gdn_layers = [i for i, a in enumerate(self.is_attn) if not a]

        # ---- hyper-connections ------------------------------------------
        self.hc_count = cfg.hc_count
        self.hc_lowrank = cfg.hc_lowrank
        self.stream = self.hc_count * self.hidden
        # Columns of the padded first-projection weight (`_fuse_hc_first_projections`) the
        # first projection multiplies: the real `[down; block_inject]` rows rounded up to 32.
        self.hc_first_cols = -(-(self.hc_lowrank + self.hc_count) // 32) * 32

        # ---- RoPE (partial, neox pairs, interleaved mRoPE) -----------------
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

        # ---- QSA sparse index (replicated on every rank) -------------------
        self.index_heads = cfg.indexer_n_heads
        self.index_kv_heads = cfg.indexer_kv_heads
        self.index_head_dim = cfg.indexer_head_dim
        self.token_topk = cfg.indexer_budget
        self.compress_ratio = cfg.indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        # Dense-attention cutoff: QSA resolves max(token_topk, seq_len_threshold); checked
        # against the engine's value because the CUDA-graph runner splits its short and long
        # decode families on it.
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
        # IndexShare for the MTP draft loop (`index_share_for_mtp_iteration`), as the engine
        # resolved it: the same lowering sizes the metadata's shared selection buffers.
        self.mtp_index_share = bool(qsa_params.mtp_index_share)
        self.index_width = (self.index_heads + self.index_kv_heads) * self.index_head_dim
        # The index query's `[q | k | v]` scratch for its norm + rope; the spare head is the
        # v slot the op needs but never reads.
        self.index_scratch_width = self.index_width + self.index_head_dim
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # ---- the MTP module ------------------------------------------------
        # Global layer index of the draft layer (the KV managers map it there, and the
        # weight table keys it there); its rope base is its own block's (a mapping or a config
        # object), else the trunk's. Routing pins the block's other settings.
        self.mtp_layer_idx = self.num_layers
        mtp_block = getattr(cfg, "mtp", None)
        mtp_theta = (
            mtp_block.get("rope_theta")
            if isinstance(mtp_block, dict)
            else getattr(mtp_block, "rope_theta", None)
        )
        self.mtp_theta = self.theta if mtp_theta is None else float(mtp_theta)

        # ---- gated deltanet ----------------------------------------------
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim

        # ---- MoE ----------------------------------------------------------
        self.num_experts = cfg.num_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.shared_inter = cfg.shared_expert_intermediate_size
        requested = model_config.moe_backend_requested.upper()
        assert requested in _TRUNK_MOE_BACKENDS, (
            f"moe_config.backend={model_config.moe_backend_requested!r} is not implemented by "
            f"this target; the supported requests are {sorted(_TRUNK_MOE_BACKENDS)}"
        )
        # Fixes the load-time layout of the expert stacks (`weights.py`).
        self.trunk_moe_backend = _TRUNK_MOE_BACKENDS[requested]
        # The CuteDSL pipeline runs every expert at once: one rank only.
        assert self.trunk_moe_backend != "CUTEDSL" or self.one_rank, (
            "moe_config.backend=CUTEDSL is implemented at one rank (tp1) only; this layout "
            f"has {mapping.world_size} ranks"
        )
        assert self.trunk_moe_backend != "CUTEDSL" or (
            self.max_verify_rows - 1 <= _CUTEDSL_MAX_DRAFT_LEN
        ), (
            f"max_draft_len={self.max_verify_rows - 1} with moe_config.backend=CUTEDSL: the "
            f"CuteDSL expert pipeline is certified at max_draft_len up to {_CUTEDSL_MAX_DRAFT_LEN}"
        )
        # Without the CuTe DSL MoE every launch runs on trtllm-gen.
        cute_dsl_moe = archs.fast_path("cute_dsl_moe", torch.cuda.get_device_capability())
        assert self.trunk_moe_backend != "CUTEDSL" or cute_dsl_moe, (
            "moe_config.backend=CUTEDSL: the CuTe DSL MoE does not run on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}; "
            "use moe_config.backend=TRTLLM"
        )
        # At one rank the CuteDSL-layout copy of the experts for prefill-sized launches
        # (`_moe_prefill_copy`, whose switch is read on every GPU); `weights.py` declares it.
        prefill_copy = _moe_prefill_copy(
            self.one_rank, self.trunk_moe_backend, model_config.max_num_tokens
        )
        self.moe_prefill_copy = cute_dsl_moe and prefill_copy
        # Cached native NVFP4 operands or the one-rank FP8 conversion, built once because
        # the shell's post-load walk reaches `post_load_weights` again.
        self._mtp_moe_nvfp4 = None
        # The FP8 draft head at tp1, set by the shell's `post_load_weights` after this core's;
        # the post-load walk calls this core's `post_load_weights` again, which leaves it.
        self.draft_head = None

        # ---- PLE -----------------------------------------------------------
        # One-based in the config.
        self.ple_layer = list(cfg.ple_layer_ids)[0] - 1
        self.ngram_size = cfg.ngram_size
        self.ple_dilation = self.ngram_size
        self.ngram_vocab_divisor = cfg.make_ngram_vocab_size_divisible_by
        eos = cfg.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.ple_host_offload = _ple_host_offload_enabled()
        # How the host-resident table is stored (`ple_host_table`).
        self.ple_host_mode = _ple_host.host_table_mode()
        if self.ple_host_mode != _ple_host.RANK and not self.ple_host_offload:
            logger.warning(
                f"{_ple_host.PLE_HOST_TABLE_ENV}={self.ple_host_mode} applies to the host-resident "
                f"n-gram table; {_PLE_HOST_OFFLOAD_ENV}=0 keeps one device copy per rank"
            )
        _check_switches(self.one_rank)
        # CPU allocation behind the offloaded table; the `ple_table` parameter is a
        # device-address view that owns none of those pages.
        self._ple_table_host = None

        # ---- the vision half (the weight table declares the tower) ---------
        vcfg = getattr(composite_config, "vision_config", None)
        assert vcfg is not None, "the composite config carries no vision_config"
        self.vision_config = vcfg
        self.image_token_id = int(composite_config.image_token_id)
        self.video_token_id = int(composite_config.video_token_id)

        # ---- KV pool storage ------------------------------------------------
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

        # ---- this rank's share ----------------------------------------------
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.heads_q = d.heads_q
        self.heads_kv = d.heads_kv
        self.kv_head_start = d.kv_head_start
        self.q_width = d.q_width
        self.kv_width = d.kv_width
        self.qkv_width = self.q_width + 2 * self.kv_width
        # The fused input projection's output row, `[q | k | v | gate | index q, k | v slot]`
        # (`weights._attn_in_w`). Its leading columns are the packed QKV the main norm + rope
        # rewrites in place; the rope addresses a row as `(Hq + Hk + Hv) * head_dim` and
        # never touches the v heads, so everything after k is "v" to it.
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
        self.local_experts = d.local_experts
        self.expert_offset = d.expert_offset
        # The runner's `intermediate_size`: each held expert's intermediate as stored, this
        # rank's slice zero-padded at tp4 (`weights.EXPERT_INTER_ALIGN`), whole otherwise.
        self.expert_inter = d.expert_inter
        # The draft layer's FP8 experts: this rank's window over the whole TP group.
        self.mtp_local_experts = d.mtp_local_experts
        self.mtp_expert_offset = d.mtp_expert_offset
        # A tep4 rank's decode forms (`TEP_FAST_DECODE_ENV`), settled with the layout, and its
        # draft layer's (`TEP_DRAFT_LAYER_ENV`).
        self.tep_fast = _tep_fast_decode(self)
        self.tep_draft_layer = _tep_draft_layer(self)
        # An attention-DP rank's draft attention input in the one-rank form (`DP_MTP_ATTN_PACK_ENV`).
        self.dp_mtp_attn_pack = _dp_mtp_attn_pack(self)
        # The attention-DP decode forms, built after load (`post_load_weights`) on a GPU:
        # `DP_HC_TILES_ENV`, `DP_INPROJ_RELEASE_ENV`.
        self.hc_tiles = False
        # `DP_HC_FIRST_X2_ENV` (opt-in), built after load on top of `hc_tiles`.
        self.hc_first_x2 = False
        self._inproj_release = None
        # `DP_GDN_REPLAY_MAP_ENV`, settled at each cache-manager bind.
        self.gdn_replay_map = False
        # `DP_GDN_CONV_LEAN_ENV`: the wish (built after load), what each bind settled, and the
        # step in flight's decision.
        self._gdn_lean_want = self._gdn_lean_on = False
        self._gdn_lean_state_ref = None
        self._gdn_xsave, self._gdn_xsave_all = {}, None
        self._gdn_step = _GdnStep()
        # `DP_GDN_INPROJ_CONV_BF16_ENV`: `{rows: _InProj}` of each bind;
        # `DP_GDN_REPLAY_NORM_BF16_ENV`: the request counts of each bind.
        self._gdn_inconv = {}
        self._gdn_rnorm = frozenset()
        self._gdn_wi_width = 4
        # `DP_OUT_PROJ_BF16_ENV` (opt-in), built after load.
        self._out_proj = None
        # `DP_DRAFT_MOE_TILE_ENV`: the draft MoE's tactic by gathered rows, built after load.
        self._draft_moe_tactics = None
        # `DP_LAUNCH_ORDER_ENV`, settled after load.
        self.dp_launch_order = False
        # `DP_DRAFT_O_MID_ENV` (opt-in): the draft output projection's rows, built after load.
        self._draft_o = None
        # `DP_OUT_PROJ_INJECT_ENV` (opt-in): the site-B fold, built after load, and the operands
        # the trunk loop hands the next output projection (`_siteb_arm`).
        self._siteb = None
        self._siteb_pending = None
        # The requests from which the grouped sparse GQA serves a step (`DP_GQA_MIN_REQUESTS_ENV`).
        self.dp_gqa_min_requests = _dp_gqa_min_requests(self)
        # The trunk verify step's bitmap-union sparse GQA floor, 0 = off (`DP_GQA_BITMAP_ENV`).
        self.dp_gqa_bitmap = _dp_gqa_bitmap(self)
        # The draft layer's one-row passes on the same kernel (`DP_DRAFT_GQA_GROUPED1_ENV`).
        self.dp_draft_gqa_g1 = _dp_draft_gqa_g1(self)
        # e4m3 index anchors (`INDEX_KEY_FP8_ENV`, lossy, opt-in).
        self.index_key_fp8 = _switch(INDEX_KEY_FP8_ENV)
        if self.index_key_fp8:
            logger.warning(
                f"{INDEX_KEY_FP8_ENV}=1: the QSA index anchors are e4m3 (lossy against bf16 "
                "keys); every context and generation worker of this deployment must set it"
            )
        # The cold index codec's group marks (`CTX_COLDIDX_ENV`): only over e4m3 anchors.
        self.cold_index_marks = self.index_key_fp8 and _switch(CTX_COLDIDX_ENV)
        # The most generation rows the cluster top-k serves (`DP_TOPK_CLUSTER_ROWS_ENV`).
        self.dp_topk_cluster_rows = _dp_topk_cluster_rows(self)
        # The trunk verify's early index scores (`DP_SCORES_EARLY_ENV`, opt-in), settled after load.
        self.dp_scores_early = False
        # The generation cluster top-k's coalesced write (`DP_TOPK_NARROW_ENV` /
        # `TEP_TOPK_NARROW_ENV`, opt-in), settled after load.
        self.topk_cowrite = False

        # Every weight but the n-gram table; meta-init intercepts torch.empty.
        self.w = _weights.MODEL_WEIGHTS.declare(d)

        # Derived once after load (`post_load_weights`).
        self._moe_scalars: list | None = None
        self._gdn_gating: dict | None = None
        self._cos_sin: torch.Tensor | None = None
        self._hc_first: dict | None = None
        # The attention op's KV scales: fp32 [1] ones over an fp8 pool, fp32 [3] ones (q, K,
        # V) over an NVFP4 one, None over bf16.
        self.kv_scale_orig_quant: torch.Tensor | None = None
        self.kv_scale_quant_orig: torch.Tensor | None = None
        # The shared expert's stream (see `_moe_block`); made after load, because a stream
        # created inside a graph capture is not capturable.
        self._side_stream: torch.cuda.Stream | None = None
        # The tensor-parallel sum of a block's partial output (`_tp_sum_op`, over the engine's
        # `allreduce_strategy`), and the routed experts' completion over this rank's rows
        # (`_expert_sum_op`).
        self.allreduce_strategy = int(model_config.allreduce_strategy)
        self.tp_sum = None
        self.expert_sum = None
        # The attention-DP Lamport workspace (`DP_MOE_COMM_ENV`), built once a load, and with
        # `lamport_step` the registry of its launch indices, which the shell's forward brackets.
        self._lamport_trip = None
        self._lamport_steps: LamportSteps | None = None
        # The router front (`DP_ROUTER_FRONT_ENV`), built after the route front.
        self._router_front: "_DpRouterFront | None" = None
        # The fused reduce-scatter (`DP_RS_INJECT_ENV`), built once a load after the router front,
        # and the draft MoE's parts that ride on it (`DP_DRAFTPACK_ENV`).
        self._rs_inject: "_DpRsInject | None" = None
        self.dp_draftpack: frozenset = frozenset()
        # The packed MoE gather (`DP_MOE_PACK_GATHER_ENV`, opt-in), built once a load after it.
        self._dp_pack: "_DpPack | None" = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
        # The decode MoE's route front (`DP_ROUTE_FRONT_ENV`), built with the targets.
        self._route_front: _RouteFront | None = None
        # The tep4 MoE finalize + all-reduce launch (`TEP_FINAR_ENV`), built once after it.
        self._tep_finar: _TepFinar | None = None
        # Everything derived from the KV cache manager (`_bind_cache_manager`).
        self._bound: _Bound | None = None
        self._binds = 0
        # Set by `begin_step`, cleared by `forward`.
        self._step_begun = False
        # MTP module operands (`_mtp_views`) and its index rope table; None with speculation
        # off.
        self._mtp: "_MtpWeights | None" = None
        self._mtp_cos_sin: torch.Tensor | None = None
        # The verify step's side-cache rollback, registered with the spec worker.
        self.spec_rollback: SpeculativeRollback | None = (
            SpeculativeRollback() if self.mtp_enabled else None
        )
        # The draft layer's own bind, keyed on the manager its draft step sees.
        self._draft_bound: _DraftBound | None = None

    # ---------------------------------------------------------------- load ---

    def _fuse_hc_first_projections(self) -> None:
        """Concatenate each hyper-connection's `down` and `block_inject` weights into one
        `[rows, stream]` tensor, zero-padded to a multiple of 128 rows, so a site's first
        projection is one GEMM: the trunk's sites, keyed `(layer, tag)` and `"fhc"`, and with
        MTP the draft layer's, keyed `"mtp_ahc"`, `"mtp_mhc"` and `"mtp_fhc"`. The
        checkpoint's parameters are rebound as row views of the padded tensors."""
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

    def post_load_weights(self) -> None:
        """Derive what the targets read, then build them, now that the weights are real.

        Not an override (DecoderModel has no such hook); named for the shell hook."""
        w = self.w
        device = w["embed"].device
        self._moe_scalars = [
            _moe_layer_scalars(w, i, self.expert_offset, self.local_experts)
            for i in range(self.num_layers)
        ]
        # The CuteDSL scalars: of the second expert copy (`moe_prefill_copy`), or of the
        # experts themselves on the CuteDSL backend.
        self._moe_scalars_cd = (
            [_moe_layer_scalars_cd(w, i) for i in range(self.num_layers)]
            if self.moe_prefill_copy or self.trunk_moe_backend == "CUTEDSL"
            else None
        )
        # The delta-rule entries index A_log and dt_bias as `ptr + head` and certify fp32;
        # the checkpoint is bf16.
        self._gdn_gating = {
            i: (
                w[f"l{i}_A_log"].float().contiguous(),
                w[f"l{i}_dt_bias"].float().contiguous(),
            )
            for i in self.gdn_layers
        }
        # Reconcile the table height against the checkpoint's id-space tensors: an index past
        # the end is a device-side assert that poisons the process.
        # Rows of the whole table: under `server` storage `ple_table` holds only this rank's.
        host = self._ple_table_host
        table_rows = host.rows if host is not None else w["ple_table"].shape[0]
        used_rows = int(w["ple_offs"][-1]) + int(w["ple_vocab"][-1])
        assert used_rows <= table_rows, (
            f"the n-gram table has {table_rows} rows but its per-head offsets and vocabulary "
            f"sizes address {used_rows}"
        )
        assert table_rows % self.ngram_vocab_divisor == 0, (table_rows, self.ngram_vocab_divisor)
        assert table_rows - used_rows < self.ngram_vocab_divisor, (
            f"the n-gram table carries {table_rows - used_rows} padding rows, more than one "
            f"divisor of {self.ngram_vocab_divisor}: its shards do not concatenate to the padded "
            "id space the checkpoint declares, and every id past the first shard would read a "
            "shifted row"
        )
        self._cos_sin = self._index_rope_table(self.theta, device)
        # Before the draft layer's views, which read its fused first projections.
        self._fuse_hc_first_projections()
        if self.mtp_enabled:
            # The draft layer's index plane rotates at the module's own rope base, like its
            # attention rope.
            self._mtp_cos_sin = (
                self._cos_sin
                if self.mtp_theta == self.theta
                else self._index_rope_table(self.mtp_theta, device)
            )
            self._mtp = _mtp_views(self)
            if w[f"l{self.mtp_layer_idx}_mtp_fc1_w"].dtype == torch.uint8 or (
                self.one_rank and device.type == "cuda" and _switch(_MTP_MOE_NVFP4_ENV)
            ):
                if self._mtp_moe_nvfp4 is None:
                    self._mtp_moe_nvfp4 = _mtp_moe_nvfp4(self)
                self._mtp = self._mtp._replace(**self._mtp_moe_nvfp4)
            self._build_draft_one_rank(device)
        if self.quant_mode in (QM_FP8_KV_CACHE, QM_NVFP4_KV_CACHE):
            # Built here, not at construction: that runs under meta init.
            scales = 3 if self.quant_mode == QM_NVFP4_KV_CACHE else 1
            self.kv_scale_quant_orig = torch.ones(scales, dtype=torch.float32, device=device)
            self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        self._build_dense_copies(device)
        # The CPU-only load checks build the targets over host weights; they launch nothing.
        if device.type == "cuda":
            # Every first-half variant `_hc_mixer` launches up to `_HC_TC_FIRST_MAX_ROWS` rows,
            # compiled here so that no serving step (or graph capture) meets a compile.
            hc_tc_first_precompile(self.stream, self.hc_lowrank, self.hc_count, device)
            self._report_ple_residency()
            self._side_stream = torch.cuda.Stream(device=device)
            # A tep4 rank's PLE front (`DecodeTarget._ple_fork`); its cuBLAS workspace is made
            # here, outside any capture.
            self._ple_stream = None
            if self.tep_fast:
                self._ple_stream = torch.cuda.Stream(device=device)
                self._ple_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._ple_stream):
                    probe = torch.zeros([1, 64], dtype=torch.bfloat16, device=device)
                    cublas_mm(probe, probe.new_zeros([64, 64]).t())
                torch.cuda.current_stream().wait_stream(self._ple_stream)
            # The prefill CuteDSL MoE's output memset, beside its FC1 (`fp4_moe_renorm_cute_dsl`):
            # its own stream, so the join before FC2 does not also wait for the shared expert.
            self._moe_memset_stream = torch.cuda.Stream(device=device)
            self.tp_sum = self._tp_sum_op()
            self.expert_sum = self._expert_sum_op()
        # A decode step's bf16 projection entry, and its GDN in_proj's: at tp1 the one-rank
        # dispatch (`_one_rank_projection`), at the other layouts the shapes' own.
        one_rank = self.one_rank
        # A tep4 rank's projections past `low_m_bf16_mm`'s table (`TEP_PROJ_ENV`).
        self.tep_proj = _tep_proj(self)
        # A tep4 rank's verify-step index keys before the PDL wait (`TEP_INDEX_PREFETCH_ENV`).
        self.tep_index_prefetch = _tep_index_prefetch(self)
        # A tep4 rank's visible-split index launch shape (`TEP_INDEX_SPLIT_ENV`).
        self.tep_index_split = _tep_index_split(self)
        # A tep4 rank's decode GQA on the e4m3 entry (`TEP_GQA_E4M3_ENV`).
        self.tep_gqa_e4m3 = _tep_gqa_e4m3(self, device)
        # A tep4 rank's in-place attention prologue (`TEP_ATTN_INPLACE_ENV`), built below.
        self.tep_attn_inplace = False
        # A tep4 rank's one-launch draft prologue (`TEP_DRAFT_PROLOGUE_ENV`), built below.
        self.tep_draft_prologue = False
        self.decode_mm = (
            _one_rank_projection
            if one_rank
            else (_tep_wide_projection if self.tep_proj else _decode_projection)
        )
        # At one rank a trtllm-gen decode MoE leaves its finalize to the next site
        # (`DecodeTarget._moe`).
        self.moe_fold = one_rank and self.trunk_moe_backend == "TRTLLM"
        # At one rank the hyper-connection sites' fast forms by row count (the decode GEMV and
        # tcgen05 rows pairs, the prefill rstd sites); the other layouts do not take them.
        self.hc_fast_sites = one_rank
        # The decode sites' rows pairs alone: at one rank, and at a tep4 rank under
        # `TEP_FAST_DECODE_ENV` (replicated sites over the all-reduced rows).
        self.hc_decode_pairs = one_rank or self.tep_fast
        in_wide = 1 if self.tep_proj else 0
        self.decode_in_mm = (
            _one_rank_projection
            if one_rank
            else (lambda mat_a, mat_b: low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL, in_wide))
        )
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        if device.type == "cuda":
            self.hc_tiles = _hc_tiles_build(self, device)
            self.hc_first_x2 = _hc_first_x2_build(self, device)
            self._inproj_release = _inproj_release_build(self, device)
            self._gdn_lean_want = _gdn_lean_build(self, device)
            self._out_proj = _out_proj_build(self, device)
            self._draft_o = _draft_o_build(self, device)
            self._siteb = _siteb_build(self, device)
            self.dp_scores_early = _dp_scores_early(self, device)
            self.topk_cowrite = _topk_cowrite_build(self, device)
            self._draft_moe_tactics = _draft_moe_tactics_build(self, device)
            self.tep_attn_inplace = _tep_attn_inplace_build(self, device)
            self.tep_draft_prologue = _tep_draft_prologue_build(self, device)
            self.dp_launch_order = _switch(DP_LAUNCH_ORDER_ENV, default=True) and (
                _dp_decode_rank(self, device)
            )
            _warn_dp_row_floors_ignored(self)
        # Built after the decode target, whose layer views it checks; a collective-free build.
        self._route_front = (
            _route_front_build(self, self._decode._layers, device)
            if device.type == "cuda"
            else None
        )
        # Collective (the group agrees); once a load.
        if self._router_front is None and device.type == "cuda":
            self._router_front = _dp_router_front_build(self, self._decode._layers, device)
        # Collective (every rank checks the same row counts and the group agrees); once a load.
        if self._rs_inject is None and device.type == "cuda":
            self._rs_inject = _dp_rs_inject_build(self, device)
        if device.type == "cuda" and _dp_decode_rank(self, device):
            self.dp_draftpack = _dp_draftpack_parts(self, self._rs_inject, device)
        # Collective where set (the group agrees and checks); once a load.
        if self._dp_pack is None and device.type == "cuda":
            self._dp_pack = _dp_pack_build(self, device)
        # Collective where set: every rank of the group agrees on its row counts.
        _tep_decode_front_build(self, self._route_front, self._decode._layers, device)
        # Collective; once a load (the post-load walk reaches this hook again).
        if self._tep_finar is None and device.type == "cuda":
            self._tep_finar = _tep_finar_build(self, device)
        # Collective where set; once a load. The walk above rebuilds `tp_sum` / `expert_sum` at
        # every pass, so a later pass binds the built launch again.
        if getattr(self, "_tep_tpsum", None) is None and device.type == "cuda":
            self._tep_tpsum = _tep_tpsum_build(self, device)
        elif getattr(self, "_tep_tpsum", None) is not None:
            _tep_tpsum_bind(self, self._tep_tpsum)
        if device.type == "cuda" and not one_rank:
            logger.info(_decode_forms(self))

    def _build_draft_one_rank(self, device) -> None:
        """At one rank, and at a tep4 rank under `TEP_DRAFT_LAYER_ENV` (over its heads), the
        draft layer's operands for `DecodeTarget._mtp_layer`: the fused
        `[q | k | v | gate | index | pad]` attention input (`weights._attn_in_w`), and at one rank
        with `_MTP_DENSE_FP8_ENV` the packed FP8 copies of its dense projections; None / empty at
        every other layout. At an attention-DP rank under `DP_MTP_ATTN_PACK_ENV` the fused
        attention input alone (`DecodeTarget._mtp_attn_front`)."""
        self._mtp_attn_in = None
        self._mtp_gemv8 = {}
        if self.dp_mtp_attn_pack and not _dp_decode_rank(self, device):
            logger.warning(
                f"{DP_MTP_ATTN_PACK_ENV} is set but ignored: it was measured on SM103 GPUs only"
            )
            self.dp_mtp_attn_pack = False
        if not (self.one_rank or self.tep_draft_layer or self.dp_mtp_attn_pack):
            return
        w = self.w
        p = f"l{self.mtp_layer_idx}_mtp_"
        attn_in = _weights._attn_in_w(
            (w[f"{p}q_w"], w[f"{p}k_w"], w[f"{p}v_w"], w[f"{p}idx_w"]), self
        )
        assert tuple(attn_in.shape) == (self.attn_in_width, self.hidden), (
            f"the draft layer's fused attention input is {tuple(attn_in.shape)}; this rank's "
            f"projection row is {self.attn_in_width} wide over hidden {self.hidden}"
        )
        self._mtp_attn_in = attn_in.t()
        # The draft-side precision copies below stay one-rank switches.
        if device.type != "cuda" or not self.one_rank:
            return
        if _switch(_MTP_DENSE_FP8_ENV):
            self._mtp_gemv8 = {
                "fc_e": pack_from_bf16(w[f"{p}fc_e"]),
                "fc_h": pack_from_bf16(w[f"{p}fc_h"]),
                "attn_in": pack_from_bf16(attn_in),
                "o": pack_from_bf16(w[f"{p}o_w"]),
                "sh_gu": pack_from_bf16(w[f"{p}sh_gu"][: 2 * self.shared_inter]),
                "sh_dn": pack_from_bf16(w[f"{p}sh_dn"]),
            }

    def _build_dense_copies(self, device) -> None:
        """The trunk's dense FP8 copies (`_DENSE_FP8_ENV`) beside the bf16 weights the one-row
        GEMV and fused paths keep reading, and the GDN in_proj's NVFP4 copies
        (`_GDN_INPROJ_NVFP4_ENV`), with every decode variant compiled now; both only at tp1,
        the layout they are certified at, and empty otherwise."""
        w = self.w
        one_rank = self.one_rank
        self._dense8, self._gemv8, self._mid, self._gemv_nv = {}, {}, {}, {}
        if one_rank and _switch(_DENSE_FP8_ENV):
            for name in w:
                if re.fullmatch(r"l\d+_(attn_in_w|o_w|in_proj_w|out_w)", name):
                    codes, scale = fp8_per_tensor_weight(w[name])
                    self._dense8[name] = (codes.t(), scale)
                    self._gemv8[name] = pack_from_bf16(w[name])
                    self._mid[name] = MidWeight(codes, scale)
                elif re.fullmatch(r"l\d+_sh_gu", name):
                    # The shared expert's gate|up without its gate-logit row, which stays bf16.
                    self._gemv8[name] = pack_from_bf16(w[name][: 2 * self.shared_inter])
                    self._mid[name] = MidWeight(w[name][: 2 * self.shared_inter])
                elif re.fullmatch(r"l\d+_sh_dn", name):
                    self._gemv8[name] = pack_from_bf16(w[name])
                    self._mid[name] = MidWeight(w[name])
                elif re.fullmatch(r"l\d+_router", name):
                    self._mid[name] = MidWeight(w[name])
            if device.type == "cuda":
                _precompile_mid(self._mid)
        if one_rank and _switch(_GDN_INPROJ_NVFP4_ENV) and device.type == "cuda":
            for name in w:
                if re.fullmatch(r"l\d+_in_proj_w", name):
                    self._gemv_nv[name] = nvfp4_pack_split_from_bf16(w[name])
            if self._gemv_nv:
                precompile_split(
                    next(iter(self._gemv_nv.values())),
                    _INPROJ_NVFP4_MAX_ROWS,
                    self.conv_dim,
                    _PAD_SLOT_ID,
                    tuple(range(2, self.max_verify_rows + 1)),
                )

    def _tp_sum_op(self):
        """The sum that completes a block's per-rank partial output, settled once: under
        attention TP the plain all-reduce over the TP group (`comm/allreduce`, op `NONE`, at
        the engine's `allreduce_strategy`, AUTO for `_MODULE_ONLY_STRATEGIES`), which every rank
        gets the same bits back from; otherwise nothing to sum."""
        if self.attn_tp_size == 1:
            return lambda x: x
        strategy = self.allreduce_strategy
        if strategy in _MODULE_ONLY_STRATEGIES:
            strategy = int(AllReduceStrategy.AUTO)
        group = allreduce_group_args(self.mapping, strategy)

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

        return tp_sum

    def _expert_sum_op(self):
        """How a MoE block's routed experts become the rows this rank owns, settled once.
        The op is `expert_sum(run, x, pad_rows)`, where `run(rows)` is this rank's expert
        window over `rows` and `pad_rows` is `_dp_pad_rows`'s count.

        Under attention TP (or at one rank) every rank holds every row: the window's partial
        over `x`, completed by `tp_sum`. Under attention DP each rank holds its own rows, as
        gpt-oss's and deepseek's default targets run it: pad them to the group's largest row
        count, gather the group's rows (`comm/allgather`, in rank order), run the window over
        all of them, reduce-scatter (`comm/reducescatter`), which sums each row's window
        partials once and hands this rank its slice, and drop the padding. Routing on the
        gathered rows makes every rank's routing of a row the same, so each of its top-k
        experts falls in exactly one window."""
        if self.dp_size == 1:
            tp_sum = self.tp_sum

            def window_sum(run, x: torch.Tensor, pad_rows: int) -> torch.Tensor:
                return tp_sum(run(x))

            return window_sum
        group = self.dp_group
        # Under MPI the plain symbols; with MPI disabled the `_pg` ones over the TP process
        # group, as `allreduce_group_args` routes them.
        process_group = self.mapping.tp_group_pg.boxed() if mpi_disabled() else None
        lamport = self._dp_lamport(group) if _dp_moe_comm() != "nccl" else None
        group_size = len(group)

        def round_trip(run, x: torch.Tensor, pad_rows: int, tail=None) -> torch.Tensor:
            rows = x.shape[0]
            padded = x if not pad_rows else nn.functional.pad(x, [0, 0, 0, pad_rows])
            if lamport is not None and lamport.takes(padded.shape[0], padded):
                # Every rank takes the same branch: the padded row count and the dtype are the
                # group's. The window's output is bf16 at the hidden width on every rank.
                front = getattr(self, "_router_front", None)
                release = front is not None and padded.shape[0] * group_size in front.release_rows
                partial = run(dep_lamport_round_trip(padded, lamport, False, release))
                if tail is not None and not pad_rows:
                    # `DP_RS_INJECT_ENV` (`_rs_inject_tail`): the reduce-scatter with the next
                    # site's injection and norm, an `_RsInjected`; None keeps the plain one (the
                    # same bytes travel either way).
                    fused = tail(partial)
                    if fused is not None:
                        return fused
                mine = dep_lamport_round_trip(partial, lamport, True)
                return mine if not pad_rows else mine[:rows]
            gathered = allgather(padded, None, group, process_group)
            mine = reducescatter(run(gathered), None, group, process_group)
            return mine if not pad_rows else mine[:rows]

        return round_trip

    def _dp_lamport(self, group):
        """The Lamport round trip of `DP_MOE_COMM_ENV` (`ops/moe/dep_lamport_round_trip`), built
        once a load over the DP group: a collective, every rank of the group builds it at
        construction. At dep4 over torch symmetric memory on the 4-rank torch group, at dep8 over
        `PeerBuffers` of the TP group (MNNVL across the two nodes); with `lamport_step` in the
        host-index form, whose registry (`_lamport_steps`) the shell's forward brackets. ValueError
        away from a dep4 or dep8 worker on SM103; RuntimeError on every rank where the group has no
        peer memory."""
        if self._lamport_trip is not None:
            return self._lamport_trip
        world = self.dp_size
        layout = _LAMPORT_LAYOUTS.get(world)
        if (
            layout is None
            or self.mapping.world_size != world
            or list(group) != list(range(world))
            or torch.cuda.get_device_capability() != archs.SM_103
        ):
            raise ValueError(
                f"{DP_MOE_COMM_ENV}={_dp_moe_comm()}: the Lamport round trip is certified on a "
                f"dep4 or dep8 worker on SM103 (this rank: world size {self.mapping.world_size}, "
                f"attention DP size {self.dp_size}, device "
                f"sm_{''.join(map(str, torch.cuda.get_device_capability()))})"
            )
        steps = LamportSteps() if _dp_moe_comm() == "lamport_step" else None
        rows = int(self.max_num_tokens)
        if world == 4:
            built = self._dp_lamport_symmetric(rows, steps)
        else:
            built = self._dp_lamport_peers(rows, steps)
        self._lamport_trip, self._lamport_steps = built, steps
        logger.info(
            f"MoE Lamport round trip: {layout}, up to {built.max_rows} rows a rank, "
            f"{'host-index' if steps is not None else 'device-counter'} slots"
        )
        return built

    def _dp_lamport_symmetric(self, rows: int, steps):
        """dep4: the workspace over torch symmetric memory on the 4-rank torch group."""
        import torch.distributed as dist

        from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

        # MPI workers have no torch process group: the collective bootstrap the stock MoE uses.
        ensure_cutedsl_torch_distributed(4)
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise RuntimeError(f"{DP_MOE_COMM_ENV}={_dp_moe_comm()} found no 4-rank torch group")
        if dist.get_rank() != self.dp_rank:
            raise RuntimeError(
                f"{DP_MOE_COMM_ENV}={_dp_moe_comm()}: the torch WORLD rank order differs from the "
                "DP rank order"
            )
        return dep_lamport_build(self.hidden, rows, self.dp_rank, 4, steps=steps)

    def _dp_lamport_peers(self, rows: int, steps):
        """dep8: the workspace over `PeerBuffers` of the TP group (= the DP group): MNNVL memory
        across the nodes of one NVLink domain, IPC within a node. Every rank agrees on the
        outcome before any rank returns: a refusal or a failed build raises on every rank."""
        from tensorrt_llm._torch.distributed.communicator import Distributed
        from tensorrt_llm._torch.distributed.peer_buffers import PeerBuffers

        world, rank = self.dp_size, self.dp_rank
        nbytes = dep_lamport_workspace_bytes(self.hidden, rows, world)
        peers = PeerBuffers.create(self.mapping, nbytes)
        if peers is None:
            raise RuntimeError(
                f"{DP_MOE_COMM_ENV}={_dp_moe_comm()} at dep8: the TP group has no peer memory "
                "(IPC within a node, MNNVL across the nodes of one NVLink domain); unset it to "
                "keep NCCL"
            )
        built, error = None, ""
        try:
            buffers = dep_lamport_buffers_from_pointers(
                peers.peer_ptrs, self.hidden, rows, world, torch.cuda.current_device()
            )
            built = dep_lamport_build_on(
                self.hidden, rows, rank, world, buffers, steps=steps, keepalive=peers
            )
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 -- every rank raises together below
            error = f"{type(e).__name__}: {e}"
        # Every rank holds its armed buffer and both launches before any of them exchanges a row.
        errors = [e for e in Distributed.get(self.mapping).tp_allgather(error) if e]
        if errors:
            raise RuntimeError(
                f"{DP_MOE_COMM_ENV}={_dp_moe_comm()} at dep8: the Lamport workspace failed on "
                f"{len(errors)} rank(s): {errors[0]}"
            )
        return built

    def _dp_pad_rows(self, md, num_tokens: int) -> int:
        """Rows this rank pads its MoE input with before the attention-DP round trip: up to
        the group's largest per-rank token count, so both collectives run in their uniform
        form. Zero without attention DP.

        A decode CUDA graph pads every rank to the captured batch, and a prefill CUDA graph
        every rank to one bucket (the engine then reports the bucket for every rank), so a
        captured step's counts are equal and its padding is zero rows."""
        if self.dp_size == 1:
            return 0
        counts = [int(n) for n in md.all_rank_num_tokens]
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.dp_rank] == num_tokens, (
            f"rank {self.dp_rank} holds {num_tokens} rows but all_rank_num_tokens says "
            f"{counts[self.dp_rank]} ({counts}); the collectives would gather the wrong split"
        )
        return max(counts) - num_tokens

    def _index_rope_table(self, theta: float, device) -> torch.Tensor:
        """The index compression's rope table at `theta` in the QSA kernels' layout:
        `[max_positions, 2 * rotary_pairs]` fp32, cos block then sin block."""
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64, device=device)
        inv_freq = (1.0 / (theta ** (exponent / self.rotary_dim))).to(torch.float32)
        positions = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()

    def _report_ple_residency(self) -> None:
        """Check where the n-gram table ended up, and report it once: offload is
        output-identical, so the pointer's memory kind read back from the driver is the
        signal that cannot be faked."""
        table = self.w["ple_table"]
        host = self._ple_table_host
        if host is not None:
            assert host.resident_on_host(), (
                f"the n-gram table ({host.describe()}) was declared in host memory, but the "
                "driver reports device memory for the pointer the forward gathers from"
            )
        else:
            kind = _weights.memory_kind(table.data_ptr())
            assert kind == _weights.CUDA_MEMORY_TYPE_DEVICE, (
                f"the n-gram table was declared with {_PLE_HOST_OFFLOAD_ENV}=0, but the driver "
                f"reports cudaMemoryType {kind} for the pointer the forward gathers from"
            )
        mib = 1024.0 * 1024.0
        free, total = torch.cuda.mem_get_info()
        print(
            f"ple n-gram table: {tuple(table.shape)} {table.dtype}, "
            f"{table.numel() * table.element_size() / mib:.1f} MiB, "
            f"{f'HOST-resident ({host.describe()})' if host is not None else 'DEVICE-resident'}"
            f"; after load: allocated {torch.cuda.memory_allocated() / mib:.1f} MiB, "
            f"driver-used {(total - free) / mib:.1f} MiB of {total / mib:.1f} MiB",
            flush=True,
        )

    def release_host_resident_weights(self) -> None:
        """Close the host-resident n-gram table and, in `node`, its broker threads."""
        _weights._close_ngram_table(self)

    # ---------------------------------------------------------------- bind ---

    def _bind_cache_manager(self, md, position_ids) -> None:
        """Bind this rank's pool views of the step's KV cache manager, checked against the
        geometry the weights were cut to.

        Keyed on the manager, not a first-forward flag: capacity estimation builds a
        throwaway manager and then the real one on the same module, and a view held across
        the swap dangles silently. The contract checks re-run on every bind."""
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
        # The manager's per-rank share is the one the weights were cut to.
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
        # `qsa_topk_indices` rows are addressed as `block_topk`-wide rows.
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
        # The pool serves both state dtypes; this layout serves its own (routing declines the rest).
        served = _ONE_RANK_SSM_STATE_DTYPES if self.one_rank else _SSM_STATE_DTYPES
        state_dtype = rank_pools.ssm[self.gdn_layers[0]].dtype
        assert state_dtype in served, (
            f"a {state_dtype} recurrent-state cache; this layout serves "
            f"{' or '.join(str(d) for d in served)} (kv_cache_config.mamba_ssm_cache_dtype)"
        )
        self.gdn_replay_map = self.mtp_enabled and _gdn_replay_map_on(
            self,
            cm,
            rank_pools.ssm[self.gdn_layers[0]].device,
            bool(getattr(cm, "use_gdn_flashinfer_ring_replay", False)),
        )
        if self.mtp_enabled and state_dtype == torch.bfloat16:
            # Every CuTe replay kernel a verify step of up to `max_num_requests` requests
            # launches (`_gdn_verify`), compiled now: the captured decode graphs do not reach
            # every launch mapping, and a compile in a serving step stalls it 5-10 s.
            maps = {}
            if self.gdn_replay_map:
                maps = {n: _gdn_replay_map(n) for n in range(1, md.max_num_requests + 1)}
                maps = {n: c for n, c in maps.items() if c is not None}
                logger.info(
                    f"GDN replay launch map ({DP_GDN_REPLAY_MAP_ENV}): rank {self.dp_rank} of "
                    f"{self.dp_size}, mapped at {sorted(maps)} requests"
                )
            self._gdn_wi_width = mm.replay_work_items.shape[1]
            gdn_replay_precompile(
                self.max_verify_rows,
                self.num_k_heads,
                self.num_v_heads,
                md.max_num_requests,
                pdl=_GDN_REPLAY_PDL,
                partition_from=CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
                work_item_width=mm.replay_work_items.shape[1],
                separate_replay_slots=_has_replay_slots(cm),
                configs=maps,
            )
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
        # The pool sizes the entries' page- and slot-addressed cells have to cover, reported
        # on every rank (the executor's workers keep the info log level off).
        first_attn, first_gdn = self.attn_layers[0], self.gdn_layers[0]
        print(
            f"modeling_v2 qwen4_exp default target: rank {self.mapping.rank} binds "
            f"{rank_pools.kv[first_attn].k.shape[0]} K/V pages, "
            f"{rank_pools.index_cache[first_attn].shape[0]} QSA index pages, "
            f"{rank_pools.conv[first_gdn].shape[0]} recurrent-state slots and "
            f"{rank_pools.ple_state[0].shape[0]} PLE slots (bind {self._binds})",
            flush=True,
        )
        # The tail of an in-forward snapshot split runs FlashInfer's in-place path. The manager
        # caps its tails at `inforward_snapshot_max_tail`; one that does not cannot be served.
        max_tail = getattr(cm, "inforward_snapshot_max_tail", 0)
        if getattr(cm, "inforward_snapshot_rows", 0) > 0 and max_tail > _GDN_INPLACE_MAX_TOKENS:
            raise NotImplementedError(
                f"the KV cache manager {type(cm).__name__} plans in-forward Mamba snapshots with "
                f"a tail of up to {max_tail} tokens: this target splits tails of at most "
                f"{_GDN_INPLACE_MAX_TOKENS} tokens (`inforward_snapshot_max_tail`)"
            )
        bound = _Bound(cm, rank_pools)
        if self.spec_rollback is not None:
            self.spec_rollback.set_index_planes(
                [rank_pools.index_cache[i] for i in self.attn_layers]
            )
        if self.mtp_enabled:
            bound = bound._replace(**self._bind_step_buffers(cm, g, mm, rank_pools))
        self._bound = bound
        # `DP_GDN_CONV_LEAN_ENV`: nothing unless the build wished it.
        _gdn_lean_bind(self, cm, md.max_num_requests)

    def _bind_step_buffers(self, cm, g: _pool.PoolGeometry, mm, pools: _pool.RankPools) -> dict:
        """The verify step's runtime-owned buffers of one manager: every verify row's conv
        window goes to a per-step buffer that `update_mamba_states` promotes from, and the
        recurrent state is the cached replay (a checkpoint per slot plus a compact history of
        the tokens accepted since it). Returns the `_Bound` fields it adds."""
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
        assert layer_caches[g0].intermediate_ssm is None, (
            "the cache manager runs the cached-replay verify path and also allocated a "
            "per-step recurrent buffer; the runtime allocates one or the other"
        )
        ssm0 = pools.ssm[g0]
        history, width = meta.replay_history_size, meta.replay_step_width
        # The bookkeeping has one row per replay slot; a step indexes it, and the history,
        # with the metadata's `replay_state_indices`.
        replay_slots = meta.prev_num_accepted_tokens.shape[0]
        _pool.check_replay_state(
            history, width, meta.prev_num_accepted_tokens, meta.cache_buf_idx, ssm0, replay_slots
        )
        if getattr(cm, "use_gdn_flashinfer_ring_replay", False):
            # FlashInfer's ring replay (the manager chose it at load: MTP with 3 or 7 drafts
            # over a bf16 state): a 32-row ring of history a slot and layer, folded in-kernel.
            ring = {}
            for i, cache in layer_caches.items():
                ring[i] = (cache.k_ring, cache.u_ring, cache.g_ring)
                _check_replay_ring(g, i, ssm0.shape[0], *ring[i])
            cache_base = layer_caches[g0].cache_base
            assert cache_base is not None and tuple(cache_base.shape) == (ssm0.shape[0],), (
                "the ring replay's per-slot start is missing or not one entry per slot"
            )
            replay = _Replay(
                {},
                meta.cache_buf_idx,
                meta.prev_num_accepted_tokens,
                history,
                width,
                ring,
                cache_base,
            )
            return dict(int_idx=int_idx, int_conv=int_conv, replay=replay)
        old = {}
        for i, cache in layer_caches.items():
            old[i] = (cache.old_x, cache.old_B, cache.old_dt, cache.old_dA_cumsum)
            _pool.check_replay_history(g, i, history, replay_slots, *old[i][:3])
        replay = _Replay(old, meta.cache_buf_idx, meta.prev_num_accepted_tokens, history, width)
        return dict(int_idx=int_idx, int_conv=int_conv, replay=replay)

    def _bind_draft_cache_manager(self, md) -> None:
        """Bind the draft layer to the manager its draft step sees: its own index plane, that
        manager's position plane and its paged K/V.

        Under attention TP that is the separate draft KV cache the worker swaps in for the
        draft loop, whose pools hold the draft layer's pages and none of the trunk's;
        otherwise the trunk's manager, which maps the draft layer as well. Keyed on the
        manager like `_bind_cache_manager`, and never first bound under capture: the runner
        runs every captured batch shape eagerly first."""
        cm = md.kv_cache_manager
        assert not torch.cuda.is_current_stream_capturing(), (
            f"the draft layer is not bound to KV cache manager {type(cm).__name__} (first use, "
            "or the manager changed since its last bind) and CUDA graph capture is active"
        )
        _pool.require(
            cm,
            ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers", "layer_offsets"),
            "reads the draft layer's QSA index plane, position plane and paged K/V",
        )
        g = _pool.pool_geometry(
            self.model_config.pretrained_config,
            self.mapping,
            tokens_per_block=md.tokens_per_block,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            dtype=self.w["embed"].dtype,
        )
        layer = self.mtp_layer_idx
        index_cache = cm.get_index_k_buffer(layer)
        position_cache = cm.get_qsa_position_buffer()
        _pool.check_position_plane(g, position_cache)
        kv = _pool.kv_planes(cm, g, layer, index_cache)
        share_slot = None
        if self.mtp_index_share:
            # The metadata sizes its shared selection buffers from the same resolved knob,
            # one row per sparse layer of whichever manager is bound.
            assert md.sparse_metadata_params.mtp_index_share, (
                "this target resolved IndexShare on but the engine's QSA metadata did not"
            )
            slots = cm.qsa_shared_topk_slots
            _pool.check_shared_selection(md.qsa_shared_topk_indices, slots, layer, self.block_topk)
            share_slot = slots[layer]
        print(
            f"modeling_v2 qwen4_exp default target: rank {self.mapping.rank} binds the draft layer to "
            f"{type(cm).__name__} ({'the trunk manager' if cm is self._bound.cm else 'separate'}"
            f"): {kv.k.shape[0]} K/V pages, {index_cache.shape[0]} QSA index pages"
            f"{'' if share_slot is None else f', IndexShare slot {share_slot}'}",
            flush=True,
        )
        self._draft_bound = _DraftBound(cm, index_cache, position_cache, kv, share_slot)

    def _draft_step_bind(self, md) -> tuple[_DraftBound, int]:
        """The draft layer's bind for this draft step's manager, and the layer's pool offset
        in it (what `thop_attention` addresses), checked on every call: the worker swaps the
        manager and the page tables in for the draft loop, and a page table still decoding
        the trunk's offsets would read and write another pool's pages."""
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
        """Which target runs this step: decode if `num_contexts == 0`, else prefill."""
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

        The shell calls it before `forward`, on every step: a breakable prefill CUDA graph
        captures `forward` and replays it without calling it, so whatever a step reads from
        the host is read here. `vision` is `(rows, mm_token_indices)` when the step carries
        media."""
        assert self._prefill is not None and self._decode is not None, (
            "load_weights must run before forward"
        )
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert lora_params is None, "LoRA is not implemented by this target"
        # The PLE layer hashes token ids, so they are required on every step.
        assert input_ids is not None, "this target's PLE layer hashes token ids"
        assert inputs_embeds is None, (
            "this target computes its token embedding itself; an engine-supplied "
            "inputs_embeds is not implemented"
        )
        # `spec_metadata` is present exactly when drafting. One-model MTP-Eagle captures no
        # trunk layer, so the trunk owes no capture hook.
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
        """Run the step `begin_step` prepared on the target it chose. Under a breakable
        prefill CUDA graph this is the body the engine captures; nothing here may read the
        step from the host. Returns the logits input, or under speculation the decode
        target's `(final, hc)` pair and the prefill target's one packed tensor
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
        `[T, hidden + stream]` tensor (a breakable prefill graph's body returns one tensor),
        of which the two are column views."""
        if isinstance(output, tuple):
            return output
        final, hc = torch.split(output, [self.hidden, self.stream], -1)
        return final, hc


#: Verify rows an attention-DP rank gathers for its heads in one `embed_rows` launch (a copy:
#: bitwise `nn.functional.embedding`): PyTorch's small-index gather runs 4-9 us a call at 4-16
#: rows, and switches to its vectorized kernel itself from 32 rows.
_VERIFY_ROWS_GATHER_MAX = 31  # capacity: fixed: a dispatch threshold


def _head_rows(core, gather_ids: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
    """The rows of `final` the verify head reads (`gather_ids`) in one `embed_rows` launch (a
    copy: bitwise the gather) where it takes the ids, else `nn.functional.embedding`: under
    attention DP up to `_VERIFY_ROWS_GATHER_MAX` of them, and at a tep4 rank under
    `TEP_FAST_DECODE_ENV` (`core.tep_fast`) at any count."""
    if not (
        gather_ids.dim() == 1
        and gather_ids.is_contiguous()
        and gather_ids.dtype in (torch.int32, torch.int64)
        and final.dim() == 2
        and final.stride(1) == 1
    ):
        return nn.functional.embedding(gather_ids, final)
    dp = (
        getattr(core, "dp_size", 1) > 1
        and gather_ids.shape[0] <= _VERIFY_ROWS_GATHER_MAX
        and final.is_cuda
    )
    if dp or getattr(core, "tep_fast", False):
        return embed_rows(gather_ids, final, 1, False)
    return nn.functional.embedding(gather_ids, final)


# --------------------------------------------------------------- weight views ---


class _LayerWeights(NamedTuple):
    """One trunk layer's weights as the views the ops read (GEMM operands are `.t()` views).
    Attention fields are None on a Gated-DeltaNet layer and vice versa."""

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
    out1_scale: torch.Tensor
    out1_gate_scale: torch.Tensor
    out2_scale: torch.Tensor
    # `[gate | up | gate logit | pad]` (`weights._cat_rows`), and the down projection.
    sh_gug: torch.Tensor
    sh_dn: torch.Tensor
    # Its gate|up rows and its gate-logit row, as row views of `sh_gug`.
    sh_gu: torch.Tensor
    sh_gate: torch.Tensor
    # The dense FP8 mode's copies (`_DENSE_FP8_ENV`), None without it: per-tensor FP8
    # `(codes_t, scale)`, packed `fp8_gemv` copies and `mid_rows_mm` views; and the GDN
    # in_proj's NVFP4 copy (`_GDN_INPROJ_NVFP4_ENV`).
    attn_in8: tuple | None = None
    o8: tuple | None = None
    in_proj8: tuple | None = None
    out8: tuple | None = None
    attn_in_g: object | None = None
    o_g: object | None = None
    in_proj_g: object | None = None
    out_g: object | None = None
    sh_gu_g: object | None = None
    sh_dn_g: object | None = None
    in_proj_nv: object | None = None
    attn_in_m: object | None = None
    o_m: object | None = None
    in_proj_m: object | None = None
    out_m: object | None = None
    sh_gu_m: object | None = None
    sh_dn_m: object | None = None
    router_m: object | None = None
    # The routed experts' CuteDSL-layout copy and its scalars (`_moe_layer_scalars_cd`), which
    # prefill-sized launches run on at one rank (`moe_prefill_copy`); None otherwise.
    fc1_w_cd: torch.Tensor | None = None
    fc1_s_cd: torch.Tensor | None = None
    fc2_w_cd: torch.Tensor | None = None
    fc2_s_cd: torch.Tensor | None = None
    alpha1_cd: torch.Tensor | None = None
    alpha2_cd: torch.Tensor | None = None
    global_sf_cd: torch.Tensor | None = None
    # On the CuteDSL backend the experts' own scalars (`_moe_layer_scalars_cd`); None on
    # trtllm-gen.
    alpha1: torch.Tensor | None = None
    alpha2: torch.Tensor | None = None
    global_sf: torch.Tensor | None = None
    # `1 + ahc_norm` / `1 + mhc_norm` in fp32, contiguous (`_norm_w1`): the rstd mixers'
    # operands (`_HC_RSTD_MIN_ROWS`); set by `PrefillTarget` only.
    ahc_norm_w1: torch.Tensor | None = None
    mhc_norm_w1: torch.Tensor | None = None


def _layer_views(core: Qwen4ExpDefaultModelingV2) -> list[_LayerWeights]:
    """Every trunk layer's `_LayerWeights`, read off the realized weights."""
    w = core.w
    if getattr(core, "trunk_moe_backend", "TRTLLM") == "CUTEDSL":
        w_dtype, s_dtype = torch.float4_e2m1fn_x2, torch.uint8
    else:
        w_dtype, s_dtype = torch.uint8, torch.float8_e4m3fn
    layers = []
    for i in range(core.num_layers):
        attn: dict = dict(attn_in=None, o=None, q_norm=None, k_norm=None, idx_qn=None, idx_kn=None)
        gdn: dict = dict(in_proj=None, out=None, conv_w=None, a_log=None, dt_bias=None, gnorm=None)
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
        cute_dsl: dict = {}
        if getattr(core, "trunk_moe_backend", "TRTLLM") == "CUTEDSL":
            scalars_cd = core._moe_scalars_cd[i]
            cute_dsl = dict(
                alpha1=scalars_cd["alpha1"],
                alpha2=scalars_cd["alpha2"],
                global_sf=scalars_cd["global_sf"],
            )
        elif getattr(core, "moe_prefill_copy", False):
            scalars_cd = core._moe_scalars_cd[i]
            cute_dsl = dict(
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
                # The expert stacks are uint8 storage; each backend reads the packed nibbles
                # and the block scales as its own dtypes.
                fc1_w=w[f"l{i}_fc1_w"].view(w_dtype),
                fc1_s=w[f"l{i}_fc1_s"].view(s_dtype),
                fc2_w=w[f"l{i}_fc2_w"].view(w_dtype),
                fc2_s=w[f"l{i}_fc2_s"].view(s_dtype),
                out1_scale=scalars["output1_scale_scalar"],
                out1_gate_scale=scalars["output1_scale_gate_scalar"],
                out2_scale=scalars["output2_scale_scalar"],
                sh_gug=w[f"l{i}_sh_gu"].t(),
                sh_dn=w[f"l{i}_sh_dn"].t(),
                sh_gu=w[f"l{i}_sh_gu"][: 2 * core.shared_inter].t(),
                sh_gate=w[f"l{i}_sh_gu"][2 * core.shared_inter : 2 * core.shared_inter + 1].t(),
                **{
                    field: getattr(core, copies, {}).get(f"l{i}_{name}")
                    for field, copies, name in (
                        ("attn_in8", "_dense8", "attn_in_w"),
                        ("o8", "_dense8", "o_w"),
                        ("in_proj8", "_dense8", "in_proj_w"),
                        ("out8", "_dense8", "out_w"),
                        ("attn_in_g", "_gemv8", "attn_in_w"),
                        ("o_g", "_gemv8", "o_w"),
                        ("in_proj_g", "_gemv8", "in_proj_w"),
                        ("out_g", "_gemv8", "out_w"),
                        ("sh_gu_g", "_gemv8", "sh_gu"),
                        ("sh_dn_g", "_gemv8", "sh_dn"),
                        ("in_proj_nv", "_gemv_nv", "in_proj_w"),
                        ("attn_in_m", "_mid", "attn_in_w"),
                        ("o_m", "_mid", "o_w"),
                        ("in_proj_m", "_mid", "in_proj_w"),
                        ("out_m", "_mid", "out_w"),
                        ("sh_gu_m", "_mid", "sh_gu"),
                        ("sh_dn_m", "_mid", "sh_dn"),
                        ("router_m", "_mid", "router"),
                    )
                },
                **cute_dsl,
            )
        )
    return layers


class _StackWeights(NamedTuple):
    """The weights outside the layer loop: the stack-final mixer (this checkpoint's final
    norm) and the PLE side path."""

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
    # `1 + fhc_norm` in fp32 (see `_LayerWeights.ahc_norm_w1`); set by `PrefillTarget` only.
    fhc_norm_w1: torch.Tensor | None = None


def _norm_w1(norm: torch.Tensor) -> torch.Tensor:
    """A grouped norm's `(1 + w)` in fp32, contiguous: what `qwen4_exp_hc_up_gate_mean_rstd`
    multiplies the scaled residual by (the `1 + w` `qwen4_exp_hc_inject_norm` forms per
    element, bitwise)."""
    return (1.0 + norm.float()).contiguous()


def _stack_views(core: Qwen4ExpDefaultModelingV2) -> _StackWeights:
    w = core.w
    return _StackWeights(
        fhc_norm=w["fhc_norm"],
        fhc_first=core._hc_first["fhc"].t(),
        fhc_up=w["fhc_up"].t(),
        ple_key=w["ple_key_w"].t(),
        ple_val=w["ple_val_w"].t(),
        # The three PLE norm weights whole: the fused tail normalizes each stream inside one
        # program.
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
    """The MTP module's weights at this rank's share (keyed at its layer index,
    `num_hidden_layers`), as the views the ops read: GEMM operands are `.t()` views. Each
    hyper-connection site is the trunk's: its whole `[stream]` grouped-norm weight, its
    `[down; block_inject; 0]` first projection (`_fuse_hc_first_projections`) and its up
    projection; `unit_inject` is the front end's per-stream `x_h + x_e` as the first site's
    injection at unit weight."""

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
    q: torch.Tensor  # [hidden, 2 * q_width]: per head its q, then its output gate
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
    # `[gate | up | gate logit | pad]` (`weights._cat_rows`), as the trunk's, and the down
    # projection.
    sh_gug: torch.Tensor
    sh_dn: torch.Tensor
    fhc_norm: torch.Tensor
    fhc_first: torch.Tensor
    fhc_up: torch.Tensor
    # The shared expert's gate|up rows and gate-logit row, row views of `sh_gug`.
    sh_gu: torch.Tensor
    sh_gate: torch.Tensor
    # Native NVFP4 or the one-rank FP8 conversion: `fc1_w` .. `fc2_s` use the trunk's
    # layout, with an activation global scale and the runner's three scalars.
    moe_gscale: torch.Tensor | None = None
    out1_scale: torch.Tensor | None = None
    out1_gate_scale: torch.Tensor | None = None
    out2_scale: torch.Tensor | None = None


def _mtp_views(core: Qwen4ExpDefaultModelingV2) -> _MtpWeights:
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
        sh_gug=w[f"{p}sh_gu"].t(),
        sh_dn=w[f"{p}sh_dn"].t(),
        sh_gu=w[f"{p}sh_gu"][: 2 * core.shared_inter].t(),
        sh_gate=w[f"{p}sh_gu"][2 * core.shared_inter : 2 * core.shared_inter + 1].t(),
        fhc_norm=w[f"{p}fhc_norm"],
        fhc_first=core._hc_first["mtp_fhc"].t(),
        fhc_up=w[f"{p}fhc_up"].t(),
    )


# ------------------------------------------------------------ shared blocks ---


def _verify_rows(core, md, spec_metadata, num_seqs: int, num_rows: int) -> tuple[int, bool]:
    """`(p, is_verify)` of a step's generation part: `num_rows` rows over `num_seqs`
    generation sequences, `p` per sequence -- 1 without speculation, else
    `runtime_draft_len + 1`. Host ints of the metadata, constants of a captured step."""
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


def _has_replay_slots(cm) -> bool:
    """Whether the manager keeps the replay history in slots of its own, which the metadata's
    `replay_state_indices` then address (otherwise it aliases the recurrent-state slots)."""
    return getattr(cm, "replay_layer_group_id", None) is not None


def _verify_state(core, mm, num_decodes: int, num_prefills: int = 0) -> tuple:
    """What a verify step's `_gdn_verify` reads besides the layer: the rows
    `update_mamba_states` promotes the conv windows from, the replay's work items, and the
    generation rows' replay slots (behind `num_prefills` context rows). Work
    items go to the kernel from `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests on, and the
    kernel leaves the history commit to the manager's all-layer commit exactly when the
    manager runs one after the step -- the stock Gated-DeltaNet layer's two conditions. The
    FlashInfer ring replay folds in-kernel and reads no work items; its history shares the
    recurrent-state slots."""
    assert core._bound.int_idx is not None, "the cache-manager bind has not run"
    int_idx = core._bound.int_idx[:num_decodes]
    if core._bound.replay.ring is not None:
        return int_idx, None, None, False, None
    partitioned = num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    all_layer = partitioned and bool(
        getattr(core._bound.cm, "use_gdn_cached_replay_all_layer_commit", False)
    )
    work_items = mm.replay_work_items[:num_decodes] if partitioned else None
    n_writes = mm.replay_n_writes if partitioned else None
    replay_indices = (
        mm.replay_state_indices[num_prefills : num_prefills + num_decodes]
        if _has_replay_slots(core._bound.cm)
        else None
    )
    return int_idx, work_items, n_writes, all_layer, replay_indices


def _tep_conv_wait(core, rows: int) -> bool:
    """Whether a verify step of `rows` rows launches its conv as a programmatic dependent of the
    input projection: a tep4 rank with its fast forms, PDL on, `_TEP_CONV_WAIT_ROWS` rows."""
    lo, hi = _TEP_CONV_WAIT_ROWS
    return bool(getattr(core, "tep_fast", False)) and _GDN_REPLAY_PDL and lo <= rows <= hi


def _gdn_verify(
    core, mq, a, b, i, L, si_d, n, p, verify, mq_out=None, conv_wait=False, z=None
) -> torch.Tensor:
    """The recurrent half of a verify step: `n` generation sequences of `p` rows, written so
    that the accepted prefix, unknown until the sampler has run, can be kept.

    The conv writes each row's window to the runtime's per-step buffer
    (`update_mamba_states` promotes the accepted one) besides advancing the pool; the cached
    replay rebuilds each slot's state from its checkpoint and pending history, runs the `p`
    rows and appends them to the history, so rejected rows cost nothing to undo. `mq` is the
    `[n * p, conv_dim]` q|k|v column view of the input projection, `a` and `b` its raw gate
    columns (column views too): the kernel applies `A_log`, `dt_bias` and the q/k L2 norm
    itself. Over a bf16 recurrent pool the replay is the CuTe kernel (`gdn_cached_replay_cute`),
    which reads `a` / `b` through their row stride; over an fp32 one the Triton kernel, which is
    handed contiguous copies. Both take the same work items and leave the same commits to the
    manager. Returns `[1, n * p, num_v_heads, head_v_dim]`.

    `mq_out` is the conv output when the input projection's GEMV ran the conv step as its
    epilogue (`fp8_gemv_conv1d_verify`, `nvfp4_split_gemv_conv1d_verify`; `mq` is then None).
    `conv_wait`: the conv is launched as a programmatic dependent of the kernel that wrote `mq`
    (`_TEP_CONV_WAIT_ROWS`), with its x rows hoisted; the caller decides.

    `z` (the gate columns): on a step whose replays write the gated norm
    (`DP_GDN_REPLAY_NORM_BF16_ENV`, `_gdn_rnorm_now`) the result is the gated norm's output,
    which `_gdn_epilogue` then takes as it is."""
    int_idx, work_items, n_writes, all_layer, replay_indices = verify
    pools = core._bound.pools
    replay = core._bound.replay
    ssm = pools.ssm[i]
    cute = ssm.dtype == torch.bfloat16
    # Only the CuTe replay is launched as the conv's programmatic dependent.
    pdl = _GDN_REPLAY_PDL and cute and replay.ring is None
    if _gdn_lean_now(core):
        # `DP_GDN_CONV_LEAN_ENV`: the conv output (token-major, as the replay reads it) and the
        # raw rows `gdn_promote_x` builds the accepted window from; no window snapshots, no pool
        # advance. A PDL dependent of the input projection: it reads the slots, the window and
        # the taps before its wait.
        assert cute and pdl, f"layer {i}: a lean step off the lean path"
        if mq_out is None:
            mq_out = gdn_conv_lean(
                mq,
                L.conv_w,
                pools.conv[i],
                si_d,
                int_idx,
                core._gdn_xsave[i],
                p,
                pad_slot_id=_PAD_SLOT_ID,
            )
        else:
            # `DP_GDN_INPROJ_CONV_BF16_ENV`: the input projection's GEMM ran the lean conv as
            # its epilogue (the conv output, the raw rows in `xsave`).
            assert core._gdn_step.inconv and mq is None, f"layer {i}: a conv output off the GEMM"
    if mq_out is None:
        # [n*p, C] -> [n, C, p]: the multi-token conv reads channel-major.
        x3 = torch.transpose(torch.reshape(mq, [n, p, core.conv_dim]), 1, 2)
        conv_out = causal_conv1d_triton_update(
            x=x3,
            conv_state=pools.conv[i],
            weight=L.conv_w,
            bias=None,
            activation="silu",
            conv_state_indices=si_d,
            pad_slot_id=_PAD_SLOT_ID,
            # Snapshots, not the accepted-token roll.
            num_accepted_tokens=None,
            intermediate_conv_window=core._bound.int_conv[i],
            intermediate_state_indices=int_idx,
            launch_dependent_kernels=pdl,
            launch_with_pdl_wait=conv_wait,
            hoist_x=conv_wait,
        )
        # Back to token-major: one contiguous copy, which the kernel reads q/k/v from directly.
        mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
    qv, kv, vv = torch.split(mq_out, [core.key_dim, core.key_dim, core.value_dim], 1)
    if replay.ring is not None:
        # FlashInfer's ring replay reads q/k/v as column slices of `mq_out` (one token stride)
        # and `a` / `b` through their row stride, and the per-row history length and ring
        # start gathered by slot; the manager advances both after the step.
        k_ring, u_ring, g_ring = replay.ring[i]
        slots = si_d.long()
        out = flashinfer_cached_replay_update(
            q=torch.reshape(qv, [n, p, core.num_k_heads, core.head_k_dim]),
            k=torch.reshape(kv, [n, p, core.num_k_heads, core.head_k_dim]),
            v=torch.reshape(vv, [n, p, core.num_v_heads, core.head_v_dim]),
            a=torch.reshape(a, [n, p, core.num_v_heads]),
            b=torch.reshape(b, [n, p, core.num_v_heads]),
            A_log=L.a_log,
            dt_bias=L.dt_bias,
            ssm_states=ssm,
            state_indices=si_d,
            k_ring=k_ring,
            u_ring=u_ring,
            g_ring=g_ring,
            hist_len=replay.prev_num_accepted[slots],
            cache_base=replay.cache_base[slots],
            history_size=replay.history,
        )
        return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])
    old_x, old_B, old_dt, old_dA = replay.old[i]
    extra = {}
    if cute and z is not None and _gdn_rnorm_now(core):
        # `DP_GDN_REPLAY_NORM_BF16_ENV`: the same launch in the norm mode; it writes the gated
        # norm's bf16 output.
        replay_op = functools.partial(
            gdn_replay_norm,
            norm=(torch.reshape(z, [n * p, core.num_v_heads, core.head_v_dim]), L.gnorm, core.eps),
        )
        if core.gdn_replay_map:
            extra["config"] = _gdn_replay_map(n)
    elif cute:
        replay_op = gdn_cached_replay_cute
        if core.gdn_replay_map:
            # `DP_GDN_REPLAY_MAP_ENV`: the measured launch mapping, None for the kernel's own.
            extra["config"] = _gdn_replay_map(n)
    else:
        replay_op = fused_recurrent_gated_delta_rule_cached_replay_update
        a, b = a.contiguous(), b.contiguous()
    out = replay_op(
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
        packed_qkv=mq_out,
        use_all_layer_commit=all_layer,
        output=None,
        replay_indices=replay_indices,
        **extra,
    )
    return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])


def _position_key(position_ids: torch.Tensor) -> tuple:
    """What identifies a step's positions buffer: its storage, dtype and `[3, N]` layout."""
    view = torch.reshape(position_ids, [_QSA_POSITION_AXES, -1])
    return (view.data_ptr(), view.dtype, tuple(view.shape), view.stride())


def _draft_positions(position_ids: torch.Tensor) -> tuple:
    """`(pos3, pos, coords)` of a draft step's mRoPE positions (`[3, 1, N]` from the trunk,
    `[3, N]` from the worker's later steps): a fresh contiguous int32 `[3, N]` copy, its
    coordinate-0 row (the index rope's) and the `[N, 3]` transpose the position plane
    stores."""
    assert position_ids.shape[0] == _QSA_POSITION_AXES, (
        f"expected the mRoPE position layout with {_QSA_POSITION_AXES} coordinate rows, got "
        f"{tuple(position_ids.shape)}"
    )
    pos3 = _copy_positions(position_ids)
    return pos3, pos3[0], torch.transpose(pos3, 0, 1)


def _draft_embedding(input_ids: torch.Tensor, embed_tokens: torch.Tensor) -> torch.Tensor:
    """The MTP front end's token embedding. A draft loop's first call embeds every slot of
    each generation request's accepted tokens, and under rejection sampling
    `chain_speculative_sampling` fills the slots past a request's accepted count with -1
    (the in-tree `Embedding` masks such ids for the same reason). Those rows feed nothing the
    step keeps: attention is causal, the worker continues from each request's last accepted
    row, and the next step's first draft call rewrites their cache positions. So a negative
    id reads row 0 instead of past the table; every valid id embeds as before. The clamp and
    the gather are one launch where `embed_rows` takes the ids (bitwise the composed path)."""
    if input_ids.dim() == 1 and input_ids.is_contiguous():
        return embed_rows(input_ids, embed_tokens, 1, True)
    return nn.functional.embedding(input_ids.clamp_min(0), embed_tokens)


def _mtp_embed_branch(core, M: _MtpWeights, G: dict, embed_tokens, input_ids, gemm):
    """The draft layer's embedding branch on the decode machinery (`DecodeTarget._mtp_layer`):
    `fc_embedding` over the next token's normalized embedding, on `gemm` (or its packed FP8 copy
    at one rank, `G`)."""
    e = _draft_embedding(input_ids, embed_tokens)
    return _draft_mm(
        flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e, G.get("fc_e"), gemm
    )


def _mtp_front(core, M: _MtpWeights, embed_tokens, input_ids, hidden_states) -> tuple:
    """The draft layer's front end and its attention-side hyper-connection. `fc_embedding`
    over the next token's normalized embedding is added to every stream of `fc_hidden`
    (one `[hidden, hidden]` matrix applied to each normalized stream of the trunk's
    residual): the trunk's site with that sum as its injection at unit weight (bitwise the
    bf16 add), then the trunk's mixer. Returns `(x, mixed, inject)`: the layer's residual,
    the mixed attention input and its injection weights."""
    rows = hidden_states.shape[0]
    assert rows <= M.unit_inject.shape[0], (
        f"a draft step of {rows} rows exceeds max_num_tokens {M.unit_inject.shape[0]}"
    )
    e = _draft_embedding(input_ids, embed_tokens)
    x_e = cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
    _, h_n = qwen4_exp_hc_inject_norm(
        hidden_states, None, None, M.pre_h_norm, core.eps, core.hc_count
    )
    x_h = torch.reshape(
        cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
        [rows, core.stream],
    )
    x, normed = qwen4_exp_hc_inject_norm(
        x_h, x_e, M.unit_inject[:rows], M.ahc_norm, core.eps, core.hc_count
    )
    mixed, inject = _hc_mixer(core, normed, M.ahc_first, M.ahc_up, True)
    return x, mixed, inject


def _mtp_qkv(core, M: _MtpWeights, mixed, pos3) -> tuple:
    """The draft layer's q (with its interleaved output gate), k and v projections at this
    rank's heads, packed into one compact `[T, qkv_width]` row and normalized and rotated in
    place at the module's rope base. Returns `(qkv, gate, k, v)`: `gate` is the
    `[T, heads_q, head_dim]` gate half of the q projection, `k` and v` views of `qkv`."""
    rows = mixed.shape[0]
    qg = cublas_mm(mixed, M.q)
    qkv = torch.empty([rows, core.qkv_width], dtype=mixed.dtype, device=mixed.device)
    q_dst, k_dst, v_dst = torch.split(
        torch.reshape(qkv, [rows, core.heads_q + 2 * core.heads_kv, core.head_dim]),
        [core.heads_q, core.heads_kv, core.heads_kv],
        1,
    )
    q_src, gate = torch.split(
        torch.reshape(qg, [rows, core.heads_q, 2 * core.head_dim]),
        [core.head_dim, core.head_dim],
        2,
    )
    q_dst.copy_(q_src)
    k_dst.copy_(torch.reshape(cublas_mm(mixed, M.k), [rows, core.heads_kv, core.head_dim]))
    v_dst.copy_(torch.reshape(cublas_mm(mixed, M.v), [rows, core.heads_kv, core.head_dim]))
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
    return qkv, gate, k_dst, v_dst


def _mtp_index_branch(core, M: _MtpWeights, mixed, draft: _DraftBound, qsa) -> torch.Tensor:
    """The draft layer's QSA index branch over its own planes, as a trunk attention layer
    runs it: the raw key stored before anything normalizes it, the query's norm and rotation
    (coordinate 0 alone, at the module's rope base), then the compression of every group this
    step completes. Returns the normalized, rotated index query."""
    req_idx, logical, block_table, tokens_per_block, pos, coords = qsa
    rows = mixed.shape[0]
    qk_index = cublas_mm(mixed, M.idx)
    token_k = torch.split(
        torch.reshape(qk_index, [rows, core.index_heads + core.index_kv_heads, -1]),
        [core.index_heads, core.index_kv_heads],
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
    index_scratch = torch.empty(
        [rows, core.index_scratch_width], dtype=mixed.dtype, device=mixed.device
    )
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
        position_ids=pos,
        is_qk_norm=True,
        use_gemma=True,
        use_mrope=False,
        mrope_section1=0,
        mrope_section2=0,
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
    _index_anchors_fp8(
        core,
        draft.index_cache,
        block_table,
        req_idx,
        logical,
        tokens_per_block,
    )
    return _index_query(core, index_scratch)


def _mtp_moe_block(
    core,
    M: _MtpWeights,
    x,
    pad_rows: int = 0,
    routed_only=False,
    before_window=None,
    tail=None,
    route=False,
) -> torch.Tensor:
    """The draft layer's MoE: renormalized top-k over the whole routing space, the FP8
    block-scale or native NVFP4 routed experts over this rank's window,
    and the replicated bf16 shared expert in the trunk's form -- one GEMM forms gate|up and
    the per-token gate logit -- combined as `routed + sigmoid(gate logit) * shared`. The
    FP8 activation scales come back transposed `[hidden / 128, tokens]` and go straight to the
    runner.

    The window's partial is completed as the trunk's is: under attention TP summed over the
    TP group; under attention DP by the round trip over the group's gathered rows
    (`core.expert_sum`), `pad_rows` padding this rank's (`_draft_pad_rows`). The routing,
    the quantization and the runner are row-wise and run on the rows the window sees, in
    chunks of at most `_MTP_MOE_MAX_ROWS`: one chunk at a TP rank's rows, several over the
    rows an attention-DP group gathers.

    Under attention DP (`DP_DRAFTPACK_ENV`): `tail` is the round trip's fused reduce-scatter
    (`_rs_inject_tail`; with `routed_only`, which then may return an `_RsInjected`), and `route`
    enters the FP8 runner with the router logits (its own Renormalize routing) instead of
    `renorm_moe_routing_op` + the pre-routed entry."""

    def run(rows: torch.Tensor) -> torch.Tensor:
        if before_window is not None:
            before_window()
        parts = []
        for lo in range(0, rows.shape[0], _MTP_MOE_MAX_ROWS):
            chunk = rows[lo : lo + _MTP_MOE_MAX_ROWS]
            if M.moe_gscale is not None:
                parts.append(_mtp_routed_nvfp4(core, M, chunk))
                continue
            logits = cublas_mm(chunk, M.router)
            if route:
                topk_ids = topk_w = None
            else:
                topk_ids, topk_w = renorm_moe_routing_op(
                    logits, topk=core.topk, output_dtype=chunk.dtype
                )
            xq, xsf = fp8_quantize_1x128(chunk, use_ue8m0=_MTP_MOE_USE_UE8M0)
            parts.append(
                fp8_block_scale_moe_runner(
                    routing_logits=logits if route else None,
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
                    local_expert_offset=core.mtp_expert_offset,
                    local_num_experts=core.mtp_local_experts,
                    routed_scaling_factor=None,
                    routing_method_type=_ROUTING_METHOD_RENORM,
                    topk_weights=topk_w,
                    topk_ids=topk_ids,
                    act_type=_ACT_TYPE_SWIGLU,
                    gemm1_clamp_limit=None,
                    tactic=_draft_moe_tactic(core, chunk.shape[0]),
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    if core.dp_size == 1:
        routed = core.tp_sum(run(x))
    elif tail is None:
        routed = core.expert_sum(run, x, pad_rows)
    else:
        assert routed_only, "the fused reduce-scatter hands back the injected residual"
        routed = core.expert_sum(run, x, pad_rows, tail)
    if routed_only:
        return routed
    return fused_sigmoid_gate_mul_add(routed, *_mtp_shared_expert(core, M, x))


def _mtp_shared_expert(core, M: _MtpWeights, x) -> tuple:
    """The draft layer's replicated bf16 shared expert in the trunk's form over `x`: one GEMM
    forms gate|up and the per-token gate logit. Returns `(gate_logits, shared)`."""
    gug = cublas_mm(x, M.sh_gug)
    shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), M.sh_dn)
    return gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1], shared


def _mtp_routed_nvfp4(core, M: _MtpWeights, x) -> torch.Tensor:
    """The draft layer's local routed experts on NVFP4 operands (`_mtp_moe_nvfp4`):
    the activation quantized at the static FC1 input global scale, then the trunk's
    logits-routed trtllm-gen runner over this rank's expert window, finalized."""
    xq, xsf = fp4_quantize(x.contiguous(), M.moe_gscale, _SF_VEC, False, False)
    return fp4_block_scale_moe_renorm_routed(
        cublas_mm(x, M.router),
        xq,
        torch.reshape(xsf, [-1]).view(torch.float8_e4m3fn),
        M.fc1_w,
        M.fc1_s,
        M.fc2_w,
        M.fc2_s,
        M.out1_scale,
        M.out1_gate_scale,
        M.out2_scale,
        core.num_experts,
        core.topk,
        core.moe_inter,
        core.mtp_expert_offset,
        core.mtp_local_experts,
    )


def _mtp_moe_nvfp4(core) -> dict:
    """Native NVFP4 draft operands, or FP8 experts requantized to NVFP4, as the
    `_MtpWeights` fields `_mtp_routed_nvfp4` reads. Only conversion frees the FP8 copies.
    The native path uses checkpoint scalars. For conversion, the runner's scalars
    follow `_moe_layer_scalars` with `1 / g_in` and `1 / g_mid` standing for the checkpoint's
    `input_scale`s and `1 / g_w` for its `weight_scale_2`: `out1_gate = 1 / (g_in * g_w1[e])`,
    `out1 = out1_gate * g_mid`, `out2 = 1 / (g_mid * g_w2[e])`. The activation scales are static
    (`_MTP_MOE_NVFP4_FC1_MARGIN` / `_FC2_MARGIN`)."""
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"
    keys = [f"{p}{k}" for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")]
    if w[keys[0]].dtype == torch.uint8:
        scalars = _moe_layer_scalars(
            w, core.mtp_layer_idx, core.mtp_expert_offset, core.mtp_local_experts, prefix="mtp_"
        )
        return dict(
            fc1_w=w[keys[0]],
            fc1_s=w[keys[1]].view(torch.float8_e4m3fn),
            fc2_w=w[keys[2]],
            fc2_s=w[keys[3]].view(torch.float8_e4m3fn),
            moe_gscale=scalars["global_scale"],
            out1_scale=scalars["output1_scale_scalar"],
            out1_gate_scale=scalars["output1_scale_gate_scalar"],
            out2_scale=scalars["output2_scale_scalar"],
        )
    fc1_w, fc1_s, fc2_w, fc2_s, g_w1, g_w2 = _weights.mtp_experts_to_nvfp4(*(w[k] for k in keys))
    for k in keys:
        w[k].data = w[k].data.new_empty(0)
    torch.cuda.empty_cache()
    trunk = range(core.num_layers)
    to_global = _weights.NVFP4_AMAX_TO_GLOBAL
    in_amax = max(float(w[f"l{i}_e_isc1"].max()) for i in trunk) * to_global
    mid_amax = max(float(w[f"l{i}_e_isc2"].max()) for i in trunk) * to_global
    inv_g_in = _MTP_MOE_NVFP4_FC1_MARGIN * in_amax / to_global
    inv_g_mid = _MTP_MOE_NVFP4_FC2_MARGIN * mid_amax / to_global
    gate1 = (inv_g_in / g_w1).contiguous()
    return dict(
        fc1_w=fc1_w,
        fc1_s=fc1_s.view(torch.float8_e4m3fn),
        fc2_w=fc2_w,
        fc2_s=fc2_s.view(torch.float8_e4m3fn),
        moe_gscale=torch.full((1,), 1.0 / inv_g_in, dtype=torch.float32, device=fc1_w.device),
        out1_scale=(gate1 / inv_g_mid).contiguous(),
        out1_gate_scale=gate1,
        out2_scale=(inv_g_mid / g_w2).contiguous(),
    )


def _mtp_tail(core, M: _MtpWeights, x, attn, inject, pad_rows: int) -> torch.Tensor:
    """The draft layer after its attention core: the gated output projection's partial summed
    over the TP group and injected, the MoE-side hyper-connection, the MoE and its injection.
    `attn` is the `[T, q_width]` gated attention output; `pad_rows` is the MoE round trip's
    padding under attention DP (`_draft_pad_rows`). The MoE-side site is the trunk's
    (`qwen4_exp_hc_inject_norm`, `_hc_mixer`). The MoE's combination `routed + sigmoid(gate
    logit) * shared` and its injection are one launch (`mtp_moe_combine_inject`, as the FP8
    target runs them), bitwise the two launches it replaces."""
    x, normed = qwen4_exp_hc_inject_norm(
        x, core.tp_sum(_draft_o(core, attn, M.o)), inject, M.mhc_norm, core.eps, core.hc_count
    )
    mixed, inject = _hc_mixer(core, normed, M.mhc_first, M.mhc_up, True)
    if getattr(core, "dp_launch_order", False) and core.dp_size > 1:
        # `DP_LAUNCH_ORDER_ENV`: the shared expert on the side stream beside the round trip,
        # launched right after its gather; joined before the combine.
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        forked = []

        def shared_expert() -> None:
            if not forked:
                with torch.cuda.stream(side):
                    forked.append(_mtp_shared_expert(core, M, mixed))

        # `DP_DRAFTPACK_ENV` (captured passes): `combine` makes the round trip's reduce-scatter
        # the fused launch over this residual with the draft head's norm, `route` the runner's
        # own routing.
        parts = getattr(core, "dp_draftpack", frozenset())
        rsi = getattr(core, "_rs_inject", None)
        captured = bool(parts) and torch.cuda.is_current_stream_capturing()
        tail = None
        if (
            "combine" in parts
            and rsi is not None
            and captured
            and not pad_rows
            and mixed.shape[0] in rsi.draft_rows
        ):

            def operands():
                gate_logits, shared = forked[0]
                return shared, gate_logits

            tail = _rs_inject_tail(core, rsi, (x, inject, M.fhc_norm), operands, main, side)
        kw = {"tail": tail} if tail is not None else {}
        if "route" in parts and captured:
            kw["route"] = True
        routed = _mtp_moe_block(core, M, mixed, pad_rows, True, shared_expert, **kw)
        if type(routed) is _RsInjected:
            # Injected and normalized for the draft head (`headnorm`: `_mtp_head_final`).
            if "headnorm" in parts:
                rsi.stash = routed
            return routed.hc
        shared_expert()  # a round trip that never called its window
        main.wait_stream(side)
        gate_logits, shared = forked[0]
    else:
        routed = _mtp_moe_block(core, M, mixed, pad_rows, routed_only=True)
        gate_logits, shared = _mtp_shared_expert(core, M, mixed)
    if _combine_inject_fits(x, routed, shared, inject):
        return mtp_moe_combine_inject(x, routed, gate_logits, shared, inject)
    return qwen4_exp_post_inject(x, fused_sigmoid_gate_mul_add(routed, gate_logits, shared), inject)


def _combine_inject_fits(x, routed, shared, inject) -> bool:
    """Whether `mtp_moe_combine_inject` takes these operands: 2-D bf16, unit column stride."""
    return all(
        t.dim() == 2 and t.dtype == torch.bfloat16 and t.stride(1) == 1
        for t in (x, routed, shared, inject)
    )


def _decode_projection(mat_a: torch.Tensor, mat_b: torch.Tensor, wide: int = 0) -> torch.Tensor:
    """A generation-only step's dense bf16 projection at every layout but tp1 (the attention
    input and output projections and the GDN output projection of `DecodeTarget`, which the
    engine captures in CUDA graphs; tp1 takes `_one_rank_projection`).

    Up to `_GEMV_MAX_ROWS` rows of the o / GDN out projection at a tp4/tep4 rank (K=1536) run
    `bf16_gemv_pdl`: its weight streams into registers before it waits on the producer.
    Everything else runs `low_m_bf16_mm`, whose table picks FlashInfer's low-M kernels at the
    rows and shapes it names (a tp4/tep4 rank's, and the whole layer's that a dep4 / dep2 rank
    holds; with `wide` its rows past the table, `TEP_PROJ_ENV`) and cuBLAS elsewhere. Both
    launch with PDL per `_DECODE_PROJ_PDL` and read the activation only after
    `griddepcontrol.wait`."""
    if mat_a.shape[0] <= _GEMV_MAX_ROWS and mat_b.shape[0] == _GEMV_TP4_K:
        return bf16_gemv_pdl(mat_a, mat_b, _DECODE_PROJ_PDL)
    return low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL, wide)


def _tep_wide_projection(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """`_decode_projection` with `low_m_bf16_mm`'s tep4 rows past its table (`TEP_PROJ_ENV`)."""
    return _decode_projection(mat_a, mat_b, 1)


def _tep_wide_mm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """A tep4 decode projection that runs on cuBLAS by default (the PLE key and value, the trunk
    shared expert), under `TEP_PROJ_ENV`: `low_m_bf16_mm` with split-K's default tactic where it
    takes the shape, up to `low_m_bf16_mm.WIDE_MAX_ROWS` rows."""
    return low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL, 2)


def _one_rank_projection(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """`_decode_projection` at tp1, where every decode projection's K is one `bf16_gemv_pdl`
    tiles: one row on that GEMV, as up to `_GEMV_MAX_ROWS` rows of a projection at most
    `_GEMV_NARROW_N` wide (the shared expert's gate logit, which `low_m_bf16_mm` would run on one
    CTA); every other call on `low_m_bf16_mm`. Both launch with PDL."""
    rows = mat_a.shape[0]
    if mat_b.shape[0] in _GEMV_TILING and (
        rows == 1 or (rows <= _GEMV_MAX_ROWS and mat_b.shape[1] <= _GEMV_NARROW_N)
    ):
        return bf16_gemv_pdl(mat_a, mat_b, _DECODE_PROJ_PDL)
    return low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL)


class _DraftRowCounts(NamedTuple):
    """The group's per-rank row counts of one draft step, in the field `_dp_pad_rows` reads."""

    all_rank_num_tokens: list


def _draft_pad_rows(core, all_rank_num_tokens, rows: int) -> int:
    """Rows a draft step pads its MoE input with before the attention-DP round trip:
    `_dp_pad_rows` over the counts the spec worker hands this draft step -- every rank's
    trunk tokens on the loop's first step, its request count on the later ones (attention-DP
    dummy requests included) -- which it passes alongside the metadata rather than writing
    into it. Zero without attention DP, and under a decode CUDA graph, which pads every rank
    to the captured batch."""
    if core.dp_size == 1:
        return 0
    assert all_rank_num_tokens is not None, (
        "a draft step under attention DP came without the group's row counts; the MoE round "
        "trip cannot pad this rank's rows to the group's largest count"
    )
    return core._dp_pad_rows(_DraftRowCounts(list(all_rank_num_tokens)), rows)


class _DraftContinue(NamedTuple):
    """A draft-extend step's continuing rows: `rows` `[gen_seqs]` int64, the QSA row metadata
    at them, and under attention DP the group's kept-row counts (None under attention TP)."""

    rows: torch.Tensor
    visible: torch.Tensor
    seq_lens: torch.Tensor
    logical: torch.Tensor
    req_idx: torch.Tensor
    counts: list | None


def _draft_continue(
    core, md, spec_metadata, all_rank_num_tokens, rows: int, gen_seqs: int, p: int, share: bool
) -> _DraftContinue | None:
    """Each request's continuing row of a draft-extend step (`p` rows a request), or None when
    every row goes on.

    The spec worker continues from row `r * p + num_accepted[r] - 1` of each request; the other
    rows matter only through the K/V and index keys they write, which every row still writes.
    Not under IndexShare (its capture keeps every row's selection), nor without the worker's
    accepted counts. Under attention DP the draft MoE's round trip then holds every rank's
    request count, so the decision is the group's: taken only when every rank is
    generation-only with `p` rows a request (`spec_metadata`'s group counts, the same host lists
    on every rank), never from this rank's batch alone."""
    accepted = getattr(md, "qsa_mtp_num_accepted", None)
    if core.one_rank or p <= 1 or share or accepted is None:
        return None
    counts = None
    if core.dp_size > 1:
        seqs = getattr(spec_metadata, "all_rank_num_seqs", None)
        gens = getattr(spec_metadata, "all_rank_num_gens", None)
        if seqs is None or gens is None or all_rank_num_tokens is None:
            return None
        counts = [int(c) for c in seqs]
        if [int(c) for c in gens] != counts or [int(t) for t in all_rank_num_tokens] != [
            p * c for c in counts
        ]:
            return None
        assert counts[core.dp_rank] == gen_seqs and rows == p * gen_seqs, (
            f"rank {core.dp_rank} drafts {gen_seqs} requests in {rows} rows, but the group's "
            f"counts say {counts[core.dp_rank]} requests of {p} rows"
        )
    elif rows != p * gen_seqs:
        return None
    meta = (
        md.qsa_visible_blocks[:rows],
        md.qsa_sequence_lengths[:rows],
        md.qsa_logical_positions[:rows],
        md.qsa_req_idx_per_token[:rows],
    )
    if accepted.is_contiguous() and all(t.is_contiguous() for t in meta):
        # The rows and the metadata at them in one launch.
        cont, meta = draft_continue_select(accepted, gen_seqs, p, meta)
    else:
        last = torch.clamp(accepted[:gen_seqs].to(torch.int64) - 1, min=0, max=p - 1)
        cont = torch.arange(gen_seqs, dtype=torch.int64, device=last.device) * p + last
        meta = tuple(torch.index_select(t, 0, cont) for t in meta)
    return _DraftContinue(cont, *meta, counts)


def _draft_rows(cont: torch.Tensor, operands: dict) -> dict:
    """`operands` at rows `cont`: the 2-D ones with unit column stride in one launch
    (`draft_rows_gather`), any other one with `index_select`. A `[T, h, d]` index query is
    gathered flat and comes back `[rows, h, d]`."""
    out, flat = {}, {}
    for name, t in operands.items():
        f = t.reshape(t.shape[0], -1) if name == "q_index" else t
        if f.dim() == 2 and f.stride(1) == 1:
            flat[name] = f
        else:
            out[name] = torch.index_select(t, 0, cont)
    if flat:
        out.update(zip(flat, draft_rows_gather(cont, tuple(flat.values()))))
    if "q_index" in flat:
        q = operands["q_index"]
        out["q_index"] = out["q_index"].view(cont.shape[0], *q.shape[1:])
    return out


def _hc_first(core, normed, first, with_inject):
    """A hyper-connection mixer's first half: the first projection over
    `[down; block_inject; 0]` (fp32 logits), then the SiLU and injection activations.
    Returns `(low, inject)`; `inject` is None without `with_inject`. Up to
    `_HC_TC_FIRST_MAX_ROWS` rows (a per-capture constant) it is one tcgen05 launch; larger
    steps run cuBLAS's fp32 logits and the activation kernel. With `core.hc_tiles`
    (`DP_HC_TILES_ENV`) the launch takes the 64-row tile or the rows-on-N kernel at their rows
    (bitwise), and with `core.hc_first_x2` (`DP_HC_FIRST_X2_ENV`, class S) two 64-row tiles in
    place of the rows-on-N kernel."""
    rows = normed.shape[0]
    if rows <= _HC_TC_FIRST_MAX_ROWS:
        return qwen4_exp_hc_decode_tc_first(
            normed,
            first,
            core.hc_lowrank,
            core.hc_count,
            with_inject,
            **_hc_first_form(core, rows, with_inject),
        )
    return qwen4_exp_hc_lowrank_act(
        cublas_mm(normed, first[:, : core.hc_first_cols], None, torch.float32),
        core.hc_lowrank,
        core.hc_count,
        with_inject,
        normed.dtype,
    )


def _hc_mixer(core, normed, first, up, with_inject, release: bool = False):
    """One hyper-connection mixer from the site's normed stream: `_hc_first`, then the up
    projection with the sigmoid gate and the stream mean in its epilogue. Returns
    `(mixed, inject)`; `inject` is None without `with_inject` (the stack-final mixer).
    `release` (`DP_INPROJ_RELEASE_ENV`, trunk site A only): the up kernel releases its
    dependents before its own wait; the caller answers for the launch after it (one that reads
    `mixed` after its own wait). With `core.hc_tiles` the up kernel runs on 64-row tiles at
    `_HC_TILE64_ROWS` rows, and at `_HC_UP2_ROWS` rows unless it releases (bitwise)."""
    low, inject = _hc_first(core, normed, first, with_inject)
    tm = _hc_up_tile(core, normed.shape[0], release)
    if tm == 128 and not release:
        return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count), inject
    mixed = qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count, release_early=release, tm=tm)
    return mixed, inject


def _hc_moe_mixer(core, normed, L: _LayerWeights):
    """The MoE-site mixer (`_hc_mixer` with injection). Under attention TP its up
    projection's epilogue also writes the routed experts' NVFP4 input -- linear block
    scales at the layer's FC1 input global scale -- which the expert window reads as is.
    Under attention DP the window runs on the group's gathered bf16 rows and quantizes
    them itself (`_routed_experts`). Returns `(mixed, inject, fp4)`, `fp4` the
    `(xq, xsf)` pair or None."""
    if core.dp_size > 1:
        mixed, inject = _hc_mixer(core, normed, L.mhc_first, L.mhc_up, True)
        return mixed, inject, None
    low, inject = _hc_first(core, normed, L.mhc_first, True)
    mixed, xq, xsf = qwen4_exp_hc_up_gate_mean_fp4(
        low, L.mhc_up, normed, core.hc_count, L.moe_gscale
    )
    return mixed, inject, (xq, xsf)


def _hc_mixer_rstd(core, hc, rstd, norm, norm_w1, first, up, with_inject):
    """`_hc_mixer` from the site's residual `hc` and its `[T, C]` rstd
    (`qwen4_exp_hc_inject_rstd`) instead of a stored normed stream: the first projection forms
    `bf16(hc * rstd * (1 + norm))` on its A tiles, the up projection's epilogue
    `hc * rstd * norm_w1` in fp32. Returns `(mixed, inject)`."""
    low, inject = qwen4_exp_hc_first_rescale_tc(
        hc, rstd, norm, first, core.hc_count, core.hc_lowrank, with_inject
    )
    return qwen4_exp_hc_up_gate_mean_rstd(low, up, hc, rstd, norm_w1, core.hc_count), inject


def _hc_moe_mixer_rstd(core, hc, rstd, L: _LayerWeights):
    """`_hc_moe_mixer` on the rstd form (see `_hc_mixer_rstd`): under attention TP the up
    projection's epilogue also writes the routed experts' NVFP4 input; under attention DP the
    window quantizes the gathered rows itself. Returns `(mixed, inject, fp4)`."""
    if core.dp_size > 1:
        mixed, inject = _hc_mixer_rstd(
            core, hc, rstd, L.mhc_norm, L.mhc_norm_w1, L.mhc_first, L.mhc_up, True
        )
        return mixed, inject, None
    low, inject = qwen4_exp_hc_first_rescale_tc(
        hc, rstd, L.mhc_norm, L.mhc_first, core.hc_count, core.hc_lowrank, True
    )
    mixed, xq, xsf = qwen4_exp_hc_up_gate_mean_rstd_fp4(
        low, L.mhc_up, hc, rstd, L.mhc_norm_w1, core.hc_count, L.moe_gscale
    )
    return mixed, inject, (xq, xsf)


def _decode_router_mid(core, L: _LayerWeights, captured: bool):
    """A decode step's router projection for layer `L`: `mid_rows_mm` over its view in the
    dense FP8 mode's rows (`_MID_MM_ROWS["router"]`), else at tp1 the step's projection entry
    (`core.decode_mm`) and under attention TP `_decode_router`.

    Under attention DP every rank routes all the rows its group gathers, so every rank must
    compute their logits with one kernel: two kernels can differ in the last bit and flip a
    near-tie in the top-k, and the ranks would then disagree on which window serves a slot. A
    `captured` step is a decode step on every rank (the engine captures only when every rank
    can) and takes `_decode_router`; an eager one can meet another rank's prefill step and takes
    the prefill router, `cublas_mm`."""
    if core.dp_size > 1:
        if not captured:
            return cublas_mm
        return getattr(core, "_router_front", None) or _decode_router
    fallback = core.decode_mm if core.one_rank else _decode_router
    if L.router_m is None:
        return fallback
    return lambda x, router: _dense_mm(x, router, None, fallback, "router", mid=L.router_m)


def _decode_router(x, router):
    """A decode step's router projection: up to `_ROUTER_LOW_M_MAX_ROWS` rows FlashInfer's
    low-M kernels with PDL (`low_m_bf16_mm`), more rows cuBLAS. Decode only: those
    kernels compile on the first call per row count, which the engine's warmup forward at
    each captured batch makes before capture; a prefill step's row count is any, so prefill
    keeps `cublas_mm`. Both allocate the logits fresh and dense, as the runner's routing
    reads them from `data_ptr()`."""
    if x.shape[0] <= _ROUTER_LOW_M_MAX_ROWS:
        return low_m_bf16_mm(x, router, _DECODE_PROJ_PDL)
    return cublas_mm(x, router)


class _RouteFront:
    """What `DP_ROUTE_FRONT_ENV` built (`_route_front_build`): per gathered row count its pinned
    tactic and, a layer, the int32 table buffer and its six views, made at build so a capture
    neither allocates nor fills one; `layers` maps a layer's FC1 weight to its index."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.tactics: dict = {}
        # Per row count whose pinned tactic the runner does not list (dep8): the listed tactic
        # the build check holds its window output to.
        self.refs: dict = {}
        self.views: dict = {}
        self.flats: dict = {}
        self.layers: dict = {}
        # `TEP_MOE_FRONT_ENV` (`_tep_decode_front_build`): the row counts whose router GEMM runs in
        # the front's launch, and each layer's staging buffer.
        self.decode: frozenset = frozenset()
        self.stages: list = []
        # `DP_ROUTE_FAST2_ENV`: the row counts compiled with two tokens a warp on the fast path,
        # and the form each takes (`_ROUTE_FRONT_FAST2_FORMS`).
        self.fast2: dict = {}
        # `DP_ROUTE_FRONT_EARLY_ENV`: the row counts compiled with the release after the wait.
        self.early: set = set()


def _route_front_tactic(runner, core, rows: int) -> tuple | None:
    """The pinned tactic of `rows` gathered rows: `_ROUTE_FRONT_TACTICS`'s where the runner lists
    it for this window, else `_ROUTE_FRONT_TACTICS_BASE`'s, else None."""
    valid = [
        list(t)
        for t in runner.get_valid_configs(
            core.topk, core.hidden, core.expert_inter, core.local_experts, rows
        )
    ]
    for table in (_ROUTE_FRONT_TACTICS, _ROUTE_FRONT_TACTICS_BASE):
        tactic = next(t for below, t in table if rows < below)
        if list(tactic) in valid:
            return tactic
    return None


def _route_front_w64_tactic(runner, core, rows: int) -> tuple:
    """dep8's `(tactic, ref)` for `rows` gathered rows over its 64-expert window:
    `_ROUTE_FRONT_TACTICS_W64`'s tactic, and None where the runner lists it, else a tactic it lists
    (of the same tile where it lists one) that the build holds the pinned one's window output to;
    `(None, None)` where the runner lists nothing."""
    valid = sorted(
        tuple(int(v) for v in t)
        for t in runner.get_valid_configs(
            core.topk, core.hidden, core.expert_inter, core.local_experts, rows
        )
    )
    table = _ROUTE_FRONT_TACTICS_W64
    if rows > _ROUTE_FRONT_MAX_ROWS and "route" in _dp_hicap(core):
        table = table + _DP_HICAP_TACTICS_W64
    tactic = next((t for below, t in table if rows < below), None)
    if tactic is None or not valid:
        return None, None
    if tactic in valid:
        return tactic, None
    same_tile = [t for t in valid if t[0] == tactic[0]]
    return tactic, (same_tile or valid)[0]


def _tep_route_front_tactic(runner, core, rows: int) -> tuple | None:
    """A tep4 rank's pinned tactic for `rows` rows: `DECODE_TACTICS`' pin for the row count's
    bucket (measured on the whole layer, where `fp4_block_scale_moe_renorm_routed` runs it; its
    window calls take the AutoTuner's) where the runner lists it for the window, else None."""
    tactic = _MOE_DECODE_TACTICS.get(last_positive_power_of_2(rows))
    valid = [
        list(t)
        for t in runner.get_valid_configs(
            core.topk, core.hidden, core.expert_inter, core.local_experts, rows
        )
    ]
    return tactic if tactic is not None and list(tactic) in valid else None


def _route_front_run(
    core,
    st: _RouteFront,
    x,
    L: _LayerWeights,
    layer: int,
    logits,
    fp4=None,
    finalize=True,
    routed=None,
):
    """The route front and the pre-routed runner over rows `x` (a built row count) and their
    router logits: `(out, xq, xsf, ids, wts)`. The runner runs this rank's window numbered from
    0 on `fp4` (the MoE site's NVFP4 codes, the same bits as the front's) or the front's own
    codes; `out` is its finalized output, or with `finalize` False its FC2 rows in the tables'
    permuted order (the slot -> row table is `st.views[rows][layer][0]`, the weights `wts`).
    `logits` None (`_tep_front_logits`): the decode front forms them in its own launch. `routed`
    (`DP_MOE_PACK_GATHER_ENV`, `x` None): the gathered rows' `(xq, xsf, ids, wts)` the packed
    gather and the routing-only front formed; the runner alone runs."""
    rows = x.shape[0] if routed is None else routed[2].shape[0]
    tactic = st.tactics[rows]
    tile = int(tactic[0])
    if routed is not None:
        xq, xsf, ids, wts = routed
    elif logits is None:
        _, xq, xsf, ids, wts = _decode_front.moe_decode_front(
            x,
            L.router.t(),
            L.moe_gscale,
            st.flats[rows][layer],
            st.stages[layer],
            tile,
            core.expert_offset,
            core.local_experts,
        )
    else:
        xq, xsf, ids, wts = moe_route_front(
            logits,
            x,
            L.moe_gscale,
            st.flats[rows][layer],
            tile,
            core.expert_offset,
            core.local_experts,
            st.fast2.get(rows, 0),
            rows in st.early,
        )
    xq_in, xsf_in = (xq, xsf) if fp4 is None else fp4
    out = st.runner.run_moe_prerouted(
        None,
        None,
        xq_in,
        # The runner reads the linear block scales flat.
        torch.reshape(xsf_in, [-1]).view(torch.float8_e4m3fn),
        L.fc1_w,
        L.fc1_s,
        None,
        None,
        None,
        None,
        L.fc2_w,
        L.fc2_s,
        None,
        L.out1_scale,
        L.out1_gate_scale,
        L.out2_scale,
        core.local_experts,
        core.topk,
        None,
        None,
        core.expert_inter,
        0,
        core.local_experts,
        None,
        ROUTING_RENORMALIZE,
        finalize,
        list(tactic),
        wts,
        ids,
        None,
        *st.views[rows][layer],
        tile,
    )[0]
    return out, xq, xsf, ids, wts


def _route_front_check(core, st: _RouteFront, L: _LayerWeights, layer: int, rows: int, gen):
    """One layer at one row count, eagerly, against today's chain at the same tactic (router
    logits -> `fp4_quantize` -> the runner's own routing over this rank's window), or at the
    listed reference tactic where the pinned one is not listed (`st.refs`), and the tables
    against their meaning: "" when the window output, the NVFP4 codes and scales, the experts and
    the weights are bitwise equal and the tables hold, else what differs."""
    x = torch.randn([rows, core.hidden], generator=gen, device=L.router.device).to(torch.bfloat16)
    logits = cublas_mm(x, L.router)
    xq0, xsf0 = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, False)
    ref = st.runner.run_moe(
        logits,
        None,
        xq0,
        torch.reshape(xsf0, [-1]).view(torch.float8_e4m3fn),
        L.fc1_w,
        L.fc1_s,
        None,
        None,
        None,
        None,
        L.fc2_w,
        L.fc2_s,
        None,
        L.out1_scale,
        L.out1_gate_scale,
        L.out2_scale,
        core.num_experts,
        core.topk,
        None,
        None,
        core.expert_inter,
        core.expert_offset,
        core.local_experts,
        None,
        ROUTING_RENORMALIZE,
        True,
        list(st.refs.get(rows) or st.tactics[rows]),
        None,
        None,
        None,
    )[0]
    ids0, wts0 = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=torch.bfloat16)
    out, xq, xsf, ids, wts = _route_front_run(core, st, x, L, layer, logits)
    bad = []
    for name, got, want in (
        ("window output", out, ref),
        ("NVFP4 codes", xq, xq0),
        ("NVFP4 block scales", xsf, xsf0),
        ("selected experts", ids.to(torch.int64), ids0.to(torch.int64)),
        ("routing weights", wts, wts0),
    ):
        if got.numel() != want.numel() or not torch.equal(
            got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8)
        ):
            bad.append(name)
    tables = _route_front_check_tables(
        st.views[rows][layer], ids, core.expert_offset, core.local_experts, int(st.tactics[rows][0])
    )
    if tables:
        bad.append(f"tables ({tables})")
    return ", ".join(bad)


def _route_front_fast2(core, rows: int) -> int:
    """`DP_ROUTE_FAST2_ENV` for a route front of `rows` gathered rows: the fast path's form
    (`_ROUTE_FRONT_FAST2_FORMS`: 2 at dep4's 128-expert window, 1 at dep8's 64) at
    `_ROUTE_FRONT_FAST2_ROWS` under attention DP with the switch on (the default), 2 at both with
    the switch at `2`, else 0. Another value raises."""
    raw = os.environ.get(DP_ROUTE_FAST2_ENV, "").strip()
    forced = raw == "2"
    on = forced or _switch(DP_ROUTE_FAST2_ENV, default=True)
    lo, hi = _ROUTE_FRONT_FAST2_ROWS
    hi = max(hi, _dp_hicap_route_rows(core)) if core.dp_size > 1 else hi
    form = _ROUTE_FRONT_FAST2_FORMS.get(core.local_experts, 0)
    if not (on and core.dp_size > 1 and form and lo <= rows <= hi):
        return 0
    return 2 if forced else form


def _route_front_early_kw(st, rows: int) -> dict:
    """`early=True` for a row count the route front releases early (`_RouteFront.early`), for the
    routing-only fronts of the packed gather, else nothing."""
    return {"early": True} if rows in getattr(st, "early", ()) else {}


def _route_front_early(core, rows: int) -> bool:
    """`DP_ROUTE_FRONT_EARLY_ENV` for a route front of `rows` gathered rows: under attention DP past
    the lean form's rows, with PDL and the switch on (the default)."""
    return (
        core.dp_size > 1
        and rows > _ROUTE_FRONT_LEAN_MAX_ROWS
        and bool(get_env_enable_pdl())
        and _switch(DP_ROUTE_FRONT_EARLY_ENV, default=True)
    )


def _route_front_build(core, layers, device) -> _RouteFront | None:
    """`DP_ROUTE_FRONT_ENV` at an attention-DP rank, `TEP_FAST_DECODE_ENV` at a tep4 rank
    (`core.tep_fast`), read once at build (`post_load_weights`, `layers` the decode target's):
    None where it is off or cannot take effect (another layout, not the trtllm-gen trunk MoE over
    CUDA weights, another routing geometry, no CUDA-graph batch size in the row range) or where
    the op is not certified (`moe_route_front.ARCHS`, SM103, and its windows: dep4's 128 experts
    over the gathered rows, `_ROUTE_FRONT_TACTICS`; dep8's 64, `_route_front_w64_tactic`; tep4's
    own rows up to `_TEP_ROUTE_FRONT_MAX_ROWS`, `_tep_route_front_tactic`). For every row count a
    graph batch size gives it compiles the route front at the pinned tactic's tile, makes each
    layer's table buffer (pre-filled with valid row indices: the kernel writes only the live
    entries), and checks every layer once at a row count that rotates over the built ones
    (`_route_front_check`). A row count pinned to a tactic the runner does not list whose check
    fails keeps the runner's routing (one warning); any other failed check leaves the whole
    switch off, with one warning."""
    dp = core.dp_size > 1
    if dp and not _switch(DP_ROUTE_FRONT_ENV, default=True):
        return None
    if not (
        (dp or core.tep_fast)
        and not core.one_rank
        and core.trunk_moe_backend == "TRTLLM"
        and device.type == "cuda"
        and (core.num_experts, core.topk) == (_ROUTE_FRONT_EXPERTS, _ROUTE_FRONT_TOP_K)
        and core.local_experts
        == (_ROUTE_FRONT_WINDOWS.get(core.dp_size) if dp else _ROUTE_FRONT_WINDOW)
        and torch.cuda.get_device_capability(device) == archs.SM_103
    ):
        return None
    # Under attention DP the group's gathered rows, at a tep4 rank its own.
    per_batch = core.dp_size * core.max_verify_rows
    lo, hi = (
        (_ROUTE_FRONT_MIN_ROWS, _dp_hicap_route_rows(core))
        if dp
        else (1, _TEP_ROUTE_FRONT_MAX_ROWS)
    )
    sizes = getattr(core.model_config, "cuda_graph_batch_sizes", None) or ()
    rows_list = sorted({per_batch * int(b) for b in sizes if lo <= per_batch * int(b) <= hi})
    if not rows_list:
        return None

    def pin(runner, rows: int) -> tuple:
        """`(tactic, listed reference or None)` of `rows`."""
        if dp and core.local_experts == _ROUTE_FRONT_WINDOWS[8]:
            return _route_front_w64_tactic(runner, core, rows)
        return (_route_front_tactic if dp else _tep_route_front_tactic)(runner, core, rows), None

    st = _RouteFront(_moe_runner())
    with torch.cuda.device(device):
        for rows in rows_list:
            tactic, ref = pin(st.runner, rows)
            # Past `MAX_ROWS` (`DP_HICAP_ENV`) the launch exists on the fast path only.
            wide = rows > _ROUTE_FRONT_MAX_ROWS
            form = _route_front_fast2(core, rows)
            if (
                tactic is None
                or (wide and not form)
                or not _route_front_compile(
                    rows,
                    core.hidden,
                    core.hidden,
                    core.local_experts,
                    int(tactic[0]),
                    form if wide else 0,
                )
            ):
                continue
            if form and _route_front_compile(
                rows, core.hidden, core.hidden, core.local_experts, int(tactic[0]), form
            ):
                st.fast2[rows] = form
            if _route_front_early(core, rows) and _route_front_compile(
                rows,
                core.hidden,
                core.hidden,
                core.local_experts,
                int(tactic[0]),
                st.fast2.get(rows, 0),
                early=True,
            ):
                st.early.add(rows)
            n = sum(table_sizes(rows, core.topk, core.local_experts, int(tactic[0])))
            fill = (torch.arange(n, device=device) % rows).to(torch.int32)
            st.tactics[rows] = tactic
            if ref is not None:
                st.refs[rows] = ref
            st.flats[rows] = [fill.clone() for _ in range(core.num_layers)]
            st.views[rows] = [
                table_views(flat, rows, core.topk, core.local_experts, int(tactic[0]))
                for flat in st.flats[rows]
            ]
        for i, L in enumerate(layers):
            st.layers[L.fc1_w.data_ptr()] = i
        built = sorted(st.tactics)
        gen = torch.Generator(device=device).manual_seed(20261119)
        for j in range(max(len(layers), len(built)) if built else 0):
            rows, i = built[j % len(built)], j % len(layers)
            if rows not in st.tactics:
                continue
            bad = _route_front_check(core, st, layers[i], i, rows, gen)
            if bad and rows > _ROUTE_FRONT_MAX_ROWS:
                # `DP_HICAP_ENV`: no two-token sort to fall back to past `MAX_ROWS`.
                logger.warning(
                    f"{DP_HICAP_ENV}: layer {i} at {rows} gathered rows differs from the runner's "
                    f"own routing ({bad}); that row count keeps the runner's routing"
                )
                for table in (st.tactics, st.refs, st.flats, st.views):
                    table.pop(rows, None)
                st.fast2.pop(rows, None)
                st.early.discard(rows)
                continue
            if bad and (rows in st.fast2 or rows in st.early):
                logger.warning(
                    f"{DP_ROUTE_FAST2_ENV} / {DP_ROUTE_FRONT_EARLY_ENV}: layer {i} at {rows} "
                    f"gathered rows (fast path form {st.fast2.get(rows, 0)}, early release "
                    f"{rows in st.early}) differs from the runner's own routing ({bad}); that row "
                    "count keeps the two-token sort and the end release"
                )
                st.fast2.pop(rows, None)
                st.early.discard(rows)
                bad = _route_front_check(core, st, layers[i], i, rows, gen)
            if bad and rows in st.refs:
                logger.warning(
                    f"{DP_ROUTE_FRONT_ENV}: layer {i} at {rows} gathered rows on the tactic "
                    f"{list(st.tactics[rows])}, which the runner does not list, differs from its "
                    f"listed {list(st.refs[rows])} ({bad}); that row count keeps the runner's "
                    "routing"
                )
                for table in (st.tactics, st.refs, st.flats, st.views):
                    table.pop(rows, None)
            elif bad:
                logger.warning(
                    f"{DP_ROUTE_FRONT_ENV if dp else TEP_FAST_DECODE_ENV}: layer {i} at {rows} "
                    f"rows differs from the runner's own routing ({bad}); every MoE block keeps it"
                )
                return None
        built = sorted(st.tactics)
        torch.cuda.synchronize(device)
    if not built:
        return None
    logger.info(
        f"MoE route front: rank {core.mapping.tp_rank} of {core.mapping.tp_size}, "
        f"{'gathered ' if dp else ''}rows {built}, tactics {[st.tactics[r] for r in built]}"
        + (
            f", two tokens a warp on the fast path at {sorted(st.fast2)} (forms "
            f"{[st.fast2[r] for r in sorted(st.fast2)]})"
            if st.fast2
            else ""
        )
        + (f", FC1 released after the wait at {sorted(st.early)}" if st.early else "")
    )
    return st


def _route_front_experts(
    core, st: _RouteFront, x, L: _LayerWeights, fp4=None, router_mm=None
) -> torch.Tensor | None:
    """`_routed_experts` of a captured generation-only step over the route front, or None where
    no route front was built for the row count (the caller's path). `fp4` is the MoE site's
    NVFP4 activation at a tep4 rank (the runner reads it), None under attention DP; `router_mm`
    the step's decode router (`_decode_router` by default, or the router front)."""
    rows = x.shape[0]
    layer = st.layers.get(L.fc1_w.data_ptr())
    if rows not in st.tactics or layer is None or x.stride(1) != 1 or x.stride(0) != core.hidden:
        return None
    logits = _tep_front_logits(st, x, L, router_mm)
    return _route_front_run(core, st, x, L, layer, logits, fp4)[0]


class _DpRouterFront:
    """What `DP_ROUTER_FRONT_ENV` built (`_dp_router_front_build`), and the decode router of a
    captured step (`_decode_router_mid`): `front(x, router)` runs `mid_rows_mm` over the layer's
    bf16 router (`mids`, by the router's pointer) at a gathered row count in `rows`, else
    `_decode_router`. The row count alone decides, so every rank of the group runs one kernel.
    `release_rows`: the gathered row counts whose Lamport gather releases its dependents."""

    def __init__(self, rows: frozenset, release_rows: frozenset, mids: dict) -> None:
        self.rows = rows
        self.release_rows = release_rows
        self.mids = mids

    def __call__(self, x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        if x.shape[0] in self.rows:
            return mid_rows_mm_op(x, self.mids[router.data_ptr()])
        return _decode_router(x, router)


def _dp_router_front_check(mw, rows: int, gen, device) -> int:
    """Logits of the bf16 router kernel at `rows` random rows off the fp32 product by more than
    one bf16 ulp (plus the K sum's rounding): their count."""
    x = torch.randn([rows, mw.k], generator=gen, device=device).to(torch.bfloat16)
    got = mid_rows_mm_op(x, mw).float()
    w = mw.w.float()
    ref = x.float() @ w.t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    tol = ulp + 2e-5 * (x.float().abs() @ w.abs().t())
    return int(((got - ref).abs() > tol).sum())


def _dp_router_front_build(core, layers, device) -> "_DpRouterFront | None":
    """`DP_ROUTER_FRONT_ENV`, read once at build (`post_load_weights`, after the route front).
    None with the switch unset, or with one warning where it cannot take effect: not a dep4 /
    dep8 rank on SM103, no Lamport transport, no graph row count in either range. Otherwise the
    releasing gather and the bf16 router of every graph row count in range are compiled now (no
    serving step or capture meets a compile) and each router row count is checked against the
    fp32 product of layer 0's router. Collective: the group engages together or not at all."""
    if not _switch(DP_ROUTER_FRONT_ENV):
        return None
    why = ""
    trip = core._lamport_trip
    if core.dp_size not in _DP_ROUTER_FRONT_RELEASE_ROWS or trip is None:
        why = (
            f"it serves a dep4 or dep8 rank on the Lamport transport ({DP_MOE_COMM_ENV}); this "
            f"rank: attention DP size {core.dp_size}, transport "
            f"{'lamport' if trip is not None else 'nccl'}"
        )
    elif torch.cuda.get_device_capability(device) != archs.SM_103:
        why = "it serves SM103 only"
    if why:
        logger.warning(f"{DP_ROUTER_FRONT_ENV} is set but ignored: {why}")
        return None
    per_batch = core.dp_size * core.max_verify_rows
    sizes = getattr(core.model_config, "cuda_graph_batch_sizes", None) or ()
    graph_rows = {per_batch * int(b) for b in sizes}
    lo, hi = _DP_ROUTER_FRONT_RELEASE_ROWS[core.dp_size]
    release = frozenset(r for r in graph_rows if lo <= r <= hi)
    lo, hi = _DP_ROUTER_FRONT_ROWS
    rows = frozenset(r for r in graph_rows if lo <= r <= min(hi, _MID_ROWS_MAX))
    mids, error = {}, ""
    try:
        with torch.cuda.device(device):
            if release:
                trip.releasing_gather_fn()
            for L in layers:
                mids[L.router.data_ptr()] = MidWeight(L.router.t())
            if rows:
                mw = mids[layers[0].router.data_ptr()]
                mid_rows_precompile(mw, min(rows), max(rows))
                gen = torch.Generator(device=device).manual_seed(20261125)
                for r in sorted(rows):
                    off = _dp_router_front_check(mw, r, gen, device)
                    if off:
                        error = (
                            f"the bf16 router at {r} rows is off the fp32 product on {off} logits"
                        )
                        break
            torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001 -- every rank decides together below
        error = f"{type(e).__name__}: {e}"
    from tensorrt_llm._torch.distributed.communicator import Distributed

    errors = [e for e in Distributed.get(core.mapping).tp_allgather(error) if e]
    if errors or not (rows or release):
        logger.warning(
            f"{DP_ROUTER_FRONT_ENV} is set but not engaged at rank {core.dp_rank}: "
            f"{errors[0] if errors else 'no CUDA-graph row count in either range'}"
        )
        return None
    logger.info(
        f"DP router front: rank {core.dp_rank} of {core.dp_size}, bf16 router at gathered rows "
        f"{sorted(rows)}, releasing gather at {sorted(release)}"
    )
    return _DpRouterFront(rows, release, mids)


def _tep_front_logits(st: _RouteFront, x, L: _LayerWeights, router_mm=None):
    """The router logits a route-front block enters with: None at a row count the decode front
    serves (`TEP_MOE_FRONT_ENV`, tep4: the front's launch forms them), else the step's decode
    router's (`router_mm`: `_decode_router` by default, or the attention-DP router front)."""
    return None if x.shape[0] in st.decode else (router_mm or _decode_router)(x, L.router)


def _tep_decode_front_check(core, st: _RouteFront, L: _LayerWeights, rows: int, gen) -> tuple:
    """The decode front at `rows` rows over one layer, eagerly, on scratch tables: `(what
    differs, logits that differ from _decode_router's)`. The first is "" when its codes, scales,
    experts and weights are bitwise the route front's over its own logits, its tables hold their
    meaning for those experts (`_route_front_check_tables`: the kernel may order the rows inside
    an expert's tile differently), the staging buffer is back at zero and the logits are within
    the op's bound of the exact GEMM (`moe_decode_front__decode.logits_off_bound`: two bf16 steps
    plus an fp32 sum's worst-case reorder error); the second is reported, not required (the
    reorder this switch makes). `flat_b` is only the route front's scratch tables."""
    device = L.router.device
    tile = int(st.tactics[rows][0])
    x = torch.randn([rows, core.hidden], generator=gen, device=device).to(torch.bfloat16)
    n = sum(table_sizes(rows, core.topk, core.local_experts, tile))
    flat_a = (torch.arange(n, device=device) % rows).to(torch.int32)
    flat_b = flat_a.clone()
    stage = _decode_front.staging_buffer(device)
    logits, xq, xsf, ids, wts = _decode_front.moe_decode_front(
        x, L.router.t(), L.moe_gscale, flat_a, stage, tile, core.expert_offset, core.local_experts
    )
    xq0, xsf0, ids0, wts0 = moe_route_front(
        logits, x, L.moe_gscale, flat_b, tile, core.expert_offset, core.local_experts
    )
    stock = _decode_router(x, L.router)
    torch.cuda.synchronize(device)
    bad = []
    for name, got, want in (
        ("NVFP4 codes", xq, xq0),
        ("NVFP4 block scales", xsf, xsf0),
        ("selected experts", ids, ids0),
        ("routing weights", wts, wts0),
    ):
        if not torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8)):
            bad.append(name)
    views = table_views(flat_a, rows, core.topk, core.local_experts, tile)
    tables = _route_front_check_tables(views, ids, core.expert_offset, core.local_experts, tile)
    if tables:
        bad.append(f"tables ({tables})")
    if bool(stage.any()):
        bad.append("the staging buffer is not back at zero")
    off = _decode_front.logits_off_bound(logits, x, L.router.t())
    if bool(off.any()):
        bad.append(f"{int(off.sum())} logits outside the op's bound of the exact GEMM")
    return ", ".join(bad), int((logits != stock).sum())


def _tep_decode_front_build(core, st: _RouteFront | None, layers, device) -> None:
    """`TEP_MOE_FRONT_ENV` at build, after the route front: fills `st.decode` / `st.stages`.
    Where the switch is set at a tep4 rank with its fast forms (host values the group shares,
    so the ranks decline together) the row counts are agreed over the group, a collective:
    each rank compiles the route front's row counts up to `moe_decode_front.MAX_ROWS` and checks
    every layer once at a row count that rotates over them (`_tep_decode_front_check`, as the
    route front's build does), and only the row counts that pass on EVERY rank are served (one
    `core.tp_sum` of 0 / 1 flags), since a block that forms its logits one way on one rank and
    the other way on another could serve a (row, slot) on no window or on two. A compile or
    check that raises declines its row count on this rank and the rank still joins the sum: a
    rank that left the build here would leave the others waiting in the collective."""
    if not _switch(TEP_MOE_FRONT_ENV):
        return
    if not core.tep_fast or device.type != "cuda":
        logger.warning(
            f"{TEP_MOE_FRONT_ENV} is set but ignored: it needs a tep4 rank with "
            f"{TEP_FAST_DECODE_ENV} on (its route front)"
        )
        return
    cap = _decode_front.MAX_ROWS
    found, mine = {}, []
    if st is not None and layers:
        gen = torch.Generator(device=device).manual_seed(20261125)
        with torch.cuda.device(device):
            built = []
            for rows in sorted(r for r in st.tactics if r <= cap):
                try:
                    ok = _decode_front.compile_front(
                        rows, core.hidden, core.hidden, core.local_experts, int(st.tactics[rows][0])
                    )
                except Exception as e:  # noqa: BLE001 - declined here, agreed over the group below
                    found[rows] = f"compile raised {type(e).__name__}: {e}"
                    continue
                if not ok:
                    found[rows] = "not compiled"
                    continue
                built.append(rows)
            moved = dict.fromkeys(built, 0)
            for j in range(max(len(layers), len(built)) if built else 0):
                rows, i = built[j % len(built)], j % len(layers)
                if rows in found:
                    continue
                try:
                    bad, n = _tep_decode_front_check(core, st, layers[i], rows, gen)
                except Exception as e:  # noqa: BLE001 - declined here, agreed over the group below
                    bad, n = f"check raised {type(e).__name__}: {e}", 0
                if bad:
                    found[rows] = f"layer {i}: {bad}"
                moved[rows] += n
            for rows in built:
                if rows not in found:
                    found[rows] = f"{moved[rows]} logits differ from the stock router's"
                    mine.append(rows)
    flags = torch.zeros([cap, core.hidden], dtype=torch.bfloat16, device=device)
    for r in mine:
        flags[0, r - 1] = 1.0
    total = core.tp_sum(flags)[0, :cap].float().cpu()
    agreed = frozenset(r for r in range(1, cap + 1) if int(total[r - 1]) == _TEP_RANKS)
    if not agreed:
        logger.warning(
            f"{TEP_MOE_FRONT_ENV} is set but serves no row count on every rank of the group "
            f"(this rank: {found}); every block keeps the stock router"
        )
        return
    st.stages = [_decode_front.staging_buffer(device) for _ in layers]
    st.decode = agreed
    logger.info(
        f"MoE decode front ({TEP_MOE_FRONT_ENV}=1): rank {core.mapping.tp_rank}, rows "
        f"{sorted(agreed)}, build check {found}"
    )


class _TepFinar:
    """What `TEP_FINAR_ENV` built (`_tep_finar_build`): the rank's workspace and launch (`op`),
    the row counts it serves, the trunk MoE launches a step, and the layer of the last launch
    (host side, to hold the slot sequence)."""

    def __init__(self, op, rows: tuple, layers: int, shared=None) -> None:
        self.op = op
        self.rows = rows
        self.layers = layers
        self.prev: int | None = None
        #: `TEP_FINAR_SHARED_ENV`: the `_TepFinarShared` the build made, None without it.
        self.shared = shared


class _TepFinarShared:
    """What `TEP_FINAR_SHARED_ENV` built: the row counts that take a shared-expert quarter
    (`rows`), each MoE layer's quarter as GEMM operands keyed by its `sh_gug` storage
    (`quarters`: `(gate|up + gate-logit rows .t(), down columns .t())`), the quarter's width of the
    intermediate (`inter_q`), and the zero shared expert and gate the block hands the next site."""

    def __init__(self, rows: tuple, quarters: dict, inter_q: int, zero_shared, zero_gate) -> None:
        self.rows = rows
        self.quarters = quarters
        self.inter_q = inter_q
        self.zero_shared = zero_shared
        self.zero_gate = zero_gate


def _tep_finar_quarter_weights(core, L, rank: int) -> tuple:
    """One MoE layer's quarter `rank` of its shared expert, made once at build: rows
    `[r q, (r + 1) q)` of the gate and of the up halves of `[gate | up | gate logit | pad]`, the
    gate-logit row, zero-padded to a multiple of 8 rows; and columns `[r q, (r + 1) q)` of the down
    projection. As the `.t()` views `cublas_mm` / `low_m_bf16_mm` read."""
    q, inter = core.shared_inter // _TEP_FINAR_QUARTERS, core.shared_inter
    gu = L.sh_gug.t()  # [gate | up | gate logit | pad, hidden] row-major
    dn = L.sh_dn.t()  # [hidden, inter] row-major
    rows = torch.cat(
        [
            gu[rank * q : (rank + 1) * q],
            gu[inter + rank * q : inter + (rank + 1) * q],
            gu[2 * inter : 2 * inter + 1],
        ]
    )
    width = -(-(2 * q + 1) // 8) * 8
    gu_q = torch.zeros([width, gu.shape[1]], dtype=gu.dtype, device=gu.device)
    gu_q[: 2 * q + 1] = rows
    return gu_q.t(), dn[:, rank * q : (rank + 1) * q].contiguous().t()


def _tep_finar_shared_build(core, rows: tuple, device) -> "_TepFinarShared | None":
    """`TEP_FINAR_SHARED_ENV` at the fused finalize all-reduce's build (`rows` the row counts it
    serves on every rank): None where it is off or cannot apply (one warning, alike on every
    rank), else every MoE layer's quarter and the zero hand-off tensors."""
    if not _switch(TEP_FINAR_SHARED_ENV):
        return None
    q = core.shared_inter // _TEP_FINAR_QUARTERS
    served = tuple(r for r in _TEP_FINAR_SHARED_ROWS if r in rows)
    why = None
    if q * _TEP_FINAR_QUARTERS != core.shared_inter or q % 8:
        why = f"shared intermediate {core.shared_inter} is not four quarters of a multiple of 8"
    elif not served:
        why = f"the fused all-reduce serves none of rows {list(_TEP_FINAR_SHARED_ROWS)}"
    if why is not None:
        logger.warning(f"{TEP_FINAR_SHARED_ENV} is set but ignored: {why}")
        return None
    rank = core.mapping.tp_rank
    quarters = {
        L.sh_gug.data_ptr(): _tep_finar_quarter_weights(core, L, rank) for L in core._decode._layers
    }
    width = next(iter(quarters.values()))[0].shape[1]
    zero_shared = torch.zeros([max(served), core.hidden], dtype=torch.bfloat16, device=device)
    # A column of a `[rows, width]` buffer, as the quarter GEMM hands its gate logit.
    zero_gate = torch.zeros([max(served), width], dtype=torch.bfloat16, device=device)
    zero_gate = zero_gate[:, 2 * q : 2 * q + 1]
    return _TepFinarShared(served, quarters, q, zero_shared, zero_gate)


def _tep_finar_takes_quarter(core, x) -> bool:
    """Whether a MoE block over `x` runs its shared-expert quarter (`TEP_FINAR_SHARED_ENV`): its
    row count takes it and the block takes the fused launch. Host values only, alike on every
    rank."""
    fin = getattr(core, "_tep_finar", None)
    shared = getattr(fin, "shared", None)
    return shared is not None and x.shape[0] in shared.rows


def _tep_finar_quarter(core, x, L) -> tuple:
    """This rank's shared-expert quarter over `x` (on the stream the caller set): `(quarter
    [rows, hidden] fp32, its gate logit [rows, 1] bf16 column view)`. Its gate|up takes the split-K
    rows of `TEP_PROJ_ENV` where the trunk shared expert does."""
    sh = core._tep_finar.shared
    gu_t, dn_t = sh.quarters[L.sh_gug.data_ptr()]
    mm = _tep_wide_mm if getattr(core, "tep_proj", False) else cublas_mm
    gu = mm(x, gu_t)
    part = cublas_mm(silu_and_mul_rows(gu, sh.inter_q), dn_t, None, torch.float32)
    return part, gu[:, 2 * sh.inter_q : 2 * sh.inter_q + 1]


def _tep_finar_selfcheck(core, op, device, rows: int, slot: int, shared: bool = False) -> tuple:
    """One eager launch at `rows` rows on synthetic window outputs (slots of other ranks'
    windows, a row with no local slot, -0.0 lanes, values over 2^-12..2^8) against the entry's
    reference over every rank's operands. `(ok, lanes that differ from the runner's finalize +
    core.tp_sum)`: the second is reported, not required (the engine's all-reduce strategy may sum
    in another order). With `shared`, the launch with a synthetic fp32 shared-expert quarter
    whose gate logits (0, 60, -200) have an exact sigmoid on both sides (0.5, 1, 0)."""
    import torch.distributed as dist

    world, rank = _TEP_FINAR.WORLD, core.mapping.tp_rank
    g = torch.Generator(device="cpu").manual_seed(9100 + 97 * rows + rank)
    owner = (torch.arange(rows).unsqueeze(1) * 3 + torch.arange(core.topk).unsqueeze(0)) % world
    owner[0] = (rank + 1) % world
    mine = owner == rank
    count = int(mine.sum())
    perm = torch.full([rows, core.topk], -1, dtype=torch.int32)
    perm[mine] = torch.randperm(count + 3, generator=g)[:count].to(torch.int32)
    scale = torch.exp2(torch.randint(-12, 9, [count + 3, 1], generator=g).float())
    fc2 = (torch.randn([count + 3, core.hidden], generator=g) * scale).to(torch.bfloat16)
    fc2[:, ::37] = -0.0
    wts = (torch.rand([rows, core.topk], generator=g) * 0.5 + 0.01).to(torch.bfloat16)
    fc2, wts, perm = fc2.to(device), wts.to(device), perm.to(device)
    mine_ops = (fc2, wts, perm)
    if shared:
        quarter = (torch.randn([rows, core.hidden], generator=g) * 0.25).to(device)
        gate = torch.zeros([rows, 8], dtype=torch.bfloat16)
        gate[:, 3] = torch.tensor([0.0, 60.0, -200.0])[torch.arange(rows) % 3].to(torch.bfloat16)
        mine_ops = mine_ops + (quarter, gate.to(device)[:, 3:4])
    got = op(*mine_ops[:3], rows, slot, *mine_ops[3:])
    everyone = [None] * world
    dist.all_gather_object(everyone, tuple(t.cpu() for t in mine_ops))
    want = tep_finalize_allreduce.reference([tuple(t.to(device) for t in ops) for ops in everyone])
    local = _tep_finar_local(fc2, wts, perm)
    if shared:
        local = (local.float() + torch.sigmoid(mine_ops[4].float()) * mine_ops[3]).to(local.dtype)
    stock = core.tp_sum(local)
    torch.cuda.synchronize(device)
    ok = tuple(got.shape) == (rows, core.hidden) and torch.equal(got, want)
    return ok, int((got != stock).sum())


def _tep_finar_local(fc2, wts, perm) -> torch.Tensor:
    """The runner's finalize of one rank's window, emulated exactly (fp32, slot order)."""
    acc = torch.zeros((perm.shape[0], fc2.shape[1]), dtype=torch.float32, device=fc2.device)
    for k in range(perm.shape[1]):
        idx = perm[:, k].long()
        rows = fc2[idx.clamp_min(0)].float()
        acc = torch.where((idx >= 0)[:, None], acc + wts[:, k : k + 1].float() * rows, acc)
    return acc.to(torch.bfloat16)


def _tep_finar_rows(core, device) -> tuple:
    """The row counts the fused launch may serve: those up to `tep_finalize_allreduce.MAX_ROWS`
    that the route front serves on EVERY rank of the group. Each rank builds and checks its route
    front over its own window, so one rank can lack a row count, or the whole front, that the
    others have; a block that took the fused launch on some ranks and `core.tp_sum` on another
    would leave both waiting. One `core.tp_sum` of 0 / 1 flags (exact: small integers) agrees on
    the intersection. Collective: every rank of the group calls it."""
    st = core._route_front
    mine = ()
    if st is not None and len(st.layers) % _TEP_FINAR.SLOTS == 0:
        mine = tuple(r for r in sorted(st.tactics) if r <= _TEP_FINAR.MAX_ROWS)
    flags = torch.zeros([_TEP_FINAR.MAX_ROWS, core.hidden], dtype=torch.bfloat16, device=device)
    for r in mine:
        flags[0, r - 1] = 1.0
    total = core.tp_sum(flags)[0, : _TEP_FINAR.MAX_ROWS].float().cpu()
    return tuple(
        r for r in range(1, _TEP_FINAR.MAX_ROWS + 1) if int(total[r - 1]) == _TEP_FINAR.WORLD
    )


#: Off by default (opt-in, bitwise). At a tep4 rank (`TEP_FAST_DECODE_ENV`) inside one NVLink
#: domain, every attention-TP sum (`core.tp_sum`: the trunk attention / GDN output projections,
#: the draft's, a MoE block's window sum where the fused finalize all-reduce does not serve it)
#: of a bf16 `[1..32, 2560]` partial runs on the fused finalize all-reduce's transport without
#: the finalize (`tep_finalize_allreduce.build_tp_sum`: one CuTe DSL launch, publish to the three
#: peers' symmetric slots, read theirs, fp32 sum in rank order, one rounding; its dependents
#: released right after its PDL wait) instead of the stock one-shot Lamport all-reduce. The sum is
#: the stock op's bit for bit except a lane whose four partials are all -0.0 (+0.0 here, the
#: transport's not-ready marker, as in the finalize launch), checked at build on every row count
#: it serves (a failure on any rank raises on every rank). Why (a 4-rank trace, 1 request): after
#: the last rank's producer ends, the stock launch needs 4.45 us (p50) and the finalize launch,
#: its finalize included, 2.30. Other sums (wider, other dtypes, more rows) keep the stock op. The
#: three slots rotate by call order. Read once, when the target is built; set at another layout
#: it warns and does nothing.
TEP_TPSUM_LAMPORT_ENV = "TRTLLM_QWEN4_EXP_TEP_TPSUM_LAMPORT"
#: Row counts the build check compares against the stock sum.
_TEP_TPSUM_CHECK_ROWS = (1, 4, 16, 20, 32)  # capacity: fixed: build-check coverage


class _TepTpSumOp:
    """`TEP_TPSUM_LAMPORT_ENV`'s `core.tp_sum`: the plain Lamport launch where it takes the
    operand, else the stock sum; the slot is the call count mod 3 (host, baked into a capture)."""

    def __init__(self, op, stock):
        self.op, self.stock, self.calls = op, stock, 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.op.takes(x):
            return self.stock(x)
        slot = self.calls % _TEP_FINAR.SLOTS
        self.calls += 1
        return self.op(x, slot)


def _tep_tpsum_bind(core, op) -> None:
    """`core.tp_sum` on the plain launch `op` (over the stock sum the core holds now), and the
    window sum rebuilt from it; a no-op when it is bound already."""
    if isinstance(core.tp_sum, _TepTpSumOp):
        return
    core.tp_sum = _TepTpSumOp(op, core.tp_sum)
    core.expert_sum = core._expert_sum_op()


def _tep_tpsum_build(core, device):
    """`TEP_TPSUM_LAMPORT_ENV` at build: None where it is off or cannot apply (one warning, on
    every rank alike); else the plain launch over the 4-rank torch group, checked bit for bit
    against the stock sum at `_TEP_TPSUM_CHECK_ROWS` on every rank, and `core.tp_sum` /
    `core.expert_sum` rebound to it."""
    if not _switch(TEP_TPSUM_LAMPORT_ENV):
        return None
    why = None
    if not core.tep_fast or core.one_rank:
        why = f"it needs a tep4 rank with {TEP_FAST_DECODE_ENV} on"
    elif core.mapping.is_multi_node():
        why = "the TP group spans nodes"
    elif core.hidden != _TEP_FINAR.HIDDEN:
        why = f"hidden {core.hidden}"
    if why is not None:
        logger.warning(f"{TEP_TPSUM_LAMPORT_ENV} is set but ignored: {why}")
        return None
    import torch.distributed as dist

    from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

    rank = core.mapping.tp_rank
    ensure_cutedsl_torch_distributed(_TEP_FINAR.WORLD)
    if not dist.is_initialized() or dist.get_world_size() != _TEP_FINAR.WORLD:
        raise RuntimeError(f"{TEP_TPSUM_LAMPORT_ENV} found no {_TEP_FINAR.WORLD}-rank torch group")
    if dist.get_rank() != rank:
        raise RuntimeError(f"{TEP_TPSUM_LAMPORT_ENV}: the torch WORLD rank order differs from TP's")
    stock = core.tp_sum
    with torch.cuda.device(device):
        op = tep_tp_sum_build(rank, get_env_enable_pdl(), early=True)
        gen = torch.Generator(device=device).manual_seed(4242 + rank)
        diffs = []
        for i, rows in enumerate(_TEP_TPSUM_CHECK_ROWS):
            x = torch.randn(rows, core.hidden, generator=gen, device=device).to(torch.bfloat16)
            # -0.0 lanes on this rank only: where every rank's partial of a lane is -0.0 the
            # stock sum keeps -0.0 and the launch +0.0 (its not-ready marker), as the finalize
            # launch's sum does.
            x[:, rank::37] = -0.0
            got = op(x, i % _TEP_FINAR.SLOTS)
            want = stock(x)
            diffs.append(int((got.view(torch.int16) != want.view(torch.int16)).sum()))
        ok = torch.tensor([int(not any(diffs))], dtype=torch.int32, device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not int(ok.item()):
        raise RuntimeError(
            f"{TEP_TPSUM_LAMPORT_ENV}: the plain Lamport sum differs from the stock sum on a rank "
            f"(this rank, rows {list(_TEP_TPSUM_CHECK_ROWS)}: lanes that differ {diffs}); unset "
            "the switch"
        )
    _tep_tpsum_bind(core, op)
    logger.info(
        f"TP sum on the plain Lamport launch ({TEP_TPSUM_LAMPORT_ENV}=1): rank {rank} of "
        f"{_TEP_FINAR.WORLD}, 1..{_TEP_FINAR.TP_SUM_MAX_ROWS} rows of {core.hidden}, "
        f"{op.workspace_bytes / (1 << 20):.2f} MiB, lanes that differ from the stock sum at the "
        f"build check {diffs}"
    )
    return op


def _tep_finar_build(core, device) -> "_TepFinar | None":
    """`TEP_FINAR_ENV` at build, after the route front: None where it is off or cannot apply
    (one warning, on every rank alike); else the workspace over the 4-rank torch group
    (bootstrapped under MPI as the attention-DP Lamport transport is), every row count the
    ranks agree on (`_tep_finar_rows`) checked eagerly on every rank, and the ranks' verdicts
    reduced: a failure on any rank raises on every rank, since a rank cannot keep the stock sum
    alone."""
    mode = _tep_finar_mode()
    if not mode:
        if _switch(TEP_FINAR_SHARED_ENV):
            logger.warning(f"{TEP_FINAR_SHARED_ENV} is set but ignored: it needs {TEP_FINAR_ENV}")
        return None
    # Host values every rank of the group shares: the ranks decline together.
    why = None
    if not core.tep_fast:
        why = f"it needs a tep4 rank with {TEP_FAST_DECODE_ENV} on"
    elif core.mapping.is_multi_node():
        why = "the TP group spans nodes"
    elif (core.hidden, core.topk) != (_TEP_FINAR.HIDDEN, _TEP_FINAR.TOP_K):
        why = f"hidden {core.hidden}, top-{core.topk}"
    if why is None:
        rows = _tep_finar_rows(core, device)
        if not rows:
            why = (
                f"the tep4 route front (SM103, CUDA graphs of 1-4 requests, MoE layers a "
                f"multiple of {_TEP_FINAR.SLOTS}) serves no row count up to "
                f"{_TEP_FINAR.MAX_ROWS} on every rank"
            )
    if why is not None:
        logger.warning(f"{TEP_FINAR_ENV} is set but ignored: {why}")
        return None
    st = core._route_front
    import torch.distributed as dist

    from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

    rank = core.mapping.tp_rank
    ensure_cutedsl_torch_distributed(_TEP_FINAR.WORLD)
    if not dist.is_initialized() or dist.get_world_size() != _TEP_FINAR.WORLD:
        raise RuntimeError(f"{TEP_FINAR_ENV} found no {_TEP_FINAR.WORLD}-rank torch group")
    if dist.get_rank() != rank:
        raise RuntimeError(f"{TEP_FINAR_ENV}: the torch WORLD rank order differs from the TP one")
    with torch.cuda.device(device):
        shared = _tep_finar_shared_build(core, rows, device)
        op = tep_finar_build(
            rank, get_env_enable_pdl(), early=mode == "early", with_shared=shared is not None
        )
        checks = [(r, False) for r in rows] + [(r, True) for r in (shared.rows if shared else ())]
        verdicts = [
            _tep_finar_selfcheck(core, op, device, r, i % _TEP_FINAR.SLOTS, quarter)
            for i, (r, quarter) in enumerate(checks)
        ]
        ok = torch.tensor([int(all(v[0] for v in verdicts))], dtype=torch.int32, device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not int(ok.item()):
        raise RuntimeError(
            f"{TEP_FINAR_ENV}: the fused finalize all-reduce failed its build check on a rank "
            f"(this rank, rows {list(rows)}: {[v[0] for v in verdicts]}); unset the switch"
        )
    logger.info(
        f"MoE finalize all-reduce ({TEP_FINAR_ENV}={mode}): rank {rank} of {_TEP_FINAR.WORLD}, "
        f"rows {list(rows)}, {op.workspace_bytes / (1 << 20):.2f} MiB, lanes that differ from "
        f"the runner's finalize + tp_sum at the build check {[v[1] for v in verdicts]}"
        + (
            f"; shared-expert quarter ({TEP_FINAR_SHARED_ENV}=1) {shared.inter_q} of "
            f"{core.shared_inter} at rows {list(shared.rows)}, {len(shared.quarters)} layers"
            if shared is not None
            else ""
        )
    )
    return _TepFinar(op, rows, len(st.layers), shared)


def _tep_finar_serves(core, fin: _TepFinar, x, L: _LayerWeights, fp4, router_mm) -> bool:
    """Whether a tep4 block takes the fused launch: a captured block of a row count every rank's
    route front serves (`fin.rows`), with the site's codes and the decode router. Host values
    only, the same on every rank."""
    st = core._route_front
    rows = x.shape[0]
    return (
        rows in fin.rows
        and rows in st.tactics
        and st.layers.get(L.fc1_w.data_ptr()) is not None
        and fp4 is not None
        and router_mm is _decode_router
        and torch.cuda.is_current_stream_capturing()
        and x.stride(1) == 1
        and x.stride(0) == core.hidden
    )


def _tep_finar_routed(core, fin: _TepFinar, x, L: _LayerWeights, fp4, router_mm, quarter=None):
    """A tep4 captured block's routed experts summed over the group in one launch (the route
    front, the runner without its finalize, `tep_finalize_allreduce`), or None where the block
    keeps the route front's finalized path and `core.tp_sum` (another row count, an eager step).
    Host values only, the same on every rank (`fin.rows` the row counts every rank's route
    front serves). `quarter` is the `(shared, gate)` of `_tep_finar_quarter` in flight on the side
    stream (`TEP_FINAR_SHARED_ENV`): the launch joins the side stream and adds it."""
    st = core._route_front
    rows = x.shape[0]
    if not _tep_finar_serves(core, fin, x, L, fp4, router_mm):
        return None
    layer = st.layers[L.fc1_w.data_ptr()]
    if fin.prev is not None and layer != (fin.prev + 1) % fin.layers:
        raise RuntimeError(
            f"{TEP_FINAR_ENV}: MoE layer {layer} follows layer {fin.prev}; three consecutive "
            "launches must take three slots"
        )
    fin.prev = layer
    logits = _tep_front_logits(st, x, L)
    fc2, _, _, _, wts = _route_front_run(core, st, x, L, layer, logits, fp4, finalize=False)
    perm = st.views[rows][layer][0].view(rows, core.topk)
    if quarter is None:
        return tep_finalize_allreduce(fc2, wts, perm, rows, layer % _TEP_FINAR.SLOTS, fin.op)
    torch.cuda.current_stream().wait_stream(core._side_stream)
    return tep_finalize_allreduce(fc2, wts, perm, rows, layer % _TEP_FINAR.SLOTS, fin.op, *quarter)


def _routed_experts(core, x, L: _LayerWeights, fp4=None, router_mm=cublas_mm) -> torch.Tensor:
    """This rank's partial of the routed experts over every row of `x`: the router logits
    over the whole routing space, and the trtllm-gen runner over this rank's expert window
    entered with them -- its own Renormalize routing (bitwise `renorm_moe_routing_op` then
    the pre-routed runner) -- on the NVFP4 activation with linear block scales at the
    layer's FC1 input global scale, finalized. `fp4` is that activation as the MoE-site
    mixer wrote it (`_hc_moe_mixer`, the rows of `x`), or None to quantize `x` here;
    `router_mm` the step's router projection (`_decode_router` at decode). A slot
    routed outside the window adds nothing, so the window outputs sum to the layer's. Under
    the intermediate split (tp4) the window is every expert at this rank's
    `expert_inter`-wide slice, and the slices' outputs sum to the layer's the same way.

    Every step is row-wise, so `x` runs in chunks of at most the runner's certified rows (a
    host decision on `T`): one chunk at a TP rank's rows, several over the rows an
    attention-DP group gathers.

    At one rank with the CuteDSL-layout copy (`moe_prefill_copy`) a launch of
    `_CUTE_DSL_MOE_MIN_ROWS` to `_CUTE_DSL_MOE_MAX_ROWS` rows runs the CuteDSL MoE over it
    instead, in one call. Without it, a context launch (the prefill router) of more than
    `_MOE_MAX_ROWS` and at most `_MOE_ONE_LAUNCH_MAX_ROWS` rows is one runner call over the
    chunks' router logits: the same bits as the chunks. On the CuteDSL backend (one rank) every
    launch is `_cute_dsl_routed`, in one call."""
    rows = x.shape[0]
    route_front = getattr(core, "_route_front", None)
    if (
        route_front is not None
        # Under attention DP the gathered rows come unquantized; a tep4 rank's come with the
        # site's codes.
        and (fp4 is None) == (core.dp_size > 1)
        and (router_mm is _decode_router or isinstance(router_mm, _DpRouterFront))
        and torch.cuda.is_current_stream_capturing()
    ):
        routed = _route_front_experts(core, route_front, x, L, fp4, router_mm)
        if routed is not None:
            return routed
    if core.trunk_moe_backend == "CUTEDSL":
        xq, xsf = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, False) if fp4 is None else fp4
        routed_out = torch.empty([rows, core.hidden], dtype=x.dtype, device=x.device)
        return _cute_dsl_routed(core, L, router_mm(x, L.router), xq, xsf, routed_out)
    if L.fc1_w_cd is not None and _CUTE_DSL_MOE_MIN_ROWS <= rows <= _CUTE_DSL_MOE_MAX_ROWS:
        if router_mm is cublas_mm:
            _ctx_form("moe cutedsl copy", rows)
        xq, xsf = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, False) if fp4 is None else fp4
        return fp4_moe_renorm_cute_dsl(
            router_mm(x, L.router),
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
            memset_stream=core._moe_memset_stream,
        )
    if (
        core.one_rank
        and router_mm is cublas_mm
        and fp4 is not None
        and _MOE_MAX_ROWS < rows <= _MOE_ONE_LAUNCH_MAX_ROWS
    ):
        # The router GEMM keeps its chunk-sized calls: cuBLAS picks its kernel by the row count,
        # and another kernel could move a logit's last bit and flip a near-tie of the top-k.
        _ctx_form("moe one launch", rows)
        logits = torch.cat(
            [
                router_mm(x[lo : lo + _MOE_MAX_ROWS], L.router)
                for lo in range(0, rows, _MOE_MAX_ROWS)
            ]
        )
        return fp4_block_scale_moe_renorm_routed(
            logits,
            fp4[0],
            torch.reshape(fp4[1], [-1]).view(torch.float8_e4m3fn),
            L.fc1_w,
            L.fc1_s,
            L.fc2_w,
            L.fc2_s,
            L.out1_scale,
            L.out1_gate_scale,
            L.out2_scale,
            core.num_experts,
            core.topk,
            core.expert_inter,
            core.expert_offset,
            core.local_experts,
        )
    if core.one_rank and router_mm is cublas_mm and rows > _MOE_MAX_ROWS:
        _ctx_form(f"moe {_MOE_MAX_ROWS}-row chunks", rows)
    sf_cols = x.shape[1] // _SF_VEC
    parts = []
    for lo in range(0, rows, _MOE_MAX_ROWS):
        hi = min(rows, lo + _MOE_MAX_ROWS)
        chunk = x[lo:hi]
        logits = router_mm(chunk, L.router)
        if fp4 is None:
            xq, xsf = fp4_quantize(chunk, L.moe_gscale, _SF_VEC, False, False)
        else:
            xq, xsf = fp4[0][lo:hi], fp4[1][lo * sf_cols : hi * sf_cols]
        parts.append(
            fp4_block_scale_moe_renorm_routed(
                logits,
                xq,
                # The runner reads the chunk's linear block scales flat.
                torch.reshape(xsf, [-1]).view(torch.float8_e4m3fn),
                L.fc1_w,
                L.fc1_s,
                L.fc2_w,
                L.fc2_s,
                L.out1_scale,
                L.out1_gate_scale,
                L.out2_scale,
                core.num_experts,
                core.topk,
                core.expert_inter,
                core.expert_offset,
                core.local_experts,
            )
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _cute_dsl_routed(core, L, logits, xq, xsf, routed_out) -> torch.Tensor:
    """The trunk's routed experts on the CuteDSL pipeline (one rank, every expert):
    Renormalize routing, the sort into per-expert row tiles, FC1 gathering its rows from the
    unpermuted NVFP4 activation `xq` / `xsf` (linear scales) with SwiGLU and the requantize to
    the layer's one FC2 input scale fused, FC2, and the routing-weighted unpermute in a fixed
    slot order, which writes every row of `routed_out` `[T, hidden]` and returns it."""
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


def _shared_expert(core, x, L: _LayerWeights, decode_gemm=None) -> tuple:
    """The shared expert and its per-token gate logit over `x`: `(shared, gate_logits)`. A
    decode step under the dense FP8 mode (`decode_gemm` its bf16 projection entry, the copies
    present) runs gate|up and down on their FP8 / mid-rows copies and the gate logit on its own
    bf16 row; otherwise one GEMM forms gate|up and the gate logit (`sh_gu`'s extra rows)."""
    if decode_gemm is not None and L.sh_gu_g is not None:
        gu = _dense_mm(x, L.sh_gu, None, decode_gemm, "sh_gu", L.sh_gu_g, mid=L.sh_gu_m)
        shared = _dense_mm(
            silu_and_mul_rows(gu, core.shared_inter),
            L.sh_dn,
            None,
            decode_gemm,
            "sh_dn",
            L.sh_dn_g,
            mid=L.sh_dn_m,
        )
        return shared, decode_gemm(x, L.sh_gate)
    # A tep4 decode step under `TEP_PROJ_ENV`: both GEMMs on the split-K low-M kernel.
    mm = (
        _tep_wide_mm
        if decode_gemm is not None
        and getattr(core, "tep_proj", False)
        and x.shape[0] <= _LOW_M_WIDE_MAX_ROWS
        else cublas_mm
    )
    gug = mm(x, L.sh_gug)
    shared = mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
    return shared, gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]


def _moe_block(
    core,
    x,
    L: _LayerWeights,
    pad_rows: int,
    fp4=None,
    router_mm=cublas_mm,
    decode_gemm=None,
    rs_next=None,
) -> tuple:
    """One MoE layer: NVFP4 routed experts plus the replicated bf16 shared expert, left
    uncombined. Returns `(routed, gate_logits, shared)`: the next `qwen4_exp_hc_inject_norm`
    forms `routed + sigmoid(gate_logits) * shared` inside its kernel. `routed` is every
    window's contribution to this rank's rows (`core.expert_sum`): under attention TP the
    window partials' sum over the TP group, the same bits on every rank; under attention DP
    the round trip over the group's gathered rows, `pad_rows` padding this rank's.

    The shared expert and its per-token gate logit run on the core's side stream,
    concurrently with the routed path, and join it before this returns. `x` is the bf16
    MoE-site mixer output and `fp4` its NVFP4 form (`_hc_moe_mixer`): present only under
    attention TP, where the window runs on `x`'s own rows. `router_mm` is the step's
    router projection (`_routed_experts`). `decode_gemm` is a decode step's bf16 projection
    entry, under which the shared expert reads the dense FP8 mode's copies when they exist.

    `rs_next` (`_dp_rs_inject_next`, None unless `DP_RS_INJECT_ENV` engages): the next site's
    `(hc, inject, norm weight)`. The round trip's reduce-scatter is then the fused launch, behind
    the join, and this returns `(_RsInjected, None, None)`; at a row count of its finalize form
    (`DP_MOE_FIN_RS_ENV`) the window enters the runner without its finalize."""
    main = torch.cuda.current_stream()
    side = core._side_stream
    fin = getattr(core, "_tep_finar", None)
    # `TEP_FINAR_SHARED_ENV`: this rank's quarter of the shared expert, which the fused launch adds.
    quarter = _tep_finar_takes_quarter(core, x) and _tep_finar_serves(
        core, fin, x, L, fp4, router_mm
    )
    # Fork: the side stream reads `x` only after everything queued on `main` so far. What it
    # allocates is consumed on `main` after the join below.
    side.wait_stream(main)
    forked = []

    def shared_expert() -> None:
        with torch.cuda.stream(side):
            forked.append(_shared_expert(core, x, L, decode_gemm))

    if quarter:
        # A tep4 rank only (`TEP_FINAR_SHARED_ENV` never serves attention DP): the side stream
        # runs this rank's quarter, which the fused launch adds; the next site reads zeros.
        with torch.cuda.stream(side):
            quarter = _tep_finar_quarter(core, x, L)
        rows = x.shape[0]
        forked.append((fin.shared.zero_shared[:rows], fin.shared.zero_gate[:rows]))
    else:
        quarter = None
        # `DP_LAUNCH_ORDER_ENV`: under attention DP the shared expert is launched from inside the
        # round trip, right after its gather (the round trip calls its window once).
        late = getattr(core, "dp_launch_order", False) and core.dp_size > 1
        if not late:
            shared_expert()
    routed = None
    if fin is not None:
        routed = _tep_finar_routed(core, fin, x, L, fp4, router_mm, quarter)
    pk = getattr(core, "_dp_pack", None)
    if (
        routed is None
        and pk is not None
        and not pad_rows
        and torch.cuda.is_current_stream_capturing()
    ):
        # `DP_MOE_PACK_GATHER_ENV`: the packed gather in place of the bf16 one (None: not here).
        packed = _dp_pack_block(core, pk, x, L, rs_next, shared_expert, forked, main, side)
        if packed is not None:
            routed, joined = packed
            if joined:
                return routed, None, None
    if routed is None:
        rsi = getattr(core, "_rs_inject", None) if rs_next is not None else None

        def window(rows: torch.Tensor) -> torch.Tensor:
            if not forked:
                shared_expert()
            if (
                rsi is not None
                and x.shape[0] in rsi.fin_rows
                and torch.cuda.is_current_stream_capturing()
                and rsi.rs.refuses_fin(*rs_next, *forked[0]) is None
            ):
                # `DP_MOE_FIN_RS_ENV`: the fused launch forms the partials from this.
                unfinalized = _route_front_unfinalized(core, rows, L, router_mm)
                if unfinalized is not None:
                    return unfinalized
            return _routed_experts(core, rows, L, fp4, router_mm)

        if rsi is None:
            routed = core.expert_sum(window, x, pad_rows)
        else:
            tail = _rs_inject_tail(core, rsi, rs_next, lambda: forked[0], main, side)
            routed = core.expert_sum(window, x, pad_rows, tail)
            if type(routed) is _RsInjected:
                # Joined in front of the fused launch; nothing is left for the next site.
                return routed, None, None
    if not forked:
        shared_expert()
    main.wait_stream(side)
    shared, gate_logits = forked[0]
    return routed, gate_logits, shared


def _attn_in_heads(core, proj):
    """The main q, k and v heads of the fused attention projection's output `proj`
    `[T, attn_in_width]`, as `[T, heads, head_dim]` views."""
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


def _attention_rope(core, proj, index_scratch, L: _LayerWeights, pos3, pos) -> None:
    """Both planes' Gemma norm + rope. The main q/k heads in place on the fused projection
    row (interleaved mRoPE over the three position rows; everything after k is passed as v
    heads, which only sets the row stride). The index projection's query and key are copied
    into the index scratch and normalized and rotated there (coordinate 0 alone), so the raw
    index key stays in `proj` for the index plane's store."""
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
        base=core.theta,
        is_neox=True,
        position_ids=pos3,
        is_qk_norm=True,
        use_gemma=True,
        use_mrope=True,
        mrope_section1=core.mrope_section1,
        mrope_section2=core.mrope_section2,
    )
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
        base=core.theta,
        is_neox=True,
        position_ids=pos,
        is_qk_norm=True,
        use_gemma=True,
        use_mrope=False,
        mrope_section1=0,
        mrope_section2=0,
    )


def _index_query(core, index_scratch):
    """The normalized, rotated index query `[T, index_heads, index_head_dim]`: the q slots
    of the index scratch."""
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
    """The QSA index branch of attention layer `i`: store the raw index key and the
    positions of `proj`'s rows into the paged index plane, then compress every group this
    step completes. The raw key is stored before anything normalizes or rotates it, because
    the compression applies both itself."""
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
    _index_anchors_fp8(
        core,
        index_cache,
        block_table,
        req_idx,
        logical,
        tokens_per_block,
    )


def _select(core, scores, row_starts, row_ends, topk_indices, live_cols=None) -> None:
    """Each row's canonical top-`block_topk` groups of its window into `topk_indices`: the
    groups strictly above the budget's last score, then the highest-index ties, ascending,
    `-1` last -- a function of the scores alone, which the expansion and the sparse GQA
    reduce in. QSA's row starts are zero, so window-local and table indices agree.

    The exact top-k and the canonical rewrite, two launches, which serve a draft step whose
    rows are the rejection kernel's accepted-token runs at every width the engine's graph
    buckets use.

    The rewrite reads only the columns of each row's window, and `scores` is as wide as the
    block table; a caller that knows a host-side bound on the windows passes it as
    `live_cols` so the rewrite reads only those columns and returns the same rows."""
    qsa_topk_select(scores, row_starts, row_ends, topk_indices, index_topk=core.block_topk)
    if live_cols is not None:
        scores = scores[:, :live_cols]
    qsa_canonical_selection(scores, row_starts, row_ends, topk_indices)


def _live_score_columns(core, md, batch: int) -> int:
    """A host-side bound on every row's selection window in a step with context rows: the
    longest sequence's compressed length, with the verify width as slack for the draft rows
    that sit past the host's KV lengths."""
    longest = int(md.kv_lens_runtime[:batch].max()) if batch > 0 else 0
    return (longest + _MAX_VERIFY_WIDTH) // core.compress_ratio + 1


def _selection(
    core,
    scores,
    visible_blocks,
    logical,
    seq_lens_q,
    topk_indices,
    row_starts,
    out,
    live_cols=None,
    context=False,
    draft=False,
):
    """The selected token indices of `scores`' rows into `out` `[rows, token_topk +
    compress_ratio - 1]`: the canonical block top-k and its expansion in one launch (a cluster
    of CTAs a row up to `qsa_topk_expand_cluster`'s rows, an attention-DP rank's
    `DP_TOPK_CLUSTER_ROWS_ENV` for generation rows, placed by `_topk_placement` for the trunk or
    the `draft` layer past 37 rows, else one CTA a row; a context chunk's rows, `context`, on the
    one-CTA op's context launch), or, for a score table neither serves, the selection and the
    expansion apart (the rewrite bounded by `live_cols`, `_select`). They all give the same
    selection."""
    cluster_rows = _TOPK_CLUSTER_MAX_ROWS
    if not context:
        cluster_rows = getattr(core, "dp_topk_cluster_rows", _TOPK_CLUSTER_MAX_ROWS)
    if _topk_expand_cluster_supports(scores, max_rows=cluster_rows):
        placement = _topk_placement(scores, draft) if cluster_rows > _TOPK_CLUSTER_MAX_ROWS else {}
        if not context and getattr(core, "topk_cowrite", False):
            placement["cowrite"] = True
        qsa_topk_expand_cluster(
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            core.compress_ratio,
            core.token_topk,
            out=out,
            **placement,
        )
        return
    if _topk_expand_supports(scores):
        if context:
            # The op's block top-k: `topk_indices` is `[rows, token_topk / compress_ratio]`.
            took = _topk_takes_context(scores, core.token_topk // core.compress_ratio)
            _ctx_form("ctx top-k cta" if took else "ctx top-k decode launch", scores.shape[0])
        qsa_topk_expand_decode(
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            core.compress_ratio,
            core.token_topk,
            out=out,
            context=context,
        )
        return
    if context:
        _ctx_form("ctx top-k select + expand", scores.shape[0])
    _select(core, scores, row_starts, visible_blocks, topk_indices, live_cols)
    out.copy_(
        qsa_expand_block_indices(
            topk_indices,
            logical,
            seq_lens_q,
            compress_ratio=core.compress_ratio,
            token_topk=core.token_topk,
        )
    )


#: Elements of one score table: `qsa_paged_index_scores` writes its output rows and
#: `qsa_topk_expand_decode` lays out its rows with int32 index arithmetic (row times the
#: table's width), so a call scores fewer rows than this over the table's columns.
_SCORE_TABLE_ELEMENTS = 2**31


def _score_and_select(
    core,
    index_cache,
    q_index,
    block_table,
    logical,
    req_idx,
    tokens_per_block: int,
    parts,
    visible_blocks,
    seq_lens_q,
    topk_indices,
    row_starts,
    selected,
    live_cols,
) -> None:
    """Score each part's rows against the index cache and select into their rows of
    `selected` (`_selection`). A part's rows go in chunks of fewer than
    `_SCORE_TABLE_ELEMENTS / columns` rows: a prefill step's context rows reach
    `max_num_tokens`, and at `max_seq_len` 262144 (65568 columns) more than 32752 of them
    pass the int32 range. A row's scores and selection read only that row's query, position
    and request, so the chunks select what one call would; below the bound a part is one
    call."""
    columns = block_table.shape[1] * tokens_per_block // core.compress_ratio
    chunk = max(1, (_SCORE_TABLE_ELEMENTS - 1) // columns)
    for lo, hi, context_rows in parts:
        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)
            scores = _index_scores(
                core,
                q=q_index[a:b],
                index_cache=index_cache,
                block_table=block_table,
                query_positions=logical[a:b],
                request_indices=req_idx[a:b],
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=context_rows,
            )
            _selection(
                core,
                scores,
                visible_blocks[a:b],
                logical[a:b],
                seq_lens_q[a:b],
                topk_indices[a:b],
                row_starts[a:b],
                selected[a:b],
                live_cols,
                context=context_rows,
            )


def _store_kv(core, kv, k, v, request_indices, logical_positions, block_table, tokens_per_block):
    """Append this step's K/V rows to the layer's paged pool: a plain move over a bf16 or fp8
    pool (an fp8 pool takes e4m3 unscaled, as the dense op does at the unit KV scales), the
    quantizing store over an NVFP4 one, at the dense op's second-level scales."""
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


def _nvfp4_context_groups(pages: list, rows: list, max_pages: int) -> list:
    """Consecutive context requests in groups of at most `max_pages` pages, a request with more
    pages alone: `(first request, end request, first row, end row)` a group, where request `r`
    has `pages[r]` pages and the next `rows[r]` rows."""
    groups = []
    first = first_row = row = listed = 0
    for r, (n_pages, n_rows) in enumerate(zip(pages, rows)):
        if r > first and listed + n_pages > max_pages:
            groups.append((first, r, first_row, row))
            first, first_row, listed = r, row, 0
        listed += n_pages
        row += n_rows
    if pages:
        groups.append((first, len(pages), first_row, row))
    return groups


def _nvfp4_context_plan(core, md, req_idx, block_table, tpb: int) -> tuple:
    """The pages a step's context requests read, for the bf16 row-tile pass over an NVFP4 pool:
    request `r` lists its first `ceil(kv_len / tpb)` table pages. The requests come in groups
    that fit `_NVFP4_CONTEXT_SCRATCH_BYTES` (`_nvfp4_context_groups`); within a group request `r`
    takes scratch pages `offsets[r]` on. Returns `(groups, table, k_scratch, v_scratch)`: per
    group `(first row, end row, slots, counts, offsets)`, the operands `qsa_nvfp4_pages_to_bf16`
    fills per layer and group; `table` (all -1 here) gets the requests' slot rows, the scratch
    pools `[pages, Hkv, tpb, D]` bf16 the largest group."""
    num_contexts = md.num_contexts
    device = block_table.device
    pages = ((md.kv_lens_runtime[:num_contexts] + (tpb - 1)) // tpb).tolist()
    rows = md.seq_lens[:num_contexts].tolist()
    page_bytes = 2 * core.heads_kv * tpb * core.head_dim * 2  # K and V, bf16
    groups = _nvfp4_context_groups(pages, rows, max(1, _NVFP4_CONTEXT_SCRATCH_BYTES // page_bytes))
    lens = md.seq_lens_cuda[:num_contexts].long()
    slots = req_idx[torch.cumsum(lens, 0) - lens]
    counts = torch.div(
        md.kv_lens_cuda_runtime[:num_contexts].long() + (tpb - 1), tpb, rounding_mode="floor"
    )
    planned = []
    for first, end, first_row, end_row in groups:
        group_counts = counts[first:end]
        planned.append(
            (
                first_row,
                end_row,
                slots[first:end],
                group_counts,
                torch.cumsum(group_counts, 0) - group_counts,
            )
        )
    listed = max(sum(pages[first:end]) for first, end, _, _ in groups)
    table = torch.full_like(block_table, -1)
    shape = (listed, core.heads_kv, tpb, core.head_dim)
    k_scratch = torch.empty(shape, dtype=torch.bfloat16, device=device)
    v_scratch = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return planned, table, k_scratch, v_scratch


def _sparse_gqa_nvfp4_context(
    core, kv, plan, q, block_table, selected, req_idx, logical, tpb, num_ctx, out, rows_per_request
) -> torch.Tensor:
    """Sparse GQA of a step with context rows over an NVFP4 pool, into `out` `[rows, q_width]`.
    With a plan (`_nvfp4_context_plan`) the first `num_ctx` rows, the context rows, take the
    row-tile pass over the plan's scratch, filled here from this layer's pool a group of
    requests at a time: at one rank the family's pass, elsewhere the target's, which also serves
    a tp4 / tep4 rank's 6 / 1 heads. The other rows, and every row without a plan, go split-K
    over the pool itself. The generation rows after the context rows come `rows_per_request` a
    request: more than one (a verify step's) takes the request-grouped entry for them on their
    own."""
    rows = q.shape[0]
    head = 0
    if plan is not None:
        groups, table, k_scratch, v_scratch = plan
        assert groups[-1][1] == num_ctx, (groups[-1][1], num_ctx)
        rowtile = one_rank_sparse_gqa_rowtile if core.one_rank else qsa_sparse_gqa_rowtile
        for first_row, end_row, slots, counts, offsets in groups:
            qsa_nvfp4_pages_to_bf16(
                kv.k,
                kv.v,
                kv.k_scale,
                kv.v_scale,
                core.kv_scale_quant_orig,
                block_table,
                slots,
                counts,
                offsets,
                table,
                k_scratch,
                v_scratch,
            )
            context = rowtile(
                q=q[first_row:end_row],
                k_cache=k_scratch,
                v_cache=v_scratch,
                block_table=table,
                selected_tokens=selected[first_row:end_row],
                request_indices=req_idx[first_row:end_row],
                tokens_per_block=tpb,
                softmax_scale=core.softmax_scale,
                query_positions=logical[first_row:end_row],
                compress_ratio=core.compress_ratio,
            )
            out[first_row:end_row].copy_(
                torch.reshape(context, [end_row - first_row, core.q_width])
            )
        head = num_ctx
    elif (
        rows_per_request > 1
        and 0 < num_ctx < rows
        and _verify_takes_grouped(
            q[num_ctx:], kv.k, block_table, selected[num_ctx:], tpb, rows_per_request
        )
    ):
        context = _sparse_gqa(
            core, kv, q[:num_ctx], block_table, selected[:num_ctx], req_idx[:num_ctx], tpb, None
        )
        out[:num_ctx].copy_(torch.reshape(context, [num_ctx, core.q_width]))
        head = num_ctx
    if head < rows:
        rest = _sparse_gqa(
            core,
            kv,
            q[head:],
            block_table,
            selected[head:],
            req_idx[head:],
            tpb,
            None,
            rows_per_request=rows_per_request if head == num_ctx else 1,
        )
        out[head:rows].copy_(torch.reshape(rest, [rows - head, core.q_width]))
    return out


def _fp8_context_pages(md, tpb: int) -> int:
    """Pool pages a step's context requests hold, `ceil(kv_len / tpb)` each, from the host's
    KV lengths: the bf16 scratch `_sparse_gqa_rowtile` widens them into over an fp8 pool."""
    lens = md.kv_lens_runtime[: md.num_contexts]
    return int(((lens + (tpb - 1)) // tpb).sum())


def _fp8_context_groups(core, md, tpb: int, fp8_pages: int) -> list | None:
    """A step's context rows in groups of consecutive requests whose pages each fit the row-tile
    scratch (`_FP8_ROWTILE_MAX_SCRATCH`), `(first row, end row, pages)` a group, or None where
    one launch fits. Without it a step over more history than the scratch holds (three long
    requests in one batch) falls back to the per-row kernel for every row."""
    page_bytes = 2 * core.heads_kv * tpb * core.head_dim * 2  # K and V, bf16
    max_pages = max(1, _FP8_ROWTILE_MAX_SCRATCH // page_bytes)
    if fp8_pages <= max_pages or md.num_contexts < 2:
        return None
    num_contexts = md.num_contexts
    pages = ((md.kv_lens_runtime[:num_contexts] + (tpb - 1)) // tpb).tolist()
    rows = md.seq_lens[:num_contexts].tolist()
    return [
        (first_row, end_row, sum(pages[first:end]))
        for first, end, first_row, end_row in _nvfp4_context_groups(pages, rows, max_pages)
    ]


def _sparse_gqa_rowtile(
    core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages, out=None
):
    """The row-tile pass over a context launch, or None where it does not serve it. A bf16
    pool is read in place. Over an fp8 pool, request slot `s`'s first `ceil((p_s + 1) / tpb)`
    pages (`p_s` the highest position the launch's rows hold there) are widened, exactly, into
    a bf16 scratch pool addressed through a table of `block_table`'s shape; `fp8_pages` is
    their host count (`_fp8_context_pages`), read back from the device when not given. A
    scratch past `_FP8_ROWTILE_MAX_SCRATCH` bytes declines (None). `out` (`[rows, Hq, D]`,
    contiguous) is where the pass writes, or None for a fresh tensor."""
    k_cache, v_cache, table = kv.k, kv.v, block_table
    if kv.k.dtype == torch.float8_e4m3fn:
        # Each slot's highest position among the launch's rows, -1 for a slot it does not
        # hold: a compare-and-max over `[slots, rows]` (a scatter max would serialize every
        # row of a request on one address).
        slots = torch.arange(block_table.shape[0], dtype=request_indices.dtype, device=q.device)
        held = request_indices[None, :] == slots[:, None]
        last = torch.where(held, query_positions[None, :].long(), -1).amax(1)
        counts = torch.div(last + tpb, tpb, rounding_mode="floor")
        if fp8_pages is None:
            fp8_pages = int(counts.sum())
        # K and V pages, bf16.
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
        out=out,
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
    rowtile_out=None,
    rowtile_min_rows=None,
):
    """Sparse GQA over the layer's paged pool. `query_positions` is a context row's bound on
    its scan (the prefill pair with `compress_ratio`), or None for decode rows. A context
    launch of `rowtile_min_rows` (default `_ROWTILE_MIN_ROWS`) rows or more over a bf16 or fp8
    pool takes the row-tile pass (`fp8_pages`: see `_sparse_gqa_rowtile`), which writes
    `rowtile_out` (`[rows, Hq, D]`, contiguous) when given and returns it. Over an NVFP4 pool
    context rows take the prefill kernel and decode rows the split-K decode kernel."""
    if rowtile_min_rows is None:
        rowtile_min_rows = _ROWTILE_MIN_ROWS
    if kv.k_scale is None:
        if query_positions is not None and q.shape[0] >= rowtile_min_rows:
            out = _sparse_gqa_rowtile(
                core, kv, q, block_table, selected, request_indices, tpb, query_positions,
                fp8_pages, rowtile_out,
            )  # fmt: skip
            if out is None:
                _ctx_form("row-tile declined, per-row kernel", q.shape[0])
            else:
                _ctx_form(f"row-tile {_rowtile_prefill.last_plan}", q.shape[0])
                if rowtile_out is not None and out.data_ptr() == rowtile_out.data_ptr():
                    _ctx_form("row-tile out in place", q.shape[0])
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
        # `output_gate` (`[rows, Hq * D]`) returns the gated output; `rows_per_request` > 1 is a
        # verify step's rows, consecutive per request, whose K/V the verify kernel loads once.
        if rows_per_request > 1:
            return qsa_paged_sparse_gqa_nvfp4_verify(
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
                rows_per_request,
                output_gate=output_gate,
            )
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
            output_gate,
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


def _index_scores(core, **kw):
    """`qsa_paged_index_scores(**kw)`, or under `INDEX_KEY_FP8_ENV` the e4m3 anchors' kernel
    (`qsa_index_scores_fp8`), which takes no launch hints (PDL, prefetch, the visible split)."""
    if not getattr(core, "index_key_fp8", False):
        return qsa_paged_index_scores(**kw)
    if kw["context_rows"]:
        tile = _takes_ctx_tile(
            kw["q"], kw["only_visible_blocks"], True, kw.get("rows_per_request", 1)
        )
        _ctx_form("index fp8 ctx tile" if tile else "index fp8 request-group", kw["q"].shape[0])
    return qsa_index_scores_fp8(
        kw["q"],
        kw["index_cache"],
        kw["block_table"],
        kw["query_positions"],
        kw["request_indices"],
        kw["tokens_per_block"],
        kw["compress_ratio"],
        kw["only_visible_blocks"],
        kw["context_rows"],
        kw.get("rows_per_request", 1),
    )


def _index_anchors_fp8(
    core, index_cache, block_table, req_idx, logical, tpb, use_pdl: bool = False
) -> None:
    """Under `INDEX_KEY_FP8_ENV`, the anchors a writer just compressed (as bf16) at `logical`'s
    group ends, converted to e4m3 in place; on the writer's stream, right behind it. `use_pdl`
    (`DP_SCORES_EARLY_ENV`, behind the attention decode prologue): a PDL launch that releases
    its dependents before its wait."""
    if getattr(core, "index_key_fp8", False):
        kw = {"use_pdl": True} if use_pdl else {}
        index_anchors_to_fp8(
            index_cache, block_table, req_idx, logical, tpb, core.compress_ratio, **kw
        )
        if getattr(core, "cold_index_marks", False):
            # The groups this writer pooled, for the cold-page codec (`CTX_COLDIDX_ENV`). A plain
            # launch: behind a PDL convert (`use_pdl`) it still waits for the convert to finish,
            # and a PDL successor starts when it ends (ordered; only the early overlap is lost).
            qsa_cold_index_mark(
                index_cache, block_table, req_idx, logical, tpb, core.compress_ratio
            )


def _decode_index_scores(
    core, q, index_cache, block_table, logical, req_idx, tpb, p: int, behind_prologue=False
):
    """`qsa_paged_index_scores` (visible blocks only) of a sparse generation step's rows, `p`
    consecutive rows a request: from `_INDEX_GROUP_MIN_ROWS` rows (and at a tep4 rank's
    `_TEP_INDEX_GROUP_ROWS`) one key gather a request, and from `_INDEX_SCORES_CUTE_MIN_GROUPS`
    groups on the CuTe kernel where it serves them. `behind_prologue`: the call's PDL
    predecessor is the trunk's attention decode prologue, after which a tep4 rank's
    `_TEP_INDEX_PREFETCH_ROWS` rows may take the prefetching request-group kernel
    (`TEP_INDEX_PREFETCH_ENV`; not under `INDEX_KEY_FP8_ENV`, whose e4m3 anchors only the fp8
    kernel reads)."""
    rows = q.shape[0]
    if (
        behind_prologue
        and getattr(core, "tep_index_prefetch", False)
        and not getattr(core, "index_key_fp8", False)
        and p > 1
        and rows in _TEP_INDEX_PREFETCH_ROWS
        and rows % p == 0
        and torch.cuda.is_current_stream_capturing()
    ):
        return qsa_paged_index_scores(
            q=q,
            index_cache=index_cache,
            block_table=block_table,
            query_positions=logical,
            request_indices=req_idx,
            tokens_per_block=tpb,
            compress_ratio=core.compress_ratio,
            only_visible_blocks=True,
            context_rows=False,
            use_pdl=True,
            prefetch=True,
            rows_per_request=p,
        )
    grouped = rows >= _INDEX_GROUP_MIN_ROWS or (core.tep_fast and rows in _TEP_INDEX_GROUP_ROWS)
    rpr = p if p > 1 and grouped else 1
    if getattr(core, "index_key_fp8", False):
        # `DP_SCORES_EARLY_ENV`: behind the prologue the anchor convert released its dependents
        # at its top, and the flat grid launches early (ignored by the other kernels).
        kw = {"early": True} if behind_prologue and getattr(core, "dp_scores_early", False) else {}
        return qsa_index_scores_fp8(
            q, index_cache, block_table, logical, req_idx, tpb, core.compress_ratio, True, False,
            p if p > 1 else 1, **kw,
        )  # fmt: skip
    if (
        rpr > 1
        and rows >= _INDEX_SCORES_CUTE_MIN_GROUPS * rpr
        and logical.dtype == torch.int64
        and req_idx.dtype == torch.int32
        and _index_groups_cute_serves(q, index_cache, rpr)
    ):
        return index_scores_groups_cute(
            q, index_cache, block_table, logical, req_idx, tpb, core.compress_ratio,
            _INDEX_SCORES_PDL, rpr,
        )  # fmt: skip
    kw = {}
    pps = getattr(core, "tep_index_split", 0)
    if pps and rpr == 1 and p > 1 and rows % p == 0:
        # `TEP_INDEX_SPLIT_ENV`: one slot a row tile when each row tile is one request's rows,
        # past one row tile only (where it was measured).
        tile = min(rows, _INDEX_SPLIT_DOT_WIDTH // q.shape[1]) if q.dim() == 3 else 0
        if tile and rows > tile and tile % p == 0:
            kw["split_shape"] = (pps, tile // p)
    return qsa_paged_index_scores(
        q=q,
        index_cache=index_cache,
        block_table=block_table,
        query_positions=logical,
        request_indices=req_idx,
        tokens_per_block=tpb,
        compress_ratio=core.compress_ratio,
        only_visible_blocks=True,
        context_rows=False,
        use_pdl=_INDEX_SCORES_PDL,
        rows_per_request=rpr,
        wide_split=True,
        **kw,
    )


def _sparse_gqa_decode(
    core, kv, q, block_table, selected, req_idx, tpb, gate, p: int, trunk: bool = False
):
    """The gated sparse GQA `bf16(attn) * sigmoid(gate)` (`[rows, heads_q, head_dim]`) of a
    generation step's rows over a bf16 or e4m3 pool on the family's CuTe decode kernels: at an
    attention-DP rank under `DP_GQA_BITMAP_ENV` the trunk's (`trunk`) bitmap-union grouped one
    where it serves the call, and under `DP_DRAFT_GQA_GROUPED1_ENV` a draft pass of one row a
    request on the same kernel at `_DRAFT_GQA_G1_ROWS` rows; at a tep4 rank under
    `TEP_GQA_E4M3_ENV` the half-height e4m3 one at the rows split-K serves; the request-grouped
    one from `_GROUPED_DECODE_MIN_REQUESTS` requests of `p` rows (an attention-DP rank's
    `DP_GQA_MIN_REQUESTS_ENV`), else the per-row split-K one up to its rows (`_splitk_serves`).
    None where none serves the call (an NVFP4 pool, a head geometry or row count they were not
    measured at): the caller then runs `_sparse_gqa` and the gate apart."""
    if kv.k_scale is not None:
        return None
    bitmap = getattr(core, "dp_gqa_bitmap", 0) if trunk else 0
    if bitmap and _gqa_bitmap_serves(bitmap, kv, q, block_table, selected, req_idx, tpb, gate, p):
        return fp8_sparse_gqa_verify_grouped(
            q, kv.k, kv.v, block_table, selected, req_idx, tpb, core.softmax_scale, p,
            output_gate=gate, min_groups=min(bitmap, _BITMAP_OP_MIN_GROUPS),
        )  # fmt: skip
    lo, hi = _DRAFT_GQA_G1_ROWS
    if (
        not trunk
        and p == 1
        and getattr(core, "dp_draft_gqa_g1", False)
        and lo <= q.shape[0] <= hi
        and _gqa_bitmap_serves(lo, kv, q, block_table, selected, req_idx, tpb, gate, 1, True)
    ):
        return fp8_sparse_gqa_verify_grouped(
            q, kv.k, kv.v, block_table, selected, req_idx, tpb, core.softmax_scale, 1,
            output_gate=gate, min_groups=min(lo, _BITMAP_OP_MIN_GROUPS),
        )  # fmt: skip
    if (
        getattr(core, "tep_gqa_e4m3", False)
        and _splitk_serves(core, q, kv.k, selected)
        and _e4m3_gqa_supports(q, kv.k, kv.v, block_table, selected, req_idx, tpb)
    ):
        return qsa_paged_sparse_gqa_e4m3_decode(
            q, kv.k, kv.v, block_table, selected, req_idx, tpb, core.softmax_scale, gate
        )
    min_requests = getattr(core, "dp_gqa_min_requests", _GROUPED_DECODE_MIN_REQUESTS)
    if (
        p > 1
        and q.shape[0] >= min_requests * p
        and qsa_sparse_gqa_grouped_cute.serves(q, kv.k, selected, p, min_groups=min_requests)
    ):
        return qsa_sparse_gqa_grouped_cute(
            q, kv.k, kv.v, block_table, selected, req_idx, tpb, core.softmax_scale, p, gate
        )
    if _splitk_serves(core, q, kv.k, selected):
        # A tep4 shard the op's table leaves out names its reach, so the first call compiles
        # every plan up to it (`_TEP_SPLITK_MAX_ROWS`) and no later row count compiles one.
        cap = None
        if core.tep_fast:
            cap = _TEP_SPLITK_MAX_ROWS.get((q.shape[1], kv.k.shape[1], kv.k.dtype))
        return qsa_sparse_gqa_splitk_cute(
            q, kv.k, kv.v, block_table, selected, req_idx, tpb, core.softmax_scale, gate,
            max_rows=cap,
        )  # fmt: skip
    return None


def _gqa_bitmap_serves(
    floor, kv, q, block_table, selected, req_idx, tpb, gate, p: int, one_row: bool = False
) -> bool:
    """Whether `fp8_sparse_gqa_verify_grouped` takes this call: a trunk verify call under
    `DP_GQA_BITMAP_ENV` (`floor` to `_BITMAP_MAX_REQUESTS` requests of `p` > 1 rows), or with
    `one_row` a draft pass of one row a request under `DP_DRAFT_GQA_GROUPED1_ENV` (at least `floor`
    rows); over an e4m3 pool, the layouts its entry asserts (q rows of unit head stride, 16-byte
    aligned; row-contiguous block table, contiguous int32 selection, contiguous request indices,
    `[rows, heads * dim]` gate rows 16-byte aligned) and the kernel's own plan (`takes_grouped`:
    SM103, a shared-memory plan, row offsets the group table addresses). Host side, on shapes
    only."""
    return (
        (p > 1 or (one_row and p == 1))
        and q.shape[0] >= floor * p
        and (one_row or q.shape[0] <= _BITMAP_MAX_REQUESTS * p)
        and q.dim() == 3
        and q.stride(2) == 1
        and q.stride(1) == q.shape[2]
        and (q.stride(0) * q.element_size()) % 16 == 0
        and q.data_ptr() % 16 == 0
        and kv.k.dtype == torch.float8_e4m3fn
        and block_table.stride(0) == block_table.shape[1]
        and selected.dtype == torch.int32
        and selected.is_contiguous()
        and req_idx.is_contiguous()
        and gate is not None
        and gate.dim() == 2
        and gate.shape[1] == q.shape[1] * q.shape[2]
        and gate.stride(1) == 1
        and (gate.stride(0) * gate.element_size()) % 16 == 0
        and gate.data_ptr() % 16 == 0
        and _bitmap_takes_grouped(
            q,
            kv.k,
            block_table,
            selected,
            tpb,
            p,
            min_groups=min(floor, _BITMAP_OP_MIN_GROUPS),
            one_row=one_row,
        )
    )


def _splitk_serves(core, q, k_cache, selected) -> bool:
    """Whether `qsa_sparse_gqa_splitk_cute` takes this decode call: where the op's own table
    serves it (`serves`, the FP8 target's routing), and at a tep4 rank under
    `TEP_FAST_DECODE_ENV` also the geometries of `_TEP_SPLITK_MAX_ROWS` up to their rows."""
    if qsa_sparse_gqa_splitk_cute.serves(q, k_cache, selected):
        return True
    if not core.tep_fast or q.dim() != 3 or k_cache.dim() != 4:
        return False
    rows, heads_q, head_dim = q.shape
    cap = _TEP_SPLITK_MAX_ROWS.get((heads_q, k_cache.shape[1], k_cache.dtype), 0)
    return (
        0 < rows <= cap
        and q.dtype == torch.bfloat16
        and head_dim == k_cache.shape[3] == _SPLITK_HEAD_DIM
        and selected.shape[1] > _SPLITK_TILE
    )


def _copy_positions(position_ids: torch.Tensor) -> torch.Tensor:
    """A fresh contiguous int32 `[3, N]` copy of the step's mRoPE positions (the engine hands
    a slice of a larger buffer, whose row stride the rope rejects)."""
    pos3 = torch.empty(
        [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
        dtype=torch.int32,
        device=position_ids.device,
    )
    pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))
    return pos3


def _step_args(md: TrtllmAttentionMetadata, num_contexts, num_ctx_tokens, paged_context) -> dict:
    """thop_attention's batch state off the prepared metadata. Tensors are engine-owned
    persistent buffers refreshed in place; Python ints are per-capture constants.
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
        # No sliding window: the full admitted length on every attention layer.
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


class _KvRangeProbe:
    """`TRTLLM_MODELING_V2_VALIDATE` over an fp8 KV pool: reports, once, the largest K/V
    magnitude the first eager forward's attention layers store.

    The pool takes K/V at the unit scale, so anything past the e4m3 maximum saturates
    silently (a bf16 inf included). One host sync per attention layer of that one forward;
    a forward under CUDA graph capture is passed over without a read."""

    def __init__(self, last_layer: int) -> None:
        self._last = last_layer
        self._worst = 0.0

    @classmethod
    def for_target(cls, core, enabled: bool) -> "_KvRangeProbe | None":
        """A probe over `core`'s attention layers when `enabled` and the pool is fp8."""
        if not enabled or core.quant_mode != QM_FP8_KV_CACHE:
            return None
        return cls(core.attn_layers[-1])

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


# ------------------------------------------------------------------ prefill ---


# TRTLLM_QWEN4_EXP_INFORWARD_VALIDATE=1: every in-forward snapshot step also runs the two
# context iterations it replaces (the chunk up to P, then the tail) on copies of the pool
# rows, and prints how far the split's states and outputs are from theirs. Debug only.
# The split the executor's warmup runs for a dummy request (`_warm_inforward_split`) reports
# as `[inforward-validate] [warmup] ...`, so a line without `[warmup]` is a served request's.
_INFORWARD_VALIDATE = _switch("TRTLLM_QWEN4_EXP_INFORWARD_VALIDATE")
_inforward_validate_reports = 0
_inforward_validate_warmup = False


def _inforward_report(what: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    global _inforward_validate_reports
    if _inforward_validate_reports >= 400:
        return
    _inforward_validate_reports += 1
    g, r = got.float(), ref.float()
    diff = float((g - r).abs().max()) if g.numel() else 0.0
    scale = float(r.abs().max()) if r.numel() else 0.0
    tag = "[warmup] " if _inforward_validate_warmup else ""
    print(
        f"[inforward-validate] {tag}{what}: bitwise={bool(torch.equal(got, ref))} "
        f"max_abs_diff={diff:.3e} ref_max_abs={scale:.3e} shape={tuple(got.shape)}",
        flush=True,
    )


def _inforward_reference_layout(split, num_tokens: int):
    """The two iterations' row sets of a split step: the rows before every tail (`main`,
    with their `cu`), and for each split request its index among the step's requests."""
    device = split.cu.device
    keep = torch.ones(num_tokens, dtype=torch.bool, device=device)
    keep[split.tail_tok] = False
    main = torch.nonzero(keep).reshape(-1)
    # Request ends in packed rows: the end of each request's last gather row.
    num_rows = split.gather.shape[0]
    is_main = torch.zeros(num_rows, dtype=torch.bool, device=device)
    is_main[split.main_rows] = True
    last = torch.ones(num_rows, dtype=torch.bool, device=device)
    last[:-1] = is_main[1:]
    ends = split.cu[1:][last]
    tails = torch.zeros_like(ends)
    request_of_split = torch.searchsorted(split.main_rows, split.split_rows)
    tail_len = split.cu_tail[1:] - split.cu_tail[:-1]
    tails[request_of_split] = tail_len
    lens = torch.diff(ends, prepend=ends.new_zeros(1)) - tails
    cu_main = torch.cat([lens.new_zeros(1), torch.cumsum(lens, 0)])
    return main, cu_main, request_of_split


class _InforwardSplit(NamedTuple):
    """A context-only step's in-forward snapshot split (`GDNMambaMetadata.inforward_split`,
    planned by the cache manager): `k` requests run through their prompt end in this chunk
    instead of taking one more context iteration for the tail past their snapshot point P.
    The recurrent layers leave the state at P in the request's slot and stage the state at
    the prompt end in the manager's staging row (`finish_inforward_snapshots`).

    `split` holds the long index tensors; the rest are this target's views of them: slot
    vectors in the pool's index dtype, the carry flags of the sequences cut at P (a tail is a
    sequence of its own that the first pass starts from zeros), and for the PLE layer the
    rows before each P its conv state (`ple_conv_win`) and id context (`ple_ngram_win`) at P
    are rebuilt from."""

    split: NamedTuple
    gather: torch.Tensor
    carried: torch.Tensor
    live: torch.Tensor
    ple_conv_win: torch.Tensor
    ple_ngram_win: torch.Tensor
    # The staging rows in the pool's index dtype.
    stage_idx: torch.Tensor


def _inforward_split(md, mm, si_all, hi_all, pools) -> _InforwardSplit | None:
    """This step's in-forward snapshot split, or None on a step without one."""
    split = getattr(mm, "inforward_split", None)
    if split is None:
        return None
    if _INFORWARD_VALIDATE:
        global _inforward_validate_warmup
        _inforward_validate_warmup = bool(getattr(mm, "_warmup_inforward_split", False))
    assert md.num_seqs == md.num_contexts, (
        "an in-forward snapshot split on a step with generation rows"
    )
    carried = torch.zeros(split.gather.shape[0], dtype=hi_all.dtype, device=hi_all.device)
    carried.index_copy_(0, split.main_rows, hi_all[: md.num_contexts])
    # The first packed row of each planned request's tail: its snapshot point P.
    points = split.cu.index_select(0, split.split_rows + 1)

    def window(width: int) -> torch.Tensor:
        back = torch.arange(width, 0, -1, dtype=points.dtype, device=points.device)
        return torch.reshape(points[:, None] - back[None, :], [-1])

    ple_conv, ple_ngram = pools.ple_state
    return _InforwardSplit(
        split=split,
        gather=split.gather.to(si_all.dtype),
        carried=carried,
        live=split.live.to(si_all.dtype),
        ple_conv_win=window(ple_conv.shape[-1]),
        ple_ngram_win=window(ple_ngram.shape[-1]),
        stage_idx=split.stage_rows.to(si_all.dtype),
    )


class _PrefillStep(NamedTuple):
    """One step's host-side state, built by `PrefillTarget.prepare` and read by the forward's
    breaks. `num_tokens` is the batch's row count `n` and `physical_tokens` the `T` the
    forward runs over (`n`, or the prefill CUDA graph bucket it was padded to), and
    `dp_pad_rows` the rows the MoE blocks pad them with under attention DP (zero otherwise,
    and on every step a prefill CUDA graph runs). A new one replaces it every step, so no
    break reads a previous step's."""

    num_tokens: int
    physical_tokens: int
    dp_pad_rows: int
    input_ids: torch.Tensor
    attn: dict
    gdn: tuple
    ple: tuple
    qsa: tuple
    vision: tuple | None
    # A verify step: the generation rows' side-cache writes are staged for the rollback.
    stage: bool = False
    # Over an NVFP4 pool, `_nvfp4_context_plan`'s row-tile plan or None.
    nvfp4_context: tuple | None = None
    # Over an fp8 pool, the context requests' pages the row-tile pass widens
    # (`_fp8_context_pages`); None otherwise.
    fp8_pages: int | None = None
    # Over an fp8 pool, `_fp8_context_groups` when those pages pass one scratch; None otherwise.
    fp8_groups: list | None = None
    # `_live_score_columns`: a bound on every row's selection window.
    live_cols: int | None = None
    # A context-only step with in-forward snapshots: `_InforwardSplit`.
    inforward: _InforwardSplit | None = None


class PrefillTarget:
    """Any batch with context rows (`num_contexts > 0`), possibly with generation rows
    beside them."""

    def __init__(self, core: Qwen4ExpDefaultModelingV2) -> None:
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks.
        self._contract_pending = step_contract_enabled()
        # Dropped once it has reported.
        self._kv_probe = _KvRangeProbe.for_target(core, self._contract_pending)
        # The rstd sites' `(1 + w)` operands (`_HC_RSTD_MIN_ROWS`), formed once here.
        self._layers = [
            L._replace(ahc_norm_w1=_norm_w1(L.ahc_norm), mhc_norm_w1=_norm_w1(L.mhc_norm))
            for L in _layer_views(core)
        ]
        stack = _stack_views(core)
        self._stack = stack._replace(fhc_norm_w1=_norm_w1(stack.fhc_norm))
        # The step `prepare` built, read by the breaks.
        self._step: _PrefillStep | None = None
        # Whether this rank reported the GDN in-place short-context path yet.
        self._gdn_inplace_reported = False

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): calling `step_args`
        surfaces an upstream metadata rename here, not mid-forward."""
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

        The four input projections are one GEMM against the fused operand; the norm + rope
        of both planes is row-wise and runs over all `T` rows here; the cache writes, the
        selection and the attention core read the step's batch and run in the
        `_attention_core` break, which leaves the attention output in `attn_out`. The gate
        is read through its row stride, and the output projection's per-rank partial is
        summed over the TP group."""
        core = self.core
        proj = _dense_mm(x, L.attn_in, L.attn_in8, cublas_mm, "in")
        gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
        _attention_rope(core, proj, index_scratch, L, pos3, pos)
        eager_on_graph(proj, proj_qkv, index_scratch, attn_out, i, inner=self._attention_core)
        fused_sigmoid_mul_inplace(attn_out, gate)
        return core.tp_sum(_dense_mm(attn_out, L.o, L.o8, cublas_mm, "out"))

    def _attention_core(self, proj, proj_qkv, index_scratch, attn_out, i) -> None:
        """Break: the index plane's store and compression, the dense/sparse decision and the
        attention core over the step's `n` rows. Writes rows `[:n]` of `attn_out`
        `[T, q_width]` in place and zeroes rows `[n:]`.

        The index plane advances on every step, including below the threshold, so a request
        growing past it finds its prefix compressed; above it the main K/V append becomes
        this layer's own call. On a verify step the generation rows' index slots (and, on
        the first attention layer, their position slots) are staged for `SpeculativeRollback`
        before the store and the compression overwrite them."""
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
        _, k_dst, v_dst = _attn_in_heads(core, proj)
        if self._kv_probe is not None and not self._kv_probe.observe(i, k_dst, v_dst):
            self._kv_probe = None
        # The packed `[q | k | v]`. The dense op reads it flat, as one compact copy. A sparse
        # context-only step over an unscaled pool reads its query heads through the fused
        # projection's rows instead: the row-tile pass and the per-row kernel take q's strides.
        if (
            sparse
            and not step.stage
            and step.attn["num_ctx_tokens"] == num_tokens
            and core._bound.pools.kv[i].k_scale is None
            and proj.stride(0) % 8 == 0
        ):
            qkv = proj[:, : core.qkv_width]
        else:
            qkv = proj_qkv[:num_tokens]
            qkv.copy_(proj[:, : core.qkv_width])
        if step.stage:
            core.spec_rollback.stage_index(core._bound.pools.position_cache)
        _index_plane(core, proj, i, L, (req_idx, logical, block_table, tokens_per_block, coords))

        if sparse:
            kv = core._bound.pools.kv[i]
            # Append the normalized, rotated K/V: nothing else writes the pool on this path,
            # and the selection reads the current token from it.
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            q_index = _index_query(core, index_scratch[:num_tokens])
            # The step's context rows come first, consecutive rows of a request, and score as
            # query tiles; its generation rows, one a request, score row by row. Each part
            # selects into its own rows.
            num_ctx = step.attn["num_ctx_tokens"]
            parts = [(0, num_ctx, True)]
            if num_ctx < num_tokens:
                parts.append((num_ctx, num_tokens, False))
            selected = torch.empty(
                [num_tokens, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=proj.device,
            )
            _score_and_select(
                core,
                core._bound.pools.index_cache[i],
                q_index,
                block_table,
                logical,
                req_idx,
                tokens_per_block,
                parts,
                visible_blocks,
                seq_lens_q,
                topk_indices,
                topk_row_starts,
                selected,
                step.live_cols,
            )
            # Context rows with the prefill pair (a bound on a row's scan), generation rows
            # without. Over an NVFP4 pool at one rank, or with the row-tile plan, the rows go
            # through `_sparse_gqa_nvfp4_context`.
            if kv.k_scale is not None and (core.one_rank or step.nvfp4_context is not None):
                _sparse_gqa_nvfp4_context(
                    core,
                    kv,
                    step.nvfp4_context,
                    _packed_q(core, qkv),
                    block_table,
                    selected,
                    req_idx,
                    logical,
                    tokens_per_block,
                    num_ctx,
                    attn_out[:num_tokens],
                    step.attn["predicted_tokens_per_seq"],
                )
                parts = []
            if parts and step.fp8_groups is not None:
                _ctx_form("fp8 context groups", num_ctx)
                # The context rows a scratch-sized group of requests at a time.
                assert parts[0] == (0, num_ctx, True), (parts[0], num_ctx)
                assert step.fp8_groups[-1][1] == num_ctx, (step.fp8_groups[-1], num_ctx)
                # The step passed `_ROWTILE_MIN_ROWS`; each group then launches on its own and
                # takes the row-tile pass from `_FP8_GROUP_ROWTILE_MIN_ROWS` rows.
                parts = [
                    (lo, hi, True, pages, _FP8_GROUP_ROWTILE_MIN_ROWS)
                    for lo, hi, pages in step.fp8_groups
                ] + [(lo, hi, context_rows, None, None) for lo, hi, context_rows in parts[1:]]
            else:
                parts = [
                    (lo, hi, context_rows, step.fp8_pages, None) for lo, hi, context_rows in parts
                ]
            for lo, hi, context_rows, pages, rowtile_min_rows in parts:
                # A context launch's row-tile pass writes its rows of `attn_out` itself.
                direct = None
                if context_rows and attn_out.is_contiguous():
                    direct = attn_out[lo:hi].view(hi - lo, core.heads_q, core.head_dim)
                    if direct.data_ptr() % 16:
                        direct = None
                out = _sparse_gqa(
                    core,
                    kv,
                    _packed_q(core, qkv)[lo:hi],
                    block_table,
                    selected[lo:hi],
                    req_idx[lo:hi],
                    tokens_per_block,
                    logical[lo:hi] if context_rows else None,
                    pages,
                    rowtile_out=direct,
                    rowtile_min_rows=rowtile_min_rows,
                )
                if direct is None or out.data_ptr() != direct.data_ptr():
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
        """One Gated-DeltaNet layer over a batch with a context prefix, at this rank's heads:
        the one input projection GEMM (q|k|v, z, a and b are column views of its output) and
        the gated-norm epilogue run over all `T` rows here, the recurrence in the `_gdn_core`
        break, which leaves its output in `core_out` `[1, T, num_v_heads, head_v_dim]`."""
        core = self.core
        proj = _dense_mm(x, L.in_proj, L.in_proj8, cublas_mm, "in")
        z = torch.split(proj, core.in_proj_split, 1)[1]
        eager_on_graph(proj, core_out, i, inner=self._gdn_core)
        return self._gdn_epilogue(core_out, z, L, x.shape[0])

    def _gdn_core(self, proj, out, i) -> None:
        """Break: the recurrence over the step's `n` rows. Writes rows `[:n]` of `out` in place
        and zeroes rows `[n:]`. Every branch is on a host int of the step.

        The context rows take one fused conv + post-conv pass (`_gdn_prefill_prep`) and
        FlashInfer's chunked recurrence, and a mixed step's one-token generation rows the
        decode step's conv and recurrence. A context sequence carrying nothing starts from
        zeros without its pool slots being cleared: the conv pass reads no carried window for
        it, and the recurrence (`has_initial_state`) gathers zeros for it.

        On a verify step the generation rows are draft chains of unknown accepted length:
        they take `_gdn_verify` (per-step conv windows and the cached replay) instead."""
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
            # The context rows as on a step without verify rows.
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
            # The two halves run separate recurrences over disjoint slots and write disjoint
            # row ranges of the output.
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
            # The generation rows: the decode step's recurrence (without the gated norm,
            # which runs with the context rows' below) over their rows of the projection,
            # whose q/k/v columns the conv step rewrote; it computes both gates and the q/k
            # norm itself.
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
        elif step.inforward is not None:
            self._gdn_inforward(mixed_qkv, a, b, L, i, qsl_all, si_all, hi_all, out, num_tokens)
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

    def _gdn_inforward(self, mq, a, b, L, i, qsl, si, hi, out, num_tokens) -> None:
        """The context-only recurrence of a step with in-forward snapshots (`_InforwardSplit`).

        The conv pass runs over whole chunks, so it leaves each planned slot the window that
        ends at the prompt end: that goes to the staging row and the window that ends at P,
        the raw inputs before P, into the slot. The recurrence is cut at P
        (`flashinfer_chunk_gdn_inforward_split`)."""
        core = self.core
        inf = self._step.inforward
        split = inf.split
        pools = core._bound.pools
        conv_states = pools.conv[i]
        stage_ssm, stage_conv = core._bound.cm.inforward_snapshot_stage_views(i)
        _ctx_form("in-forward snapshot split", num_tokens)
        ref = (
            self._gdn_inforward_reference(mq, a, b, L, i, qsl, si, hi, num_tokens)
            if _INFORWARD_VALIDATE
            else None
        )
        window = mq.index_select(0, split.conv_win)
        q, k, v, g, beta = self._gdn_prefill_prep(mq, a, b, L, i, qsl, si, hi)
        slots = split.tail_slots
        stage_conv.index_copy_(0, split.stage_rows, conv_states.index_select(0, slots))
        conv_states.index_copy_(
            0,
            slots,
            torch.transpose(torch.reshape(window, [split.k, conv_states.shape[-1], -1]), 1, 2).to(
                conv_states.dtype
            ),
        )
        flashinfer_chunk_gdn_inforward_split(
            q,
            k,
            v,
            g,
            beta,
            pools.ssm[i],
            cu_seqlens=split.cu,
            gather=inf.gather,
            has_initial_state=inf.carried,
            main_rows=split.main_rows,
            live=inf.live,
            tail_tok=split.tail_tok,
            cu_tail=split.cu_tail,
            tail_slots=split.tail_slots,
            stage=stage_ssm,
            stage_rows=split.stage_rows,
            stage_idx=inf.stage_idx,
            tail_lo=split.tail_lo,
            tail_hi=split.tail_hi,
            output=out[:, :num_tokens],
        )
        if ref is not None:
            ssm_p, conv_p, ssm_end, conv_end, out_ref, live, of_split = ref
            _inforward_report(
                f"gdn L{i} ssm@P (all requests)", pools.ssm[i].index_select(0, live), ssm_p
            )
            _inforward_report(
                f"gdn L{i} conv@P (all requests)", conv_states.index_select(0, live), conv_p
            )
            rows = split.stage_rows
            _inforward_report(
                f"gdn L{i} ssm@end (split)",
                stage_ssm.index_select(0, rows),
                ssm_end.index_select(0, of_split),
            )
            _inforward_report(
                f"gdn L{i} conv@end (split)",
                stage_conv.index_select(0, rows),
                conv_end.index_select(0, of_split),
            )
            _inforward_report(f"gdn L{i} output", out[0, :num_tokens], out_ref)

    def _gdn_inforward_reference(self, mq, a, b, L, i, qsl, si, hi, num_tokens):
        """`_INFORWARD_VALIDATE`: the chunk up to P and then the tail, as two context
        iterations run them, over copies of the step's pool rows (slot `r` = request `r`).
        Returns the states at P, the states at the prompt end and the output rows."""
        core = self.core
        split = self._step.inforward.split
        pools = core._bound.pools
        live = split.live
        main, cu_main, of_split = _inforward_reference_layout(split, num_tokens)
        ssm = pools.ssm[i].index_select(0, live).contiguous()
        conv = pools.conv[i].index_select(0, live).contiguous()
        idx = torch.arange(live.shape[0], dtype=si.dtype, device=si.device)
        out_ref = torch.zeros(
            num_tokens, core.num_v_heads, core.head_v_dim, dtype=mq.dtype, device=mq.device
        )

        def run(rows, cu, slots, carried):
            qv = gdn_prefill_conv_post(
                x=mq.index_select(0, rows).contiguous(),
                weight=L.conv_w,
                conv_states=conv,
                query_start_loc=cu.to(qsl.dtype),
                cache_indices=slots,
                has_initial_state=carried,
                a=a.index_select(0, rows).contiguous(),
                b=b.index_select(0, rows).contiguous(),
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
            o, _ = flashinfer_chunk_gated_delta_rule(
                *qv,
                scale=None,
                initial_state=ssm,
                initial_state_indices=slots,
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=cu,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                output=None,
                state_workspace=None,
                has_initial_state=carried,
                linear_gate=True,
            )
            out_ref.index_copy_(0, rows, o[0].to(out_ref.dtype))

        run(main, cu_main, idx, hi)
        ssm_p, conv_p = ssm.clone(), conv.clone()
        tail_slots = of_split.to(si.dtype)
        run(split.tail_tok, split.cu_tail, tail_slots, torch.ones_like(hi[: split.k]))
        return ssm_p, conv_p, ssm, conv, out_ref, live, of_split

    def _gdn_prefill_prep(self, mq, a, b, L, i, qsl, si, hi):
        """The context rows' recurrence inputs in one pass over `mq`, the token-major q|k|v
        column view of the projection: the causal conv against each sequence's carried
        window (published back to the pool), SiLU, the q|k|v split with q/k L2-normalized,
        and the decay/beta gates (the decay as the factor `exp(g)`, `linear_gate=True`)."""
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
        """The gated RMS norm (sigmoid gate, plain non-gemma weight) and the output
        projection, whose per-rank partial is summed over the TP group."""
        core = self.core
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
        )
        o = torch.reshape(o, [num_tokens, core.value_dim])
        return core.tp_sum(_dense_mm(o, L.out, L.out8, cublas_mm, "out"))

    def _ple_block(self, hc, hc_out, normed_out) -> None:
        """Break: the n-gram side path on its decoder layer, over the step's `n` rows, added
        into the stream and normalized for the layer's attention-side site. Writes rows
        `[:n]` of `hc_out` (the stream) and `normed_out` (its grouped norm) in place and zeroes
        rows `[n:]`.

        Per-request stateful: ids become one `ngram_size`-wide window per token
        (predecessors from the paged id history); each window hashes to one row per head;
        rows gather FP8 or BF16 vectors from one table (`ple_ngram_embed`); the key/value projections,
        the gate, a dilated depthwise short conv over a paged state, the residual add and the
        next site's grouped norm are `ple_gate_conv_inject_norm`. A sequence carrying nothing
        is seeded with `eos_token_id`.

        On a verify step the generation rows' pre-step pool rows and ids are staged before
        the first launch publishes, and the conv input the fused launch writes out after it,
        for `SpeculativeRollback` to rewrite both pools to the accepted prefix."""
        core = self.core
        S = self._stack
        step = self._step
        conv_states, ngram_pool, si_all, carry_all, qsl = step.ple
        num_tokens = step.num_tokens
        next_norm = self._layers[core.ple_layer].ahc_norm
        conv_input = None
        inf = step.inforward
        ple_ref = None
        if inf is not None and _INFORWARD_VALIDATE:
            ple_ref = self._ple_inforward_reference(
                hc, conv_states, ngram_pool, si_all, carry_all, qsl
            )
        if step.stage:
            core.spec_rollback.stage_ple_verify(ngram_pool, conv_states, step.input_ids)
        if step.stage or inf is not None:
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
            row_window=_ple_window(core),
        )
        if _ple_needs_sum(core):
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
        if inf is not None:
            # In-forward snapshots: both pools hold each planned slot's state at the prompt
            # end. Stage it and rebuild the state at P from the rows before P: the conv
            # input tail and the last ids.
            split = inf.split
            slots = split.tail_slots
            stage_conv, stage_ngram = core._bound.cm.inforward_ple_stage_views(core.ple_layer)
            stage_conv.index_copy_(0, split.stage_rows, conv_states.index_select(0, slots))
            stage_ngram.index_copy_(0, split.stage_rows, ngram_pool.index_select(0, slots))
            conv_states.index_copy_(
                0,
                slots,
                torch.transpose(
                    torch.reshape(
                        conv_input.index_select(0, inf.ple_conv_win),
                        [split.k, conv_states.shape[-1], -1],
                    ),
                    1,
                    2,
                ).to(conv_states.dtype),
            )
            ngram_pool.index_copy_(
                0,
                slots,
                torch.reshape(step.input_ids.index_select(0, inf.ple_ngram_win), [split.k, -1]).to(
                    ngram_pool.dtype
                ),
            )
            if ple_ref is not None:
                conv_p, ngram_p, conv_end, ngram_end, hc_ref, normed_ref, live, of_split = ple_ref
                rows = split.stage_rows
                _inforward_report(
                    "ple conv@P (all requests)", conv_states.index_select(0, live), conv_p
                )
                _inforward_report(
                    "ple ngram@P (all requests)", ngram_pool.index_select(0, live), ngram_p
                )
                _inforward_report(
                    "ple conv@end (split)",
                    stage_conv.index_select(0, rows),
                    conv_end.index_select(0, of_split),
                )
                _inforward_report(
                    "ple ngram@end (split)",
                    stage_ngram.index_select(0, rows),
                    ngram_end.index_select(0, of_split),
                )
                _inforward_report("ple hc_out", hc_out[:num_tokens], hc_ref)
                _inforward_report("ple normed", normed_out[:num_tokens], normed_ref)
        hc_out[num_tokens:].zero_()
        normed_out[num_tokens:].zero_()

    def _ple_inforward_reference(self, hc, conv_states, ngram_pool, si_all, carry_all, qsl):
        """`_INFORWARD_VALIDATE`: the PLE block over the chunk up to P and then over the tail,
        as two context iterations run it, on copies of the step's pool rows."""
        core = self.core
        S = self._stack
        step = self._step
        split = step.inforward.split
        num_tokens = step.num_tokens
        live = split.live
        main, cu_main, of_split = _inforward_reference_layout(split, num_tokens)
        conv = conv_states.index_select(0, live).contiguous()
        ngram = ngram_pool.index_select(0, live).contiguous()
        idx = torch.arange(live.shape[0], dtype=si_all.dtype, device=si_all.device)
        next_norm = self._layers[core.ple_layer].ahc_norm
        hc_ref = torch.zeros_like(hc[:num_tokens])
        normed_ref = torch.zeros_like(hc[:num_tokens])

        def run(rows, cu, slots, carried):
            n = rows.shape[0]
            ids = step.input_ids.index_select(0, rows).contiguous()
            cu = cu.to(qsl.dtype)
            ng = ple_ngram_embed(
                ids, ngram, slots, carried, cu, core.w["ple_table"], S.ple_tscale, S.ple_mult,
                S.ple_vocab, S.ple_offs, core.eos_token_id, n,
                row_window=_ple_window(core),
            )  # fmt: skip
            if _ple_needs_sum(core):
                ng = core.tp_sum(ng)
            o, nm = ple_gate_conv_inject_norm(
                hc.index_select(0, rows).contiguous(),
                cublas_mm(ng, S.ple_key),
                cublas_mm(ng, S.ple_val),
                S.ple_nk_w,
                S.ple_nq_w,
                S.ple_nc_w,
                S.ple_conv_w,
                conv,
                slots,
                cu,
                next_norm,
                core.eps,
                core.hc_count,
                _PLE_CLAMP_MIN,
                core.ple_dilation,
                _PAD_SLOT_ID,
                n,
                carried,
            )
            hc_ref.index_copy_(0, rows, o)
            normed_ref.index_copy_(0, rows, nm)

        run(main, cu_main, idx, carry_all)
        conv_p, ngram_p = conv.clone(), ngram.clone()
        run(
            split.tail_tok,
            split.cu_tail,
            of_split.to(si_all.dtype),
            torch.ones_like(carry_all[: split.k]),
        )
        return conv_p, ngram_p, conv, ngram, hc_ref, normed_ref, live, of_split

    def _splice(self, h) -> None:
        """Break: write the step's vision rows over their placeholder rows of the token
        embedding `h` `[T, hidden]` in place, and zero rows `[n:]`."""
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

        # The activation rows are the batch's rows, or a prefill CUDA graph bucket's.
        physical_tokens = input_ids.shape[0]
        assert physical_tokens in (md.num_tokens, md.padded_num_tokens), (
            f"the engine handed {physical_tokens} activation rows for a batch its metadata "
            f"describes as {md.num_tokens} tokens (padded_num_tokens={md.padded_num_tokens})"
        )
        num_tokens = md.num_tokens
        # Under attention DP the MoE blocks pad the `T` rows to the group's largest count. A
        # prefill CUDA graph pads every rank to one bucket, so the body it captures pads
        # nothing; that zero is what makes the count a per-capture constant.
        dp_pad_rows = core._dp_pad_rows(md, physical_tokens)
        assert physical_tokens == num_tokens or dp_pad_rows == 0, (
            f"a prefill CUDA graph bucket of {physical_tokens} rows with {dp_pad_rows} attention-DP "
            "padding rows: the graph would freeze the group's row counts"
        )
        batch = md.num_seqs
        num_prefills = md.num_contexts
        num_decodes = batch - num_prefills
        num_prefill_tokens = md.num_ctx_tokens
        num_decode_tokens = num_tokens - num_prefill_tokens
        if not self._gdn_inplace_reported and num_prefill_tokens <= _GDN_INPLACE_MAX_TOKENS:
            # The recurrence of a context this short runs in place over the pool rows
            # (`flashinfer_chunk_gated_delta_rule`'s short path).
            self._gdn_inplace_reported = True
            print(
                f"modeling_v2 qwen4_exp default target: rank {core.mapping.rank} runs short GDN "
                "context chunks in place over the recurrent pool"
            )
        # `DP_GDN_CONV_LEAN_ENV`: a step with context rows keeps the stock conv pair.
        _gdn_step_clear(core)
        # Rows per generation sequence: 1 without speculation, else `runtime_draft_len + 1`.
        gen_p, is_verify = _verify_rows(core, md, spec_metadata, num_decodes, num_decode_tokens)
        attn_args["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        hi_all = mm.has_initial_states[:batch]
        si_p, si_d = torch.split(si_all, [num_prefills, num_decodes], 0)
        # The single-token decode kernel takes its slot vector through a 16-byte-aligned
        # pointer; the generation half of the int32 slot vector starts `num_prefills` rows
        # in, so one copy per step makes it aligned for every layer.
        si_d = torch.clone(si_d)
        hi_p = hi_all[:num_prefills]
        # `query_start_loc` covers the context prefix; `_long` the whole batch, and its prefix
        # the context rows' chunked recurrence when generation rows run beside them.
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
            _verify_state(core, mm, num_decodes, num_prefills) if is_verify else None,
        )

        # PLE carry flags must cover the whole batch, but `has_initial_states` is filled for
        # the context prefix only; every generation sequence carries, so pad with True.
        ple_carry = nn.functional.pad(hi_p, [0, num_decodes], mode="constant", value=1.0)
        # Under speculation `query_start_loc_long` has one row per generation sequence, so
        # with `gen_p > 1` both PLE stateful calls would be mis-segmented: rebuilt from
        # `seq_lens_cuda`, which holds the true counts during the trunk forward.
        ple_qsl = (
            qsl_long
            if gen_p == 1
            else cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        )
        pools = core._bound.pools
        ple_state = (pools.ple_state[0], pools.ple_state[1], si_all, ple_carry, ple_qsl)

        # Dense/sparse decision once per forward, from a host-side metadata read: a
        # per-capture constant.
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
        )

        # A verify step's generation rows write the PLE pools and the QSA side caches for
        # every draft row, accepted or not; `SpeculativeRollback` keeps what its commit needs
        # to undo the rejected ones. Reopened on every forward, so a step whose commit never
        # ran leaves nothing for the next one. Context rows are never staged.
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
        fp8_groups = None
        if (
            sparse
            and core.quant_mode == QM_FP8_KV_CACHE
            and num_prefill_tokens >= _ROWTILE_MIN_ROWS
        ):
            fp8_pages = _fp8_context_pages(md, md.tokens_per_block)
            fp8_groups = _fp8_context_groups(core, md, md.tokens_per_block, fp8_pages)
        nvfp4_context = None
        if (
            sparse
            and core.quant_mode == QM_NVFP4_KV_CACHE
            and num_prefill_tokens > _NVFP4_ROWTILE_MIN_ROWS
        ):
            nvfp4_context = _nvfp4_context_plan(
                core, md, qsa[0], md.qsa_block_table, md.tokens_per_block
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
            fp8_groups=fp8_groups,
            nvfp4_context=nvfp4_context,
            live_cols=_live_score_columns(core, md, batch),
            inforward=_inforward_split(md, mm, si_all, hi_all, pools),
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
        replays for every step it pads to that bucket without running this Python. So nothing
        here reads the step: every op is row-wise and sized by `T` alone, and what depends on
        the step's batch runs in one of the four breaks (`_splice`, `_ple_block`,
        `_attention_core`, `_gdn_core`), each writing rows `[:n]` of a fixed output allocated
        here and zeroing rows `[n:]`. The tensor-parallel sums run on all `T` rows, which
        every rank holds alike; under attention DP the MoE round trip pads them by the step's
        `dp_pad_rows`, zero whenever a graph captures this body."""
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

        # Scratch shared by every layer that uses it; each break rewrites its whole output.
        proj_qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)
        gdn_out = torch.empty(
            [1, num_tokens, core.num_v_heads, core.head_v_dim], dtype=dt, device=dev
        )
        ple_hc = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)
        ple_normed = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)

        # The embedding repeated across the hc_count residual streams.
        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        layers, S = self._layers, self._stack
        pad_rows = self._step.dp_pad_rows
        # A host decision on `T` alone (each captured bucket takes one branch): at one rank, from
        # `_HC_RSTD_MIN_ROWS` rows, every site but the PLE layer's attention-side one is
        # `qwen4_exp_hc_inject_rstd` and the rstd mixer.
        by_rstd = core.hc_fast_sites and num_tokens >= _HC_RSTD_MIN_ROWS
        normed = rstd = None
        if by_rstd:
            _, rstd = qwen4_exp_hc_inject_rstd(hc, None, None, core.eps, hc_count)
        else:
            _, normed = qwen4_exp_hc_inject_norm(
                hc, None, None, layers[0].ahc_norm, core.eps, hc_count
            )
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                eager_on_graph(hc, ple_hc, ple_normed, inner=self._ple_block)
                hc, normed = ple_hc, ple_normed
                # The PLE kernel stores this site's normed stream: the stored-normed mixer
                # reads it (cheaper than a second pass for its rstd).
                mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            elif by_rstd:
                mixed, inject = _hc_mixer_rstd(
                    core, hc, rstd, L.ahc_norm, L.ahc_norm_w1, L.ahc_first, L.ahc_up, True
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
                mixed, inject, fp4 = _hc_moe_mixer_rstd(core, hc, rstd, L)
            else:
                hc, normed = qwen4_exp_hc_inject_norm(
                    hc, block_out, inject, L.mhc_norm, core.eps, hc_count
                )
                mixed, inject, fp4 = _hc_moe_mixer(core, normed, L)
            next_norm = layers[i + 1].ahc_norm if i + 1 < len(layers) else S.fhc_norm
            # The MoE block output is formed inside the injection: routed + sigmoid(gate) *
            # shared, rounded once.
            routed, gate_logits, shared = _moe_block(core, mixed, L, pad_rows, fp4)
            if by_rstd:
                hc, rstd = qwen4_exp_hc_inject_rstd(
                    hc, routed, inject, core.eps, hc_count, shared, gate_logits
                )
            else:
                hc, normed = qwen4_exp_hc_inject_norm(
                    hc, routed, inject, next_norm, core.eps, hc_count, shared, gate_logits
                )
            del routed, gate_logits, shared, fp4
        # The stack-final mixer with no injection weight is the final norm.
        if by_rstd:
            final, _ = _hc_mixer_rstd(
                core, hc, rstd, S.fhc_norm, S.fhc_norm_w1, S.fhc_first, S.fhc_up, False
            )
        else:
            final, _ = _hc_mixer(core, normed, S.fhc_first, S.fhc_up, False)
        if not core.mtp_enabled:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the `[T, stream]` stream
        # it was collapsed from. They leave packed in one tensor, because a breakable prefill
        # graph's body returns one tensor (`split_spec_output`).
        return torch.cat([final, hc], -1)

    # ------------------------------------------------------------- the draft ---

    def _capture_selection(self, md, topk_indices, visible_blocks, compact=False) -> None:
        """IndexShare: stash, per request, the selection of the row the draft loop continues
        from, with that row's complete-group count, in the draft layer's row of the shared
        buffers: a context request's last row of this chunk, and a generation request's last
        accepted row (`num_accepted - 1` past its first). A `compact` selection holds each
        context request's last row, then every generation row (`mtp_forward`'s `keep`)."""
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
        if compact:
            num_contexts = md.num_contexts
            rows[:num_contexts] = torch.arange(num_contexts, device=rows.device)
            rows[num_contexts:] -= md.num_ctx_tokens - num_contexts
        slot = self.core._draft_bound.share_slot
        md.qsa_shared_topk_indices[slot, :batch].copy_(torch.index_select(topk_indices, 0, rows))
        md.qsa_shared_topk_visible_blocks[slot, :batch].copy_(
            torch.index_select(visible_blocks, 0, rows)
        )
        # Host-side readiness, recorded in program order.
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
        """One draft step whose metadata still carries context rows: the draft loop's first
        (draft-extend) step of a batch with context rows, eager. Every row writes the layer's
        KV and index planes; the worker gathers one row per request from the result. Under
        attention TP only `keep` -- each context request's last row, then every generation
        row -- goes on past those writes (the query side included on the sparse path; the
        dense call's metadata describes every row), and the other returned rows keep the
        layer's input stream. Under attention DP every row runs the whole layer.

        The draft layer's attention is the trunk prefill target's: the context rows score
        as query tiles and bound their sparse scan by their positions, the generation rows
        (`p` a request) score row by row. `all_rank_num_tokens` is the group's row counts of
        this draft step, which pad the MoE round trip under attention DP (`_draft_pad_rows`)."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, "the trunk's cache-manager bind has not run"
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows and hidden_states.shape[1] == core.stream, (
            f"the draft step got {input_ids.shape[0]} token ids and {tuple(hidden_states.shape)} "
            f"hidden states; the MTP module reads one {core.stream}-wide stream row per token"
        )
        # Projected on every call: the worker rewrites `attn_metadata` in place between
        # draft steps.
        step = self.step_args(md)
        num_ctx = md.num_ctx_tokens
        gen_seqs = md.num_seqs - md.num_contexts
        p, _ = _verify_rows(core, md, None, gen_seqs, rows - num_ctx)
        step["predicted_tokens_per_seq"] = p
        draft, layer_idx = core._draft_step_bind(md)
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        pad_rows = _draft_pad_rows(core, all_rank_num_tokens, rows)
        # The trunk's dense/sparse decision from the same host buffer: every draft step of
        # one engine forward takes the branch its trunk took.
        batch = md.num_seqs
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold
        # IndexShare: the only draft step with context rows is the loop's first, which always
        # runs the indexer and, with sharing on, captures the selection the later steps reuse.
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
        qkv, gate, k_dst, v_dst = _mtp_qkv(core, M, mixed, pos3)
        q_index = _mtp_index_branch(
            core, M, mixed, draft, (req_idx, logical, block_table, tokens_per_block, pos, coords)
        )
        keep = None
        if core.dp_size == 1 and num_ctx > md.num_contexts:
            ctx_lens = md.seq_lens_cuda[: md.num_contexts].to(torch.long)
            keep = torch.cat(
                [
                    torch.cumsum(ctx_lens, 0) - 1,
                    torch.arange(num_ctx, rows, dtype=torch.long, device=x.device),
                ]
            )
        if sparse and keep is not None:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            q_rows = keep.shape[0]
            q_req, q_logical = req_idx[keep], logical[keep]
            visible_blocks = md.qsa_visible_blocks[:rows][keep]
            seq_lens_q = md.qsa_sequence_lengths[:rows][keep]
            row_starts = md.qsa_topk_row_starts[:rows][keep]
            topk_indices = md.qsa_topk_indices[:q_rows]
            scores = _index_scores(
                core,
                q=q_index[keep],
                index_cache=draft.index_cache,
                block_table=block_table,
                query_positions=q_logical,
                request_indices=q_req,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                # One row per context request: no run of rows to share a key gather.
                context_rows=False,
            )
            selected = torch.empty(
                [q_rows, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=qkv.device,
            )
            _selection(
                core,
                scores,
                visible_blocks,
                q_logical,
                seq_lens_q,
                topk_indices,
                row_starts,
                selected,
                _live_score_columns(core, md, batch),
                draft=True,
            )
            if share:
                self._capture_selection(md, topk_indices, visible_blocks, compact=True)
            q_att = _packed_q(core, qkv)[keep]
            attn = torch.empty([q_rows, core.q_width], dtype=qkv.dtype, device=qkv.device)
            if kv.k_scale is not None and core.one_rank:
                # One row per context request, then `p` per generation request: the split-K
                # kernel for the first, the verify kernel for the rest where it serves.
                _sparse_gqa_nvfp4_context(
                    core,
                    kv,
                    None,
                    q_att,
                    block_table,
                    selected,
                    q_req,
                    q_logical,
                    tokens_per_block,
                    md.num_contexts,
                    attn,
                    p,
                )
            else:
                out = _sparse_gqa(
                    core, kv, q_att, block_table, selected, q_req, tokens_per_block, q_logical
                )
                attn.copy_(torch.reshape(out, [q_rows, core.q_width]))
            gate = torch.reshape(gate[keep], [q_rows, core.q_width])
        elif sparse:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            seq_lens_q = md.qsa_sequence_lengths[:rows]
            row_starts = md.qsa_topk_row_starts[:rows]
            parts = [(0, num_ctx, True)]
            if num_ctx < rows:
                parts.append((num_ctx, rows, False))
            selected = torch.empty(
                [rows, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=qkv.device,
            )
            _score_and_select(
                core,
                draft.index_cache,
                q_index,
                block_table,
                logical,
                req_idx,
                tokens_per_block,
                parts,
                visible_blocks,
                seq_lens_q,
                topk_indices,
                row_starts,
                selected,
                _live_score_columns(core, md, batch),
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
        fused_sigmoid_mul_inplace(attn, gate)
        out = x
        if keep is not None:
            # The sparse path attended the kept rows only; the dense call wrote every row.
            if not sparse:
                attn = attn[keep]
            x, inject = x[keep], inject[keep]
        x = _mtp_tail(core, M, x, attn, inject, pad_rows)
        return x if keep is None else out.index_copy_(0, keep, x)


# ------------------------------------------------------------------- decode ---


class DecodeTarget:
    """Generation-only batches (no context rows): one token per sequence, every op inside
    the captured decode graph."""

    def __init__(self, core: Qwen4ExpDefaultModelingV2) -> None:
        self.core = core
        # See PrefillTarget.__init__.
        self._contract_pending = step_contract_enabled()
        self._kv_probe = _KvRangeProbe.for_target(core, self._contract_pending)
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)
        # The trunk step's positions buffer (`_position_key`) and its `pos3`, which the first
        # draft step of the same forward reads instead of a copy of its own.
        self._trunk_pos3: tuple | None = None

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
        return _step_args(
            md,
            num_contexts=0,
            num_ctx_tokens=0,
            paged_context=self.core.quant_mode == QM_NVFP4_KV_CACHE,
        )

    # --------------------------------------------------------------- blocks ---

    def _attention_prologue(self, proj, q_index, index_view, index_cache, L, qsa, side, stage):
        """At one rank, without the fused prologue: the norm + rope of both planes on this stream
        and the QSA index branch (the verify step's staging, the raw-key store, then compression)
        on the side stream, which reads only the projection's index-key columns and the index
        pools; the attention core joins it before reading either."""
        core = self.core
        pools = core._bound.pools
        num_tokens = proj.shape[0]
        req_idx, logical, _, _, _, _, block_table, tokens_per_block, pos3, _, coords, _ = qsa
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
            index_mrope_section1=0,
            index_mrope_section2=0,
        )
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
                core.spec_rollback.stage_index(pools.position_cache)
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
            _index_anchors_fp8(
                core,
                index_cache,
                block_table,
                req_idx,
                logical,
                tokens_per_block,
            )

    def _attention_one_rank(self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage):
        """`_attention_block` at one rank, where a layer holds every head: the input
        projection's output row is `[q | k | v | gate | index q, k | v slot]`. Over a bf16 or
        NVFP4 pool above the dense threshold everything up to the index scores is one launch,
        `one_rank_attn_decode_prologue` (over an NVFP4 pool it quantizes the K/V append; on a
        single-token step of up to `_PROLOGUE_SCORE_ROWS` rows it writes the index scores too);
        otherwise `_attention_prologue`. The sparse path reads q/k/v as views of the
        projection, selects with `qsa_topk_expand_decode` where it serves, and fuses the
        output gate into the sparse GQA; the dense path takes one compact copy of `[q | k | v]`.
        A verify step stages the index slots for `SpeculativeRollback` first."""
        core = self.core
        pools = core._bound.pools
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
            _,
            coords,
            sparse,
        ) = qsa
        num_tokens = x.shape[0]
        main = torch.cuda.current_stream()
        side = core._side_stream
        index_cache = pools.index_cache[i]
        kv = pools.kv[i]
        rows_per_request = step["predicted_tokens_per_seq"]
        fused_prologue = sparse and (kv.k_scale is not None or kv.k.dtype == x.dtype)
        if fused_prologue and stage:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                core.spec_rollback.stage_index(pools.position_cache)
        proj = _dense_mm(
            x, L.attn_in, L.attn_in8, core.decode_mm, "in", L.attn_in_g, mid=L.attn_in_m
        )
        qkv_view, gate, index_view = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        q_dst, k_dst, v_dst = _attn_in_heads(core, proj)
        q_index = _index_query(core, index_scratch)
        scores = None
        # The prologue's in-launch scores read bf16 anchors: e4m3 anchors score after it.
        if (
            fused_prologue
            and not stage
            and num_tokens <= _PROLOGUE_SCORE_ROWS
            and not getattr(core, "index_key_fp8", False)
        ):
            scores = torch.empty(
                (num_tokens, block_table.shape[1] * tokens_per_block // core.compress_ratio),
                dtype=torch.float32,
                device=proj.device,
            )
        if fused_prologue:
            if stage:
                main.wait_stream(side)
            one_rank_attn_decode_prologue(
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
                pools.position_cache,
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
                index_mrope_section1=0,
                index_mrope_section2=0,
                compress_mrope_section=_QSA_MROPE_SECTION,
                k_scale_cache=kv.k_scale,
                v_scale_cache=kv.v_scale,
                kv_scale_orig_quant=None if kv.k_scale is None else core.kv_scale_orig_quant,
                rows_per_request=rows_per_request,
                index_scores=scores,
            )
            _index_anchors_fp8(core, index_cache, block_table, req_idx, logical, tokens_per_block)
        else:
            self._attention_prologue(proj, q_index, index_view, index_cache, L, qsa, side, stage)
        if self._kv_probe is not None and not self._kv_probe.observe(i, k_dst, v_dst):
            self._kv_probe = None
        gated = False
        if sparse:
            if not fused_prologue:
                main.wait_stream(side)
                _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            if scores is None:
                scores = _index_scores(
                    core,
                    q=q_index,
                    index_cache=index_cache,
                    block_table=block_table,
                    query_positions=logical,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=False,
                    use_pdl=fused_prologue,
                    prefetch=fused_prologue and not stage and num_tokens <= _SPARSE_DECODE_MAX_ROWS,
                    rows_per_request=rows_per_request,
                )
            if _topk_expand_supports(scores):
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
                selected = torch.empty(
                    [num_tokens, core.token_topk + core.compress_ratio - 1],
                    dtype=torch.int32,
                    device=proj.device,
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
                )
            if kv.k_scale is not None:
                attn = _sparse_gqa(
                    core,
                    kv,
                    q_dst,
                    block_table,
                    selected,
                    req_idx,
                    tokens_per_block,
                    None,
                    output_gate=gate,
                    rows_per_request=rows_per_request,
                )
                gated = True
            else:
                gated = qsa_paged_sparse_gqa.fuses_output_gate(q_dst, kv.k, selected, gate)
                if (
                    gated
                    and num_tokens <= _SPARSE_DECODE_MAX_ROWS
                    and _sparse_decode_supports(q_dst, kv.k, selected)
                ):
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
            qkv.copy_(qkv_view)
            out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=pools.pool_idx[i],
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
        return _dense_mm(out, L.o, L.o8, core.decode_mm, "out", L.o_g, mid=L.o_m)

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage, q_in=None):
        """One gated GQA layer plus its QSA index branch for a generation-only batch, at this
        rank's heads. `q_in` (`DP_INPROJ_RELEASE_ENV`): the `_InProj` the input projection runs
        on behind a releasing site A.

        The four input projections are one GEMM against the fused operand; the norm + rope of
        both planes runs on it; one compact copy of its leading `[q | k | v]` feeds the dense
        op and the sparse GQA's query heads, the sparse K/V append reads k and v as views of
        it, and the index branch reads the raw key from it. The index plane is maintained on
        every step, at both ends of the threshold; above it the dense core is replaced and the
        main K/V append becomes this layer's own call; over a bf16 or e4m3 pool all of this up
        to the index scores is one launch (`qwen4_exp_attn_decode_prologue`), which writes the
        packed buffer and the index query itself. A verify step's rows (`stage`) come
        `predicted_tokens_per_seq` a request, each scored and attended on its own, and their
        index slots are staged for `SpeculativeRollback` before the store. At one rank
        `_attention_one_rank` runs instead."""
        core = self.core
        if core.one_rank:
            return self._attention_one_rank(x, i, L, qkv, index_scratch, attn_out, qsa, step, stage)
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
        if q_in is not None:
            proj = _inproj_mm(x, q_in)
        else:
            proj = _dense_mm(
                x, L.attn_in, L.attn_in8, core.decode_mm, "in", L.attn_in_g, mid=L.attn_in_m
            )
        qkv_view, gate, _ = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        pools = core._bound.pools
        # Over a bf16 or e4m3 pool the sparse path's norm + rope, packed copy, K/V append,
        # index store and compression are one launch; it writes the packed buffer itself, or
        # (`TEP_ATTN_INPLACE_ENV`, a few rows over e4m3) normalizes q and k in place instead.
        prologue = sparse and pools.kv[i].k_scale is None
        inplace = (
            prologue
            and core.tep_attn_inplace
            and num_tokens <= _TEP_ATTN_INPLACE_MAX_ROWS
            and pools.kv[i].k.dtype == torch.float8_e4m3fn
        )
        if stage:
            core.spec_rollback.stage_index(pools.position_cache)
        if inplace:
            kv = pools.kv[i]
            one_rank_attn_decode_prologue(
                proj,
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
                num_heads_q=core.heads_q,
                num_heads_k=core.heads_kv,
                head_dim=core.head_dim,
                index_heads=core.index_heads,
                index_head_dim=core.index_head_dim,
                index_q_col=core.index_col,
                rotary_dim=core.rotary_dim,
                eps=core.eps,
                mrope_section1=core.mrope_section1,
                mrope_section2=core.mrope_section2,
                index_mrope_section1=0,
                index_mrope_section2=0,
                compress_mrope_section=_QSA_MROPE_SECTION,
                rows_per_request=step["predicted_tokens_per_seq"],
            )
            _index_anchors_fp8(
                core, pools.index_cache[i], block_table, req_idx, logical, tokens_per_block
            )
        elif prologue:
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
                rows_per_request=step["predicted_tokens_per_seq"],
            )
            _index_anchors_fp8(
                core,
                pools.index_cache[i],
                block_table,
                req_idx,
                logical,
                tokens_per_block,
                use_pdl=getattr(core, "dp_scores_early", False),
            )
        else:
            _, k_dst, v_dst = _attn_in_heads(core, proj)
            _attention_rope(core, proj, index_scratch, L, pos3, pos)
            _index_plane(
                core, proj, i, L, (req_idx, logical, block_table, tokens_per_block, coords)
            )
            # The packed `[q | k | v]` as one compact copy: the dense op reads it flat, and
            # the sparse GQA reads its query heads from it.
            qkv.copy_(qkv_view)
        if self._kv_probe is not None:
            # Both branches leave the normalized, rotated K/V in the packed copy (in place in
            # the projection under `inplace`).
            if inplace:
                _, k_rows, v_rows = _attn_in_heads(core, proj)
            else:
                k_rows, v_rows = torch.split(
                    torch.reshape(qkv, [qkv.shape[0], -1, core.head_dim])[:, core.heads_q :],
                    core.heads_kv,
                    1,
                )
            if not self._kv_probe.observe(i, k_rows, v_rows):
                self._kv_probe = None

        gated = False
        if sparse:
            kv = pools.kv[i]
            p = step["predicted_tokens_per_seq"]
            if not prologue:
                _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            scores = _decode_index_scores(
                core,
                _index_query(core, index_scratch),
                core._bound.pools.index_cache[i],
                block_table,
                logical,
                req_idx,
                tokens_per_block,
                p,
                behind_prologue=prologue,
            )
            selected = torch.empty(
                [num_tokens, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=proj.device,
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
            )
            q = _attn_in_heads(core, proj)[0] if inplace else _packed_q(core, qkv)
            attn = _sparse_gqa_decode(
                core, kv, q, block_table, selected, req_idx, tokens_per_block, gate, p, trunk=True
            )
            gated = attn is not None
            if not gated:
                attn = _sparse_gqa(
                    core, kv, q, block_table, selected, req_idx, tokens_per_block, None
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
        if not gated:
            fused_sigmoid_mul_inplace(out, gate)
        return core.tp_sum(
            _out_proj(
                core,
                out,
                L.o,
                "attn_o",
                lambda: _dense_mm(out, L.o, L.o8, core.decode_mm, "out", L.o_g, mid=L.o_m),
            )
        )

    def _hc_site(
        self,
        hc,
        block_out,
        inject,
        norm,
        first,
        up,
        with_inject,
        normed=None,
        moe=None,
        release=False,
    ):
        """One hyper-connection site: the previous block's injection (none at the stack's first
        site and after the PLE add), this site's grouped norm and its mixer. Returns `(hc, mixed,
        inject)`. At one rank, and at a tep4 rank (`hc_decode_pairs`), the row count, a
        per-capture constant, picks the form: one row runs the CuTe GEMV pair, up to
        `_HC_DECODE_ROWS_MAX` rows the tcgen05 rows pair, larger batches
        `qwen4_exp_hc_inject_norm` and `_hc_mixer`, which every batch takes at the other layouts.
        A tep4 rank's site never gets the top-k pair (its MoE does not fold), so the first kernel
        forms only `routed + sigmoid(gate) * shared`. `normed`, when given, is this site's grouped
        norm of `hc` already computed (the PLE tail's, no injection pending), which only the
        inject-norm route reads (the first kernels of the other two compute it themselves).

        After a MoE block `block_out` is its routed output and `moe` the `(gate_logits, shared,
        topk_weights, topk_rows)` `_moe` left uncombined: the site's first kernel forms the
        block output itself; with the top-k pair `block_out` is the runner's permuted FC2 rows
        and that kernel also forms their finalize.

        `release` (`DP_INPROJ_RELEASE_ENV`, an attention-DP rank's trunk site A): the mixer's up
        kernel releases its dependents before its wait; only the inject-norm route takes it, the
        one an attention-DP rank runs."""
        core = self.core
        gate_logits, shared, topk_w, topk_rows = moe if moe is not None else (None,) * 4
        n_cols = core.hc_lowrank + (core.hc_count if with_inject else 0)
        fast = core.hc_decode_pairs
        assert not (release and fast), "the early release serves the inject-norm route only"
        if fast and _HC_DECODE_GEMV_MAX_ROWS < hc.shape[0] <= _HC_DECODE_ROWS_MAX:
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
                topk_w,
                topk_rows,
            )
            mixed, _, _ = qwen4_exp_hc_decode_rows_mix(low, rstd, up, hc, norm, core.hc_count)
            return hc, mixed, inject
        if fast and hc.shape[0] <= _HC_DECODE_GEMV_MAX_ROWS:
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
        mixed, inject = _hc_mixer(core, normed, first, up, with_inject, release)
        return hc, mixed, inject

    def _mtp_layer(
        self,
        embed_tokens,
        input_ids,
        hidden_states,
        md,
        step,
        draft,
        layer_idx,
        rows,
        gen_seqs,
        p,
        sparse,
        share,
        reuse,
        pos3,
        coords,
        cont=None,
    ) -> torch.Tensor:
        """The draft layer on the decode machinery, at one rank and at a tep4 rank under
        `TEP_DRAFT_LAYER_ENV` (over its shard): the front end's two projections, both
        hyper-connection sites as `_hc_site` (the front end's per-stream `x_h + x_e` is the
        first site's injection at unit weight, the attention output the second's), the attention
        as `_mtp_attention`, the MoE as `_mtp_moe`. Returns the stream with the MoE output
        injected.

        At a tep4 rank the front end's embedding branch runs on the side stream beside the
        hidden branch; the attention output is this rank's heads' partial, summed over the group
        before the second site, and the MoE's combination and injection are one launch. With
        `cont` (a draft-extend step's continuing rows, `_draft_continue`) every row writes its
        K/V and index key, and only the continuing rows run on from the index scores; the result
        holds them at their rows, the others unwritten (the worker reads only them)."""
        core = self.core
        M, G = core._mtp, core._mtp_gemv8
        gemm = core.decode_mm if md.is_cuda_graph else cublas_mm
        tp = not core.one_rank
        main = torch.cuda.current_stream()
        if tp:
            # The embedding branch reads only the ids and the table: on the side stream,
            # joined before the first site.
            side = core._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                x_e = _mtp_embed_branch(core, M, G, embed_tokens, input_ids, gemm)
        else:
            x_e = _mtp_embed_branch(core, M, G, embed_tokens, input_ids, gemm)
        _, h_n = qwen4_exp_hc_inject_norm(
            hidden_states, None, None, M.pre_h_norm, core.eps, core.hc_count
        )
        x_h = torch.reshape(
            _draft_mm(
                torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h, G.get("fc_h"), gemm
            ),
            [rows, core.stream],
        )
        if tp:
            main.wait_stream(side)
            if not torch.cuda.is_current_stream_capturing():
                # Made on the side stream, read here: an eager step's allocator must not hand
                # the block back to the side stream before this stream is done with it.
                x_e.record_stream(main)
        x, mixed, inject = self._hc_site(
            x_h, x_e, M.unit_inject[:rows], M.ahc_norm, M.ahc_first, M.ahc_up, True
        )
        attn = self._mtp_attention(
            mixed,
            md,
            step,
            draft,
            layer_idx,
            rows,
            gen_seqs,
            p,
            sparse,
            share,
            reuse,
            pos3,
            coords,
            gemm,
            cont,
        )
        if not tp:
            x, mixed, inject = self._hc_site(
                x, attn, inject, M.mhc_norm, M.mhc_first, M.mhc_up, True
            )
            return qwen4_exp_post_inject(x, self._mtp_moe(mixed), inject)
        attn = core.tp_sum(attn)
        full = x
        if cont is not None:
            kept = _draft_rows(cont.rows, {"x": x, "inject": inject})
            x, inject = kept["x"], kept["inject"]
        x, mixed, inject = self._hc_site(x, attn, inject, M.mhc_norm, M.mhc_first, M.mhc_up, True)
        out = self._mtp_moe(mixed, gemm, post=(x, inject))
        if cont is None:
            return out
        out_full = full.new_empty([rows, out.shape[1]])
        out_full.index_copy_(0, cont.rows, out)
        return out_full

    def _mtp_attention(
        self,
        x,
        md,
        step,
        draft,
        layer_idx,
        rows,
        gen_seqs,
        p,
        sparse,
        share,
        reuse,
        pos3,
        coords,
        gemm,
        cont=None,
    ) -> torch.Tensor:
        """The draft layer's gated GQA and QSA index branch over the draft manager's planes
        (`draft`), as `_attention_one_rank` runs a trunk layer: one input projection, the norm +
        rope of both planes in one launch at the module's rope base, the index store and
        compression on the side stream, the K/V append, the index scores and the one-launch
        top-k + expansion, the gated sparse GQA, the output projection. Under IndexShare a draft
        decode step (`reuse`) skips the index branch and rebuilds its selection from the
        draft-extend step's capture; the draft-extend step (`share`) captures it.

        At a tep4 rank (`TEP_DRAFT_LAYER_ENV`) everything past the K/V append runs on the
        multi-rank decode ops the generic pass runs (`_draft_sparse`), at the continuing rows
        `cont` when given, and the result is this rank's partial of the output projection."""
        core = self.core
        M = core._mtp
        dt, dev = x.dtype, x.device
        proj, q_index = self._mtp_attn_front(
            x, md, draft, rows, reuse, pos3, coords, gemm, p=0 if core.one_rank else p
        )
        qkv_view, gate, _ = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        q_dst, k_dst, v_dst = _attn_in_heads(core, proj)
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
        if not core.one_rank:
            return _draft_mm(
                self._mtp_attention_tp(
                    md,
                    step,
                    draft,
                    layer_idx,
                    proj,
                    q_index,
                    rows,
                    gen_seqs,
                    p,
                    sparse,
                    share,
                    reuse,
                    cont,
                ),
                M.o,
                None,
                gemm,
            )
        gated = False
        if sparse:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            seq_lens_q = md.qsa_sequence_lengths[:rows]
            if reuse:
                slot = draft.share_slot
                assert slot in md.qsa_shared_topk_captured_slots, (
                    "a draft decode step would reuse a selection that no draft-extend step of "
                    "this loop captured"
                )
                qsa_mtp_reuse_topk(
                    shared_indices=md.qsa_shared_topk_indices[slot],
                    shared_visible_blocks=md.qsa_shared_topk_visible_blocks[slot],
                    request_indices=req_idx,
                    visible_blocks=visible_blocks,
                    output=topk_indices,
                )
                selected = qsa_expand_block_indices(
                    topk_indices,
                    logical,
                    seq_lens_q,
                    compress_ratio=core.compress_ratio,
                    token_topk=core.token_topk,
                )
            else:
                scores = _index_scores(
                    core,
                    q=q_index,
                    index_cache=draft.index_cache,
                    block_table=block_table,
                    query_positions=logical,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=False,
                    rows_per_request=p,
                )
                if _topk_expand_supports(scores):
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
                    selected = torch.empty(
                        [rows, core.token_topk + core.compress_ratio - 1],
                        dtype=torch.int32,
                        device=dev,
                    )
                    _selection(
                        core,
                        scores,
                        visible_blocks,
                        logical,
                        seq_lens_q,
                        topk_indices,
                        md.qsa_topk_row_starts[:rows],
                        selected,
                        draft=True,
                    )
                if share:
                    self._capture_selection(md, topk_indices, visible_blocks, gen_seqs, p)
            if kv.k_scale is not None:
                attn = _sparse_gqa(
                    core,
                    kv,
                    q_dst,
                    block_table,
                    selected,
                    req_idx,
                    tokens_per_block,
                    None,
                    output_gate=gate,
                )
                gated = True
            else:
                gated = qsa_paged_sparse_gqa.fuses_output_gate(q_dst, kv.k, selected, gate)
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
            out = torch.reshape(attn, [rows, core.q_width])
        else:
            qkv = torch.empty([rows, core.qkv_width], dtype=dt, device=dev)
            qkv.copy_(qkv_view)
            out = thop_attention(
                q=qkv,
                output=torch.empty([rows, core.q_width], dtype=dt, device=dev),
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
        if not gated:
            fused_sigmoid_mul_inplace(out, gate)
        return _draft_mm(out, M.o, core._mtp_gemv8.get("o"), gemm)

    def _mtp_attn_front(self, x, md, draft, rows, reuse, pos3, coords, gemm, p=0) -> tuple:
        """The draft layer's attention input in the one-rank form: one GEMM against the fused
        `[q | k | v | gate | index | pad]` operand (`core._mtp_attn_in`), the norm + rope of both
        planes in one launch at the module's rope base, and (unless `reuse`) the raw index key's
        store and the compression on the side stream, joined before this returns. Returns
        `(proj, q_index)`: the projection row and the normalized, rotated index query.

        With `p` (`p` rows a request) at a tep4 rank under `TEP_DRAFT_PROLOGUE_ENV`, over the
        e4m3 pool and not under `reuse`, all of it and the K/V append are one launch
        (`one_rank_attn_decode_prologue`), and `self._kv_done` holds `proj` so that
        `_mtp_attention_tp` does not append K/V again."""
        core = self.core
        M = core._mtp
        dt, dev = x.dtype, x.device
        proj = _draft_mm(x, core._mtp_attn_in, core._mtp_gemv8.get("attn_in"), gemm)
        index_view = proj[:, core.qkv_width + core.q_width :]
        index_scratch = torch.empty([rows, core.index_scratch_width], dtype=dt, device=dev)
        q_index = _index_query(core, index_scratch)
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
        kv = draft.kv
        if (
            p
            and not reuse
            and getattr(core, "tep_draft_prologue", False)
            and kv.k_scale is None
            and kv.k.dtype == torch.float8_e4m3fn
            and rows % p == 0
        ):
            one_rank_attn_decode_prologue(
                proj, q_index, pos3, coords, core._mtp_cos_sin, M.q_norm, M.k_norm, M.idx_qn,
                M.idx_kn, kv.k, kv.v, draft.index_cache, draft.position_cache, req_idx, logical,
                block_table, tokens_per_block, core.compress_ratio, num_heads_q=core.heads_q,
                num_heads_k=core.heads_kv, head_dim=core.head_dim, index_heads=core.index_heads,
                index_head_dim=core.index_head_dim, index_q_col=core.qkv_width + core.q_width,
                rotary_dim=core.rotary_dim, eps=core.eps, mrope_section1=core.mrope_section1,
                mrope_section2=core.mrope_section2, index_mrope_section1=0,
                index_mrope_section2=0, compress_mrope_section=_QSA_MROPE_SECTION,
                rows_per_request=p,
            )  # fmt: skip
            _index_anchors_fp8(
                core, draft.index_cache, block_table, req_idx, logical, tokens_per_block
            )
            self._kv_done = proj
            return proj, q_index
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        qwen4_exp_qk_norm_rope(
            proj,
            q_index,
            pos3,
            core._mtp_cos_sin,
            M.q_norm,
            M.k_norm,
            M.idx_qn,
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
            index_mrope_section1=0,
            index_mrope_section2=0,
        )
        if not reuse:
            with torch.cuda.stream(side):
                token_k = torch.split(
                    torch.reshape(
                        index_view,
                        [
                            rows,
                            core.index_scratch_width // core.index_head_dim,
                            core.index_head_dim,
                        ],
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
                _index_anchors_fp8(
                    core,
                    draft.index_cache,
                    block_table,
                    req_idx,
                    logical,
                    tokens_per_block,
                )
        main.wait_stream(side)
        return proj, q_index

    def _mtp_attention_tp(
        self, md, step, draft, layer_idx, proj, q_index, rows, gen_seqs, p, sparse, share, reuse,
        cont,
    ) -> torch.Tensor:  # fmt: skip
        """`_mtp_attention` at a tep4 rank (and the generic pass at an attention-DP rank under
        `DP_MTP_ATTN_PACK_ENV`) past its norm + rope and index branch: the K/V append of every
        row, then the generic pass's sparse core (`_draft_sparse`) at the rows that go on, or the
        dense op over every row. Returns the gated `[n, q_width]` attention output of those
        rows."""
        core = self.core
        _, k_dst, v_dst = _attn_in_heads(core, proj)
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        # `[q | k | v | gate]`: the leading columns of the projection row.
        head = proj[:, : core.qkv_width + core.q_width]
        # `TEP_DRAFT_PROLOGUE_ENV`: the front appended this `proj`'s K/V already. The mark is
        # cleared on every path (a dense pass appends through `thop_attention`), so it never
        # outlives its pass.
        kv_done = getattr(self, "_kv_done", None) is proj
        self._kv_done = None
        if sparse:
            if not kv_done:
                _store_kv(
                    core, draft.kv, k_dst, v_dst, req_idx, logical, md.qsa_block_table,
                    md.tokens_per_block,
                )  # fmt: skip
            if cont is not None:
                kept = {"head": head}
                if not reuse:
                    kept["q_index"] = q_index
                kept = _draft_rows(cont.rows, kept)
                head, q_index = kept["head"], kept.get("q_index")
            n = head.shape[0]
            return self._draft_sparse(
                md,
                draft,
                torch.reshape(head[:, : core.q_width], [n, core.heads_q, core.head_dim]),
                torch.reshape(head[:, core.qkv_width :], [n, core.heads_q, core.head_dim]),
                None if reuse else q_index,
                req_idx,
                logical,
                cont,
                rows,
                gen_seqs,
                p,
                share,
                reuse,
            )
        qkv = torch.empty([rows, core.qkv_width], dtype=proj.dtype, device=proj.device)
        qkv.copy_(head[:, : core.qkv_width])
        out = thop_attention(
            q=qkv,
            output=torch.empty([rows, core.q_width], dtype=proj.dtype, device=proj.device),
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
        fused_sigmoid_mul_inplace(out, head[:, core.qkv_width :])
        return out if cont is None else torch.index_select(out, 0, cont.rows)

    def _mtp_moe(self, x, gu_mm=cublas_mm, post=None) -> torch.Tensor:
        """The draft layer's MoE at one rank: FP8 block-scale or native NVFP4 routed experts
        (FP8 can be converted with `_MTP_MOE_NVFP4_ENV`) and the bf16
        shared expert (its dense projections on `_draft_mm`), combined as `routed +
        sigmoid(gate logit) * shared`; the NVFP4 path runs the shared expert on the side
        stream.

        At a tep4 rank (`TEP_DRAFT_LAYER_ENV`): `gu_mm` is the step's decode entry for the shared
        expert's gate|up, the shared expert runs on the side stream for either expert format, the
        routed window's partial is summed over the group, and with `post = (x, inject)` the
        combination and its injection into `x` are one launch (`mtp_moe_combine_inject`, bitwise
        the two launches) and the result is the injected stream."""
        core = self.core
        M, G = core._mtp, core._mtp_gemv8

        def shared_expert():
            gu = _draft_mm(x, M.sh_gu, G.get("sh_gu"), gu_mm)
            shared = _draft_mm(
                silu_and_mul_rows(gu, core.shared_inter), M.sh_dn, G.get("sh_dn"), cublas_mm
            )
            return shared, cublas_mm(x, M.sh_gate)

        if core.one_rank and M.moe_gscale is None:
            routed = _mtp_moe_block(core, M, x, routed_only=True)
            shared, gate_logits = shared_expert()
            return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared, gate_logits = shared_expert()
        if core.one_rank:
            routed = _mtp_routed_nvfp4(core, M, x)
        else:
            # `_mtp_moe_block` sums the window's partial over the group for either format.
            routed = _mtp_moe_block(core, M, x, routed_only=True)
        main.wait_stream(side)
        if post is None:
            return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)
        x_post, inject = post
        if _combine_inject_fits(x_post, routed, shared, inject):
            return mtp_moe_combine_inject(x_post, routed, gate_logits, shared, inject)
        return qwen4_exp_post_inject(
            x_post, fused_sigmoid_gate_mul_add(routed, gate_logits, shared), inject
        )

    def _hc_moe_site(self, hc, block_out, inject, L: _LayerWeights):
        """`_hc_site` at an MoE site (always with injection). Under attention TP (or at one
        rank) it also returns the routed experts' NVFP4 input (linear scales at the layer's FC1
        input global scale), written by the mixer's own last launch; under attention DP the
        window quantizes the group's gathered rows itself and `fp4` is None. Returns `(hc,
        mixed, inject, fp4)`."""
        core = self.core
        if type(block_out) is _SiteBFolded:
            # `DP_OUT_PROJ_INJECT_ENV`: the block's output projection injected already.
            return _siteb_site(core, block_out, L)
        if core.dp_size > 1:
            hc, mixed, inject = self._hc_site(
                hc, block_out, inject, L.mhc_norm, L.mhc_first, L.mhc_up, True
            )
            return hc, mixed, inject, None
        n_cols = core.hc_lowrank + core.hc_count
        fast = core.hc_decode_pairs
        if fast and _HC_DECODE_GEMV_MAX_ROWS < hc.shape[0] <= _HC_DECODE_ROWS_MAX:
            hc, low, inject, rstd = qwen4_exp_hc_decode_rows_first(
                hc, block_out, inject, L.mhc_norm, L.mhc_first, core.hc_count, n_cols, core.eps
            )
            mixed, xq, xsf = qwen4_exp_hc_decode_rows_mix(
                low, rstd, L.mhc_up, hc, L.mhc_norm, core.hc_count, L.moe_gscale
            )
            return hc, mixed, inject, (xq, xsf)
        if fast and hc.shape[0] <= _HC_DECODE_GEMV_MAX_ROWS:
            hc, part, ss = qwen4_exp_hc_decode_first_cute(
                hc, block_out, inject, L.mhc_norm, L.mhc_first, core.hc_count, n_cols
            )
            mixed, inject, xq, xsf = qwen4_exp_hc_decode_mix_fp4_cute(
                part, ss, L.mhc_up, hc, L.mhc_norm, core.eps, core.hc_count, True, L.moe_gscale
            )
            return hc, mixed, inject, (xq, xsf)
        hc, normed = qwen4_exp_hc_inject_norm(
            hc, block_out, inject, L.mhc_norm, core.eps, core.hc_count
        )
        mixed, inject, fp4 = _hc_moe_mixer(core, normed, L)
        return hc, mixed, inject, fp4

    def _moe(
        self, x, L: _LayerWeights, pad_rows: int, fp4, fold: bool, captured: bool, rs_next=None
    ) -> tuple:
        """One decode MoE block, left uncombined for the next site: `(routed, (gate_logits,
        shared, topk_weights, topk_rows))`. With `fold` (one rank, where the window is the whole
        layer and needs no sum) the runner skips its finalize: `routed` is its permuted FC2 rows
        and the top-k pair its routing weights and index map, which the next site's first kernel
        gathers; otherwise the pair is None and `routed` the finalized rows (`_moe_block`).
        `captured` is the step's CUDA-graph state, which picks the router (`_decode_router_mid`).
        `rs_next` (`_dp_rs_inject_next`, never with `fold`): the block injects and normalizes for
        the next site itself where it can, and this then returns `(_RsInjected, None)`."""
        core = self.core
        router_mm = _decode_router_mid(core, L, captured)
        if not fold:
            routed, gate_logits, shared = _moe_block(
                core, x, L, pad_rows, fp4, router_mm, core.decode_mm, rs_next
            )
            if type(routed) is _RsInjected:
                return routed, None
            return routed, (gate_logits, shared, None, None)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        # From `_ROUTER_FIRST_MIN_ROWS` rows the router launches before the shared expert (the
        # fork above precedes it, so the side stream does not wait for it).
        router_first = x.shape[0] >= _ROUTER_FIRST_MIN_ROWS
        if router_first:
            logits = router_mm(x, L.router)
        with torch.cuda.stream(side):
            shared, gate_logits = _shared_expert(core, x, L, core.decode_mm)
        if not router_first:
            logits = router_mm(x, L.router)
        xq, xsf = fp4
        rows, topk_w, topk_rows = fp4_block_scale_moe_renorm_permuted(
            logits,
            xq,
            torch.reshape(xsf, [-1]).view(torch.float8_e4m3fn),
            L.fc1_w,
            L.fc1_s,
            L.fc2_w,
            L.fc2_s,
            L.out1_scale,
            L.out1_gate_scale,
            L.out2_scale,
            core.num_experts,
            core.topk,
            core.expert_inter,
        )
        main.wait_stream(side)
        return rows, (gate_logits, shared, topk_w, topk_rows)

    def _gdn_block(self, x, i, L, si_all, verify, gen_p, captured=False, q_in=None) -> torch.Tensor:
        """One Gated-DeltaNet layer over a generation-only batch, one token per sequence, at
        this rank's heads: one GEMM for all four input projections (q|k|v, z, a and b are
        column views of its output), the conv step over the q|k|v columns in place, then the
        recurrence and the gated norm as one streaming launch that computes each (row,
        v-head) on its own, so over a bf16 pool a row's output and state do not depend on its
        batch. The output projection's per-rank partial is summed over the TP group.

        A verify step (`verify`, `gen_p` rows a sequence) takes `_gdn_verify` and the gated
        norm epilogue instead.

        With the in_proj's NVFP4 copy (`_GDN_INPROJ_NVFP4_ENV`) every step of up to
        `_INPROJ_NVFP4_MAX_ROWS` rows reads it; with its packed FP8 copy (`_DENSE_FP8_ENV`) the
        rows `_dense_mm` gives it. In a captured step (`captured`) those GEMVs run the conv step
        as their epilogue -- a single-token step's and, up to `_VERIFY_CONV_FUSED_MAX_ROWS`
        rows, a verify step's with its window snapshots and the pool advance -- reading the
        pool, taps and slots before their PDL wait, bitwise the projection + conv they
        replace.

        `q_in` (`DP_INPROJ_RELEASE_ENV`): the `_InProj` the input projection runs on behind a
        releasing site A (attention DP; the layer has no other decode copy)."""
        core = self.core
        pools = core._bound.pools
        num_tokens = x.shape[0]
        is_verify = verify is not None
        nv = (
            L.in_proj_nv
            if L.in_proj_nv is not None and num_tokens <= _INPROJ_NVFP4_MAX_ROWS and _gemv_ready(x)
            else None
        )
        if q_in is not None:
            assert nv is None and L.in_proj_g is None, f"layer {i}: another in_proj copy"
        nv_conv = nv is not None and captured and not is_verify
        fp8_conv = (
            not nv_conv
            and captured
            and not is_verify
            and L.in_proj_g is not None
            and _gemv_ready(x)
            and num_tokens <= _FP8_CONV_FUSED_MAX_ROWS
        )
        verify_fused = (
            captured
            and is_verify
            # `DP_GDN_CONV_LEAN_ENV`: a lean step's conv is `gdn_conv_lean` in every layer.
            and not _gdn_lean_now(core)
            and (L.in_proj_g is not None or nv is not None)
            and _gemv_ready(x)
            and num_tokens <= min(_FP8_GEMV_ROWS["in"][1], _VERIFY_CONV_FUSED_MAX_ROWS)
        )
        if verify_fused:
            verify_op = (
                fp8_gemv_conv1d_verify_op if nv is None else nvfp4_split_gemv_conv1d_verify_op
            )
            mq_out, zab = verify_op(
                x,
                L.in_proj_g if nv is None else nv,
                pools.conv[i],
                L.conv_w,
                si_all,
                core._bound.int_conv[i],
                verify[0],
                core.conv_dim,
                gen_p,
                _PAD_SLOT_ID,
            )
            z, a, b = torch.split(zab, core.in_proj_split[1:], 1)
            out = _gdn_verify(
                core, None, a, b, i, L, si_all, num_tokens // gen_p, gen_p, verify, mq_out
            )
            return self._gdn_epilogue(out, z, L, num_tokens)
        if nv_conv:
            proj = nvfp4_split_gemv_conv1d_op(
                x, nv, pools.conv[i], L.conv_w, si_all, core.conv_dim, _PAD_SLOT_ID
            )
        elif fp8_conv:
            proj = fp8_gemv_conv1d_op(
                x, L.in_proj_g, pools.conv[i], L.conv_w, si_all, core.conv_dim, _PAD_SLOT_ID
            )
        elif nv is not None:
            proj = nvfp4_split_gemv_op(x, nv)
        elif (
            q_in is not None and is_verify and _gdn_inconv_now(core) and si_all.dtype == torch.int32
        ):
            # `DP_GDN_INPROJ_CONV_BF16_ENV`: the lean conv as the release GEMM's epilogue (one
            # launch for the two); z | a | b are column views of its output.
            mq_out, proj = mid_mm_conv_bf16(
                x,
                q_in.w,
                q_in.plan,
                L.conv_w,
                pools.conv[i],
                si_all,
                verify[0],
                core._gdn_xsave[i],
                gen_p,
                mpad=q_in.mpad,
                pad_slot_id=_PAD_SLOT_ID,
            )
            z, a, b = torch.split(proj, core.in_proj_split, 1)[1:]
            out = _gdn_verify(
                core, None, a, b, i, L, si_all, num_tokens // gen_p, gen_p, verify, mq_out, z=z
            )
            return self._gdn_epilogue(out, z, L, num_tokens, _gdn_rnorm_now(core))
        elif q_in is not None:
            proj = _inproj_mm(x, q_in)
        else:
            # FlashInfer's low-M kernels at the rows its table names (a verify step's few rows
            # a request), cuBLAS past them; the dense FP8 copies where `_dense_mm` takes them.
            proj = _dense_mm(
                x,
                L.in_proj,
                L.in_proj8,
                core.decode_in_mm,
                "in",
                L.in_proj_g,
                mid=L.in_proj_m,
            )
        if is_verify:
            mixed_qkv, z, a, b = torch.split(proj, core.in_proj_split, 1)
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
                conv_wait=_tep_conv_wait(core, num_tokens),
                z=z,
            )
            return self._gdn_epilogue(out, z, L, num_tokens, _gdn_rnorm_now(core))
        if not (nv_conv or fp8_conv):
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
        if (
            captured
            and core.one_rank
            and pools.ssm[i].dtype == torch.float32
            and num_tokens <= _GDN_DECODE_FUSED_MAX_ROWS
        ):
            # At one rank over an fp32 pool, the recurrence and the gated norm as one launch
            # that reads this layer's pool rows before its PDL wait: in a captured
            # single-token step nothing ahead of it writes them.
            o = gdn_decode_gated_norm_pdl(
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
            return _dense_mm(o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m)
        # It computes both gates and the q/k L2 norm itself, at the softplus pair (1, 20) and
        # epsilon 1e-6 this target's constants name.
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
        return core.tp_sum(
            _out_proj(
                core,
                o,
                L.out,
                "gdn_out",
                lambda: _dense_mm(o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m),
            )
        )

    def _gdn_epilogue(self, out, z, L, num_tokens, normed=False) -> torch.Tensor:
        """A verify step's gated RMS norm (sigmoid gate, plain non-gemma weight) and the
        output projection, whose per-rank partial is summed over the TP group. `normed`: the
        replay already wrote the gated norm's output (`DP_GDN_REPLAY_NORM_BF16_ENV`)."""
        core = self.core
        if normed:
            o = torch.reshape(out, [num_tokens, core.value_dim])
            return core.tp_sum(
                _out_proj(
                    core,
                    o,
                    L.out,
                    "gdn_out",
                    lambda: _dense_mm(
                        o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m
                    ),
                )
            )
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
            # `DP_GDN_CONV_LEAN_ENV`: a PDL dependent of the replay whatever the grid.
            **({"pdl": True} if _gdn_lean_now(core) else {}),
        )
        o = torch.reshape(o, [num_tokens, core.value_dim])
        return core.tp_sum(
            _out_proj(
                core,
                o,
                L.out,
                "gdn_out",
                lambda: _dense_mm(o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m),
            )
        )

    def _ple_front(self, input_ids, ple_state, stage) -> tuple:
        """The PLE block's front, which reads only the step's ids, the PLE pools and weights:
        the rollback's PLE staging (a verify step), the n-gram rows (summed over attention TP
        where the table is split), the key and value projections. Returns `(key, value)`."""
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
            row_window=_ple_window(core),
        )
        if _ple_needs_sum(core):
            ngram = core.tp_sum(ngram)
        # A tep4 rank under `TEP_PROJ_ENV`: 9-16 rows on the split-K low-M kernel.
        ple_mm = (
            _tep_wide_mm
            if getattr(core, "tep_proj", False)
            and _TEP_PLE_WIDE_MIN_ROWS <= ngram.shape[0] <= _LOW_M_WIDE_MAX_ROWS
            else cublas_mm
        )
        return ple_mm(ngram, S.ple_key), ple_mm(ngram, S.ple_val)

    def _ple_fork(self, input_ids, ple_state, stage) -> tuple:
        """`_ple_front` on the PLE stream (`core._ple_stream`), forked from this stream at the
        start of the forward, so that it runs beside the layers before the PLE layer; the PLE
        layer joins it (`_ple_block`). The same kernels on the same operands: no collective runs
        there (`_ple_forks`). An eager step's key and value are marked for this stream, which
        reads them after the join, before the allocator may reuse them."""
        main = torch.cuda.current_stream()
        fork = self.core._ple_stream
        fork.wait_stream(main)
        with torch.cuda.stream(fork):
            key, value = self._ple_front(input_ids, ple_state, stage)
        if not torch.cuda.is_current_stream_capturing():
            key.record_stream(main)
            value.record_stream(main)
        return key, value

    def _ple_forks(self) -> bool:
        """Whether a decode forward forks its PLE front (`_ple_fork`): at a tep4 rank under
        `TEP_FAST_DECODE_ENV` with layers before the PLE layer, and no TP sum in the front (two
        streams must not issue collectives of one group)."""
        core = self.core
        return (
            core.tep_fast
            and core.ple_layer > 0
            and getattr(core, "_ple_stream", None) is not None
            and not _ple_needs_sum(core)
        )

    def _ple_block(self, hc, input_ids, ple_state, next_norm, stage, front=None) -> tuple:
        """The n-gram side path on its decoder layer, added into the stream and normalized
        for the layer's attention-side site. Returns `(hc, normed)`. The same steps as the
        prefill target's `_ple_block`, over up to `rows` rows a sequence; every generation
        sequence carries its history by construction, so no carry flags are passed. A verify
        step (`stage`) stages what `SpeculativeRollback` rewrites the two pools from: both
        pools' pre-step rows before either launch publishes, and the conv input the second
        launch writes out. `front` is `_ple_fork`'s `(key, value)` when the front was forked,
        joined here; else the front runs here (`_ple_front`)."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, qsl, rows = ple_state
        if front is None:
            key, value = self._ple_front(input_ids, ple_state, stage)
        else:
            torch.cuda.current_stream().wait_stream(core._ple_stream)
            key, value = front
        conv_input = torch.empty_like(hc) if stage else None
        out = ple_gate_conv_inject_norm(
            hc,
            key,
            value,
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
        )
        if stage:
            core.spec_rollback.stage_ple_conv_input(conv_input)
        return out

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
        # `coords` is the `[N, 3]` view of the engine's positions the position plane stores;
        # `pos3` the contiguous `[3, N]` copy, built below (by `verify_step_setup` on a
        # captured verify step).
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)

        hc_count = core.hc_count
        if input_ids.dim() == 1 and input_ids.is_contiguous():
            # The embedding and its copy into every hyper-connection stream in one launch.
            hc = embed_rows(input_ids, core.w["embed"], hc_count, False)
        else:
            hc = torch.cat([nn.functional.embedding(input_ids, core.w["embed"])] * hc_count, -1)
        num_tokens = hc.shape[0]
        dt = hc.dtype
        dev = hc.device
        batch = md.num_seqs
        # Decode CUDA graphs pad with real dummy requests, so they are in the metadata.
        assert num_tokens == md.num_tokens, (
            f"the engine handed {num_tokens} activation rows for a generation batch its "
            f"metadata describes as {md.num_tokens} tokens"
        )
        step["predicted_tokens_per_seq"] = 1
        # Zero rows under a decode CUDA graph, which pads every attention-DP rank to the
        # captured batch.
        pad_rows = core._dp_pad_rows(md, num_tokens)
        # Rows per generation sequence: 1 without speculation, else `runtime_draft_len + 1`.
        gen_p, is_verify = _verify_rows(core, md, spec_metadata, batch, num_tokens)
        step["predicted_tokens_per_seq"] = gen_p
        si_all = mm.state_indices[:batch]
        verify = _verify_state(core, mm, batch) if is_verify else None
        # `DP_GDN_CONV_LEAN_ENV`: what the previous trunk step decided is not this step's; a
        # generation-only verify step runs lean in all its GDN layers or in none.
        _gdn_step_clear(core)
        if is_verify and getattr(core, "_gdn_lean_on", False):
            _gdn_lean_step(core, batch, num_tokens, gen_p)
        pools = core._bound.pools
        # The corrected positions the engine's decode prologue wrote for this step
        # (`SpecStepMetadataPrologue`), consumed here whichever branch reads them.
        step_pos3 = take_step_positions(md, position_ids)
        # A captured verify step builds `pos3`, the PLE segmentation vector and the rollback's
        # staging in one launch (`verify_step_setup`).
        rollback = core.spec_rollback
        fused_setup = (
            is_verify
            and md.is_cuda_graph
            and batch < _VERIFY_SETUP_MAX_SEQS
            and position_ids.stride(-1) == 1
            and rollback.index_planes is not None
        )
        if fused_setup:
            pos3, ple_qsl, staged_slots, staged_pages, staged_within, staged_index = (
                verify_step_setup(
                    position_ids,
                    md.seq_lens_cuda[:batch],
                    si_all,
                    md.qsa_req_idx_per_token[:num_tokens],
                    md.qsa_logical_positions[:num_tokens],
                    md.qsa_block_table,
                    md.tokens_per_block,
                    *rollback.index_planes,
                )
            )
        else:
            pos3 = step_pos3 if step_pos3 is not None else _copy_positions(position_ids)
            # Under speculation `query_start_loc_long` has one row per generation sequence:
            # with `gen_p > 1` the PLE segmentation is rebuilt from `seq_lens_cuda`.
            ple_qsl = (
                mm.query_start_loc_long
                if gen_p == 1
                else cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
            )
        pos = pos3[0]
        self._trunk_pos3 = (_position_key(position_ids), pos3)
        ple_state = (pools.ple_state[0], pools.ple_state[1], si_all, ple_qsl, gen_p)

        # Staging for the speculative rollback (see PrefillTarget.prepare); every row here is
        # a generation row. A captured step stages and commits the PLE pools in one launch
        # each.
        if rollback is not None:
            rollback.discard()
        if fused_setup:
            rollback.begin_staged(
                num_gens=batch,
                rows_per_gen=gen_p,
                slots=staged_slots,
                pages=staged_pages,
                within=staged_within,
                index_all=staged_index,
            )
        elif is_verify:
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
            pos,
            coords,
            sparse,
        )

        # Scratch shared by all attention layers: thop_attention overwrites rows
        # [:num_tokens] of `output` in place, and the packed buffers are fully rewritten every
        # layer (the QKV one on the dense path only).
        qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)

        layers, S = self._layers, self._stack
        # Each block's output and injection go into the next site's first kernel (`_hc_site`),
        # after an MoE block with its combine (and at one rank its finalize, `fold`); before
        # the PLE layer the pending block is added here, the PLE tail taking the site's norm.
        block_out = inject = moe = None
        # `DP_RS_INJECT_ENV`: the next site's norm, which the last MoE block's fused
        # reduce-scatter already formed (nothing else is pending then).
        rs_normed = None
        # A tep4 rank's PLE front runs beside the layers before the PLE layer.
        ple_front = self._ple_fork(input_ids, ple_state, is_verify) if self._ple_forks() else None
        for i, L in enumerate(layers):
            normed, rs_normed = rs_normed, None
            if i == core.ple_layer:
                if block_out is not None:
                    if moe is not None:
                        block_out = fused_sigmoid_gate_mul_add(block_out, moe[0], moe[1])
                    hc = qwen4_exp_post_inject(hc, block_out, inject)
                    block_out = inject = moe = None
                hc, normed = self._ple_block(
                    hc, input_ids, ple_state, L.ahc_norm, is_verify, ple_front
                )
            # `DP_INPROJ_RELEASE_ENV`: where the block's input projection runs on the PDL bf16
            # GEMM (`q_in`, None with the switch off), site A releases its dependents early.
            q_in = _inproj_release_at(core, i, num_tokens)
            hc, mixed, inject = self._hc_site(
                hc,
                block_out,
                inject,
                L.ahc_norm,
                L.ahc_first,
                L.ahc_up,
                True,
                normed,
                moe,
                release=q_in is not None,
            )
            # `DP_OUT_PROJ_INJECT_ENV`: where this layer's site B folds, the residual and the
            # injection ride the block's output projection (False with the switch off).
            armed = _siteb_arm(core, L, i, num_tokens, hc, inject)
            try:
                if core.is_attn[i]:
                    block_out = self._attention_block(
                        mixed, i, L, qkv, index_scratch, attn_out, qsa, step, is_verify, q_in
                    )
                else:
                    block_out = self._gdn_block(
                        mixed, i, L, si_all, verify, gen_p, md.is_cuda_graph, q_in
                    )
            finally:
                if armed:
                    _siteb_disarm(core)
            hc, mixed, inject, fp4 = self._hc_moe_site(hc, block_out, inject, L)
            block_out, moe = self._moe(
                mixed,
                L,
                pad_rows,
                fp4,
                core.moe_fold and i + 1 != core.ple_layer,
                md.is_cuda_graph,
                _dp_rs_inject_next(
                    core, md.is_cuda_graph, pad_rows, num_tokens, i, layers, S.fhc_norm, hc, inject
                ),
            )
            if type(block_out) is _RsInjected:
                # The block's reduce-scatter injected and normalized for the next site.
                hc, rs_normed = block_out
                block_out = inject = moe = None
        # The stack-final site with no injection weight: its mix *is* the final norm.
        hc, final, _ = self._hc_site(
            hc, block_out, inject, S.fhc_norm, S.fhc_first, S.fhc_up, False, rs_normed, moe
        )
        if spec_metadata is None:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the `[T, stream]` stream
        # it was collapsed from.
        return final, hc

    # ------------------------------------------------------------- the draft ---

    def _capture_selection(self, md, topk_indices, visible_blocks, batch, p) -> None:
        """IndexShare: stash, per request, the selection of its last accepted draft-extend
        row with that row's complete-group count, in the draft layer's row of the shared
        buffers. Every request carries `p` rows here, and the worker continues the draft loop
        from row `num_accepted - 1` of each (the golden token counts as accepted)."""
        assert md.qsa_mtp_num_accepted is not None, (
            "IndexShare needs the worker's accepted counts to pick each request's capture row; "
            "the draft loop did not supply them"
        )
        accepted = md.qsa_mtp_num_accepted[:batch].to(torch.int64)
        rows = torch.arange(batch, dtype=torch.int64, device=topk_indices.device) * p
        rows = rows + torch.clamp(accepted - 1, min=0, max=p - 1)
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
        """One draft step over generation rows only: every draft step past the first in an
        engine forward (one row a request), and the first (`p` rows a request) when the
        batch had no context. Inside the captured decode graph when the trunk step is.

        Under IndexShare the loop's draft-extend step (`p` rows) runs the draft indexer and
        captures each request's selection; every draft decode step after it skips the index
        branch and rebuilds its selection from that capture with `qsa_mtp_reuse_topk`, which
        appends the groups completed since (the expansion adds the incomplete tail, so the
        tokens drafted since stay visible). What a skipped step would have written to the
        index plane sits at or past the newly committed length, and the next engine step's
        draft-extend rewrites it before anything reads it. `all_rank_num_tokens` is the
        group's row counts of this draft step, which pad the MoE round trip under attention DP
        (`_draft_pad_rows`)."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, "the trunk's cache-manager bind has not run"
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows and hidden_states.shape[1] == core.stream, (
            f"the draft step got {input_ids.shape[0]} token ids and {tuple(hidden_states.shape)} "
            f"hidden states; the MTP module reads one {core.stream}-wide stream row per token"
        )
        # Projected on every call: the worker rewrites `attn_metadata` in place between
        # draft steps.
        step = self.step_args(md)
        gen_seqs = md.num_seqs
        p, _ = _verify_rows(core, md, None, gen_seqs, rows)
        step["predicted_tokens_per_seq"] = p
        draft, layer_idx = core._draft_step_bind(md)
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        pad_rows = _draft_pad_rows(core, all_rank_num_tokens, rows)
        rsi = getattr(core, "_rs_inject", None)
        if rsi is not None:
            # `DP_DRAFTPACK_ENV` `headnorm`: no earlier pass's combine reaches this pass's head.
            rsi.stash = None
        # The trunk's dense/sparse decision from the same host buffer: the worker advances
        # only the device KV lengths through the draft loop, so every draft step of one engine
        # forward takes the branch its trunk took, in the graph family the runner chose.
        sparse = gen_seqs > 0 and int(md.kv_lens_runtime[:gen_seqs].max()) > core.dense_threshold
        share = core.mtp_index_share and md.qsa_in_mtp_draft_loop
        reuse = share and md.qsa_indexer_skip_topk
        assert not share or reuse == (p == 1), (
            f"the worker's reuse flag ({md.qsa_indexer_skip_topk}) disagrees with the step's "
            f"shape ({p} rows per request): only the draft decode steps after the draft-extend "
            "step reuse its selection"
        )
        # The first draft step reads the trunk's positions buffer, whose int32 copy the trunk
        # made (taken once); a contiguous int32 `position_ids` (every later draft step) is read
        # in place. The step's ops only read positions.
        trunk, self._trunk_pos3 = self._trunk_pos3, None
        if trunk is not None and trunk[0] == _position_key(position_ids):
            pos3 = trunk[1]
            pos, coords = pos3[0], torch.transpose(pos3, 0, 1)
        elif position_ids.dtype == torch.int32 and position_ids.is_contiguous():
            pos3 = torch.reshape(position_ids, [_QSA_POSITION_AXES, -1])
            pos, coords = pos3[0], torch.transpose(pos3, 0, 1)
        else:
            pos3, pos, coords = _draft_positions(position_ids)
        M = core._mtp
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block

        if core.one_rank:
            return self._mtp_layer(
                embed_tokens,
                input_ids,
                hidden_states,
                md,
                step,
                draft,
                layer_idx,
                rows,
                gen_seqs,
                p,
                sparse,
                share,
                reuse,
                pos3,
                coords,
            )
        # A draft-extend step goes on past the K/V and index-key writes on each request's
        # continuing row only (`_draft_continue`); the MoE round trip then pads by the group's
        # kept-row counts.
        cont = _draft_continue(
            core, md, kwargs.get("spec_metadata"), all_rank_num_tokens, rows, gen_seqs, p, share
        )
        if core.tep_draft_layer:
            return self._mtp_layer(
                embed_tokens,
                input_ids,
                hidden_states,
                md,
                step,
                draft,
                layer_idx,
                rows,
                gen_seqs,
                p,
                sparse,
                share,
                reuse,
                pos3,
                coords,
                cont,
            )
        n = rows
        if cont is not None:
            n = gen_seqs
            pad_rows = _draft_pad_rows(core, cont.counts, n)
        x, mixed, inject = _mtp_front(core, M, embed_tokens, input_ids, hidden_states)
        if core.dp_mtp_attn_pack and core._mtp_attn_in is not None:
            # `DP_MTP_ATTN_PACK_ENV`: the attention input in the one-rank form, then the
            # multi-rank core past it (a rank's own row counts and collectives are unchanged).
            gemm = core.decode_mm if md.is_cuda_graph else cublas_mm
            proj, q_index = self._mtp_attn_front(mixed, md, draft, rows, reuse, pos3, coords, gemm)
            attn = self._mtp_attention_tp(
                md, step, draft, layer_idx, proj, q_index, rows, gen_seqs, p, sparse, share,
                reuse, cont,
            )  # fmt: skip
            kept = {"x": x, "inject": inject}
            if cont is not None:
                kept = _draft_rows(cont.rows, kept)
            out = _mtp_tail(core, M, kept["x"], attn, kept["inject"], pad_rows)
            if cont is None:
                return out
            full = out.new_empty([rows, out.shape[1]])
            full.index_copy_(0, cont.rows, out)
            return full
        qkv, gate, k_dst, v_dst = _mtp_qkv(core, M, mixed, pos3)
        q_index = None
        if not reuse:
            q_index = _mtp_index_branch(
                core,
                M,
                mixed,
                draft,
                (req_idx, logical, block_table, tokens_per_block, pos, coords),
            )
        # The step's operands at the rows that go on (all of them without `cont`).
        kept = {"x": x, "inject": inject}
        if sparse:
            kept.update(qkv=qkv, gate=gate)
            if q_index is not None:
                kept["q_index"] = q_index
        if cont is not None:
            kept = _draft_rows(cont.rows, kept)
        if sparse:
            _store_kv(core, draft.kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            g = kept["gate"]
            attn = self._draft_sparse(
                md,
                draft,
                _packed_q(core, kept["qkv"]),
                g,
                kept.get("q_index"),
                req_idx,
                logical,
                cont,
                rows,
                gen_seqs,
                p,
                share,
                reuse,
            )
            gated = True
        else:
            g = gate
            gated = False
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
        if not gated:
            fused_sigmoid_mul_inplace(attn, g)
        if cont is not None and not sparse:
            attn = torch.index_select(attn, 0, cont.rows)
        out = _mtp_tail(core, M, kept["x"], attn, kept["inject"], pad_rows)
        if cont is None:
            return out
        # The worker reads only the continuing rows; the others are left unwritten.
        full = out.new_empty([rows, out.shape[1]])
        full.index_copy_(0, cont.rows, out)
        return full

    def _draft_sparse(
        self, md, draft, q, gate, q_index, req_idx, logical, cont, rows, gen_seqs, p, share, reuse
    ) -> torch.Tensor:
        """A multi-rank draft step's sparse core after its K/V append, at the rows that go on
        (all `rows` without `cont`, each request's continuing row with it): the index scores and
        the selection on the family's decode ops (`_decode_index_scores`, `_selection`), or under
        IndexShare a draft decode step's selection rebuilt from the draft-extend step's capture
        (`reuse`; `share` captures it), then the gated sparse GQA (`_sparse_gqa_decode`, else
        `_sparse_gqa` and the gate apart). `q` is `[n, heads_q, head_dim]`, `gate` its output gate
        and `q_index` the normalized index query at the same rows (None under `reuse`). Returns
        the gated attention output `[n, q_width]`."""
        core = self.core
        kv = draft.kv
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
        n = q.shape[0]
        topk_indices = md.qsa_topk_indices[:n]
        if cont is None:
            visible_blocks = md.qsa_visible_blocks[:rows]
            seq_lens_q = md.qsa_sequence_lengths[:rows]
            q_logical, q_req_idx = logical, req_idx
        else:
            visible_blocks, seq_lens_q = cont.visible, cont.seq_lens
            q_logical, q_req_idx = cont.logical, cont.req_idx
        selected = torch.empty(
            [n, core.token_topk + core.compress_ratio - 1],
            dtype=torch.int32,
            device=q.device,
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
                request_indices=req_idx,
                visible_blocks=visible_blocks,
                output=topk_indices,
            )
            selected.copy_(
                qsa_expand_block_indices(
                    topk_indices,
                    logical,
                    seq_lens_q,
                    compress_ratio=core.compress_ratio,
                    token_topk=core.token_topk,
                )
            )
        else:
            scores = _decode_index_scores(
                core,
                q_index,
                draft.index_cache,
                block_table,
                q_logical,
                q_req_idx,
                tokens_per_block,
                p if cont is None else 1,
            )
            _selection(
                core,
                scores,
                visible_blocks,
                q_logical,
                seq_lens_q,
                topk_indices,
                md.qsa_topk_row_starts[:n],
                selected,
                draft=True,
            )
            if share:
                self._capture_selection(md, topk_indices, visible_blocks, gen_seqs, p)
        attn = _sparse_gqa_decode(
            core, kv, q, block_table, selected, q_req_idx, tokens_per_block, gate,
            p if cont is None else 1,
        )  # fmt: skip
        if attn is not None:
            return torch.reshape(attn, [n, core.q_width])
        attn = _sparse_gqa(core, kv, q, block_table, selected, q_req_idx, tokens_per_block, None)
        attn = torch.reshape(attn, [n, core.q_width])
        fused_sigmoid_mul_inplace(attn, gate)
        return attn


# ------------------------------------------------------------ the MTP module ---


def _mtp_head_final(core, hidden_states: torch.Tensor) -> torch.Tensor:
    """The MTP module's head input: its stack-final hyper-connection mixer over the grouped norm of
    `hidden_states` (the module's un-collapsed residual), `[rows, hidden]`."""
    M = core._mtp
    # `DP_DRAFTPACK_ENV` `headnorm`: the draft combine of this pass already formed the norm of
    # exactly these rows (the stash holds its output, so no other tensor can share its storage).
    rsi = getattr(core, "_rs_inject", None)
    stash = None
    if rsi is not None:
        stash, rsi.stash = rsi.stash, None
    if stash is not None and _same_tensor(hidden_states, stash.hc):
        normed = stash.normed
    else:
        _, normed = qwen4_exp_hc_inject_norm(
            hidden_states, None, None, M.fhc_norm, core.eps, core.hc_count
        )
    final, _ = _hc_mixer(core, normed, M.fhc_first, M.fhc_up, False)
    return final


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether `a` is a view of exactly `b`'s elements (storage, shape, strides, dtype)."""
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


#: Opt-in: under attention DP (a dep4 / dep8 rank on SM103) the vocabulary projection of the
#: heads it names runs over a group of attention-DP ranks that each hold a slice of `lm_head`
#: (upstream's `GroupShardedVerify`): the members' rows are gathered, each member projects all of
#: them on its 1/G of the vocabulary with the bf16 weight (a view, no copy) and the tokens are
#: picked across the slices, exchanging a few words a row instead of streaming the whole head at
#: every rank. Comma list of `draft` (the greedy draft tokens, through the spec worker's
#: `draft_head` hook) and `verify` (the target tokens of greedy and temperature-only steps), or
#: `1` for both. Only steps in which the group holds one row count it serves take it, and only
#: row counts at which this rank's slice logits are the full head's bits (checked at build on
#: every rank and agreed over the group): a greedy row's token is the full head's argmax. A
#: temperature-only row is an exact draw from `softmax(logits / T)` at its own Philox seed and
#: offset, but not the stock sampler's draw (class S: the text of a sampled request changes as it
#: would with another seed). Measured (GB300, our own kernels over the same slices): the verify
#: head -0.11..-0.13 ms a step at dep4 12 requests a rank, the draft head part of a -0.47 ms
#: step bundle. Read once, after load.
DP_HEAD_SHARD_ENV = "TRTLLM_QWEN4_EXP_DP_HEAD_SHARD"
#: Which ranks form a group: `node` (the default, upstream's rule: `min(gpus_per_node, DP size)`
#: consecutive ranks, IPC peer memory) or `domain` (every attention-DP rank, MNNVL peer memory
#: across the nodes of one NVLink domain: dep8 on 4-GPU trays projects 1/8 of the vocabulary a
#: rank instead of 1/4, as measured).
DP_HEAD_SHARD_GROUP_ENV = "TRTLLM_QWEN4_EXP_DP_HEAD_SHARD_GROUP"
#: Requests a rank each head was measured to serve (the verify head with 4 rows a request).
_DP_HEAD_SHARD_REQUESTS = {"verify": (1, 32), "draft": (1, 64)}  # capacity: fixed: dispatch


def _dp_head_shard_heads() -> frozenset:
    """`DP_HEAD_SHARD_ENV`: the heads it names (empty: off)."""
    raw = os.environ.get(DP_HEAD_SHARD_ENV, "")
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return frozenset()
    if value in _SWITCH_ON:
        return frozenset({"draft", "verify"})
    heads = frozenset(h.strip() for h in value.split(","))
    if not heads <= {"draft", "verify"}:
        raise ValueError(f"{DP_HEAD_SHARD_ENV}={raw!r}: expected 0, 1 or a list of draft, verify")
    return heads


def _dp_head_shard_group() -> str:
    """`DP_HEAD_SHARD_GROUP_ENV`: `node` (unset or empty) or `domain`."""
    raw = os.environ.get(DP_HEAD_SHARD_GROUP_ENV, "")
    value = raw.strip().lower() or "node"
    if value not in ("node", "domain"):
        raise ValueError(f"{DP_HEAD_SHARD_GROUP_ENV}={raw!r}: expected node or domain")
    return value


class _ShardVerifyLogits(VocabShardedLogits):
    """What the spec worker gets as `logits` in a step `DP_HEAD_SHARD_ENV`'s verify head served:
    the step's target tokens, already drawn (`_sample_tokens_for_batch` takes them through
    `select_tokens`). It holds no logits; the worker reads `dim()` and `device` besides, and the
    shell swaps the object for a placeholder in the outputs it returns."""

    def __init__(self, tokens: torch.Tensor, vocab: int) -> None:
        self.tokens, self.vocab = tokens, vocab

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def num_rows(self) -> int:
        return self.tokens.shape[0]

    @property
    def shape(self) -> tuple:
        return (self.num_rows, self.vocab)

    def dim(self) -> int:
        return 2

    def gather(self) -> torch.Tensor:
        raise RuntimeError(
            f"{DP_HEAD_SHARD_ENV}: this step's target logits are vocabulary slices on the ranks "
            "of a group; a consumer of the whole distribution cannot be served"
        )

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        return self.tokens


class _DpHeadShard:
    """`DP_HEAD_SHARD_ENV`'s state on one attention-DP rank: the group (`GroupShardedVerify`), the
    row counts each head serves (the same sets on every member), and the step's per-rank request
    counts, which the draft hook decides from.

    Every member of a group makes the same calls in the same order with the same rows, so each
    decision reads only what the members hold alike: the step's per-rank request and token
    counts, the CUDA-graph flag, the group-synchronized sampling variant
    (`samples_from_vocab_shards` makes the engine exchange it) and deploy-time constants. A rank
    whose own rows differ from the group's in a served step raises: its peers would wait."""

    def __init__(self, group, core, verify_rows: frozenset, draft_rows: frozenset) -> None:
        self.group, self.core = group, core
        self.verify_rows, self.draft_rows = verify_rows, draft_rows
        self.placeholder = torch.full(
            (max(verify_rows, default=1), 1),
            float("nan"),
            dtype=torch.float32,
            device=group.weight.device,
        )
        self.counts: list | None = None
        self.graph = False

    def _members(self, counts) -> list:
        g = self.group
        return list(counts[g.first_dp_rank : g.first_dp_rank + g.group_size])

    def begin_step(self, spec_metadata) -> None:
        """Stash the step's per-rank request counts and its CUDA-graph flag (host values)."""
        counts = getattr(spec_metadata, "subseq_all_rank_num_tokens", None)
        self.counts = None if counts is None else [int(n) for n in counts]
        self.graph = bool(getattr(spec_metadata, "is_cuda_graph", False))

    def _engages(self, rows: int, counts, served: frozenset, what: str) -> bool:
        if not self.graph or counts is None:
            return False
        members = self._members(counts)
        if any(c != members[0] for c in members) or members[0] not in served:
            return False
        if rows != members[0]:
            raise RuntimeError(
                f"{DP_HEAD_SHARD_ENV}: rank {self.core.dp_rank} holds {rows} {what} rows in a step "
                f"whose group holds {members}"
            )
        return True

    def draft_engages(self, rows: int) -> bool:
        return self._engages(rows, self.counts, self.draft_rows, "draft")

    def tokens(self, final: torch.Tensor, params=(None, None, None)) -> torch.Tensor:
        """This rank's tokens of `final` `[R, hidden]` bf16 over the group's slices (two group
        launches and the slice GEMM); `params` the rows' temperatures, seeds and offsets."""
        return self.group.sample(self.group.logits(final, *params))

    def verify_tokens(self, worker, spec_metadata, attn_metadata, head_rows):
        """The target tokens of a step the verify head serves, int64 `[rows]`, or None."""
        if not self.verify_rows or not self.graph:
            return None
        if spec_metadata.is_ragged_verify or getattr(worker, "guided_decoder", None) is not None:
            return None
        greedy = spec_metadata.is_all_greedy_sample
        if not (greedy or spec_metadata.is_all_temperature_only):
            return None
        tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        if not self._engages(head_rows.shape[0], tokens, self.verify_rows, "verify"):
            return None
        params = (None, None, None)
        if not greedy:
            rows = head_rows.shape[0]
            params = (
                spec_metadata.temperatures[:rows],
                spec_metadata.seeds[:rows],
                spec_metadata.offsets[:rows],
            )
        return self.tokens(head_rows.contiguous(), params)


class _DpShardDraftHead:
    """`DP_HEAD_SHARD_ENV`'s `draft`: the spec worker's `draft_head` hook. The worker asks
    `supports(rows)` once a draft pass and, on True, calls `sample` in place of
    `MTPLayer.shared_head` + the greedy draft sampler."""

    def __init__(self, core, shard: _DpHeadShard) -> None:
        self.core, self.shard = core, shard

    def supports(self, rows: int) -> bool:
        return self.shard.draft_engages(rows)

    def sample(self, hidden_states: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Greedy draft tokens, int32 `[rows]` (into `out` when given)."""
        tokens = self.shard.tokens(_mtp_head_final(self.core, hidden_states))
        if out is None:
            return tokens.to(torch.int32)
        out.copy_(tokens)
        return out


@torch.no_grad()
def _dp_head_shard_check(group, lm_head, counts) -> list:
    """The row counts in `counts` at which this rank's slice logits (the group's GEMM over
    `group_size * rows` test rows) differ in any bit from the full head called `rows` rows at a
    time, as each rank calls it. On this rank alone; also the slice GEMM's warm-up.

    Without grad: it runs in `post_load_weights`, outside the executor's inference mode, where
    the head's weight still requires grad and `torch.mm(..., out=)` refuses such an input."""
    bad = []
    g, lo, n = group.group_size, group.vocab_offset, group.local_vocab
    dev = group.weight.device
    for rows in sorted(counts):
        gen = torch.Generator(device=dev).manual_seed(20261125 + 1009 * rows + group.rank)
        x = torch.randn((g * rows, group.hidden), generator=gen, device=dev)
        x = (x * torch.exp(torch.randn_like(x[:, :1]))).to(torch.bfloat16)
        mine = torch.mm(x, group.weight.t(), out=group.slice_logits[: g * rows])
        for p in range(g):
            full = lm_head(x[p * rows : (p + 1) * rows])[:, lo : lo + n]
            if not torch.equal(
                mine[p * rows : (p + 1) * rows].view(torch.int16), full.view(torch.int16)
            ):
                bad.append(rows)
                break
    return bad


def _dp_head_shard_build(shell, device):
    """`DP_HEAD_SHARD_ENV`, read once after load (the shell's `post_load_weights`): the rank's
    `_DpHeadShard` (and the worker's `draft_head` hook), or None. Collective over the attention-DP
    group where set at a layout it serves: the group's peer buffers, then one all-gather over the
    attention-DP group of whether each rank's group was built and the row counts whose slice
    logits differ on a rank (a count no rank failed is served; a group missing anywhere leaves the
    heads unchanged on every rank). Everything deciding before the collectives is the layout, the
    deploy configuration or the env, the same on every rank."""
    heads = _dp_head_shard_heads()
    if not heads:
        return None
    core = shell.model
    worker = shell.spec_worker
    spec = shell.spec_config
    lm_head = getattr(shell, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    why = None
    if not _dp_decode_rank(core, device):
        why = "it serves a dep4 / dep8 rank on SM103"
    elif worker is None or not core.mtp_enabled:
        why = "there is no MTP spec worker"
    elif getattr(spec, "use_rejection_sampling", False) or getattr(spec, "enable_penalty", False):
        why = "rejection sampling and the occurrence penalties read the whole target distribution"
    elif getattr(spec, "use_relaxed_acceptance_for_thinking", False):
        why = "relaxed acceptance reads the target distribution"
    elif (
        getattr(spec, "acceptance_rate_window_size", None)
        and getattr(spec, "acceptance_rate_threshold", None) is not None
    ):
        # As upstream's group verify head (`DSv4DSparkWorker.prepare_vocab_sharded_sampling`):
        # the acceptance-rate gate can stop one rank's drafting alone, and its peers would wait
        # in the group's draft-head exchange.
        why = "the acceptance-rate gate can stop one rank's drafting alone"
    elif getattr(core.mapping, "enable_lm_head_tp_in_adp", False):
        why = "enable_lm_head_tp_in_adp shards the heads its own way"
    elif "draft" in heads and not (
        getattr(worker, "is_mtp_eagle", False) and getattr(worker, "draft_head", 1) is None
    ):
        why = "the spec worker has no free `draft_head` hook in an MTP-Eagle draft loop"
    elif not (
        type(lm_head) is LMHead
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and tuple(weight.shape) == (core.vocab, core.hidden)
        and getattr(lm_head, "bias", None) is None
        # The group's slice GEMM writes bf16 logits; a head with another output dtype (fp32
        # `lm_head_output_dtype`) would differ in every count's bits (as upstream's group head).
        and getattr(lm_head, "output_dtype", None) in (None, torch.bfloat16)
    ):
        why = "lm_head is not the replicated dense bf16 head with bf16 logits"
    if why is not None:
        logger.warning(f"{DP_HEAD_SHARD_ENV} is set but ignored: {why}; the heads are unchanged")
        return None
    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.distributed.peer_buffers import PeerTransport
    from tensorrt_llm._torch.speculative.group_sharded_sampling import GroupShardedVerify

    domain = _dp_head_shard_group() == "domain"
    size = core.dp_size if domain else min(core.mapping.gpus_per_node, core.dp_size)
    per = core.max_verify_rows
    sizes = sorted(
        {int(b) for b in (getattr(core.model_config, "cuda_graph_batch_sizes", None) or ())}
    )
    want = {}
    for head in heads:
        lo, hi = _DP_HEAD_SHARD_REQUESTS[head]
        if head == "verify" and "head" in _dp_hicap(core):
            lo, hi = _DP_HICAP_HEAD_VERIFY
        scale = per if head == "verify" else 1
        want[head] = frozenset(scale * b for b in sizes if lo <= b <= hi)
    max_rows = max((max(r) for r in want.values() if r), default=0)
    if not max_rows:
        # The deploy configuration's, the same on every rank: nothing collective has run.
        logger.warning(
            f"{DP_HEAD_SHARD_ENV} is set but ignored: no CUDA-graph batch size in the measured "
            f"range; the heads are unchanged"
        )
        return None
    counts = frozenset().union(*want.values())
    group, error, bad = None, "", []
    try:
        # Collective over this rank's group only. Its answer is the same on the group's members,
        # but under `node` another group may answer otherwise (its peer buffers failed), so the
        # whole attention-DP group agrees below before any rank returns.
        group = GroupShardedVerify.create(
            core.mapping, weight, size, max_rows, None if domain else PeerTransport.IPC
        )
        if group is None:
            error = (
                f"no group of {size} ranks ({DP_HEAD_SHARD_GROUP_ENV}="
                f"{'domain' if domain else 'node'}) over this mapping"
            )
        else:
            bad = _dp_head_shard_check(group, lm_head, counts)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001 -- every rank agrees below
        error = f"{type(e).__name__}: {e}"
    reports = Distributed.get(core.mapping).tp_allgather((error, bad))
    errors = [e for e, _ in reports if e]
    refused = frozenset().union(*(frozenset(b) for _, b in reports))
    if errors:
        logger.warning(
            f"{DP_HEAD_SHARD_ENV} is set but ignored on every rank: {errors[0]}; the heads are "
            "unchanged"
        )
        return None
    shard = _DpHeadShard(
        group,
        core,
        want.get("verify", frozenset()) - refused,
        want.get("draft", frozenset()) - refused,
    )
    if shard.draft_rows:
        worker.draft_head = _DpShardDraftHead(core, shard)
    logger.info(
        f"attention-DP heads over vocabulary slices ({DP_HEAD_SHARD_ENV}): rank {core.dp_rank} "
        f"of {core.dp_size}, group of {size} ({group.buffers.transport.value}), slice "
        f"[{group.vocab_offset}, {group.vocab_offset + group.local_vocab}), verify rows "
        f"{sorted(shard.verify_rows)}, draft rows {sorted(shard.draft_rows)}, not bitwise on some "
        f"rank (full head kept): {sorted(refused)}"
    )
    return shard if (shard.verify_rows or shard.draft_rows) else None


class MTPLayer:
    """The checkpoint's MTP module, replayed once per draft step by the spec worker: one more
    decoder layer (two hyper-connections, gated GQA over its own QSA indexer, a 512-expert
    FP8 MoE) behind a two-branch front end (`fc_embedding` over the next token's embedding,
    `fc_hidden` over the trunk's residual per stream). It returns the residual un-collapsed;
    `shared_head` owns the module's mixer. The step runs on the targets (`mtp_forward`),
    routed by the phase each call's metadata shows: the worker rewrites `attn_metadata`
    between draft steps, so one engine step's draft calls cross from prefill to decode.

    Not an `nn.Module`: the weights live in the core's `ParameterDict`, and a second
    registration would break the weight load's both-directions coverage."""

    def __init__(self, core: Qwen4ExpDefaultModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor
        # The spec worker when it samples with the TP key exchange (`_sharded_head`).
        self.worker = None
        # The engine step's `spec_metadata`, set by the shell before the worker runs.
        self.spec_metadata = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def _sharded_head(self, lm_head, rows: int) -> bool:
        """Whether the draft logits may stay this rank's vocabulary shard under attention TP:
        the worker then takes the greedy draft tokens through the TP key exchange (row maxima
        and one exchange, the gathered head's argmax with ties to the lowest id) instead of
        the all-gather. Not under advanced draft sampling or guided decoding, which read the
        gathered logits. Host values every rank shares, as the FP8 target decides it."""
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
        """The module's own head: its stack-final hyper-connection mixer (a gated mixer with
        its own parameters, not a plain RMS norm) over the grouped norm, then the trunk's
        `lm_head` through the shell's logits processor (vocab-parallel and gathered under
        attention TP; this rank's bf16 vocabulary shard where `_sharded_head` allows). The norm
        and the mixer are the trunk's stack-final site's."""
        core = self.core
        final = _mtp_head_final(core, hidden_states)
        draft_head = getattr(core, "draft_head", None)
        if draft_head is None:
            if self._sharded_head(lm_head, final.shape[0]):
                return lm_head(final, gather_output=False)
            return self.logits_processor.forward(
                final, lm_head, attn_metadata, return_context_logits
            )
        # At tp1 the FP8 draft head, its logits bf16 when only the draft sampler's argmax reads
        # them.
        return self.logits_processor.forward(
            final,
            draft_head,
            attn_metadata,
            return_context_logits,
            widen=not self.logits_processor.draft_argmax_only,
        )

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
    """The container the spec worker reaches the drafter through: `mtp_layers` (`[0]`,
    MTP-Eagle replays one layer), `embed_tokens` and `lm_head`. `embed_tokens` resolves
    through the core on every read: this container is built while parameters are meta
    tensors, and materialization replaces the tensor objects."""

    def __init__(self, core: Qwen4ExpDefaultModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class SpeculativeRollback:
    """The target's auxiliary speculative-state handler: it undoes what a verify step's
    rejected draft rows wrote into the side-cache planes the runtime's own state promotion
    does not cover: the PLE n-gram id history and dilated-conv state, each attention layer's
    QSA index-K plane and the shared QSA position plane. The Gated-DeltaNet state is not
    here: the runtime promotes it.

    For a generation request that accepted `a` rows (golden token included, `1 <= a <= p`)
    the commit rewrites the n-gram pool row to the id window of row `a - 1` minus its oldest
    id, the conv pool row to columns `[a, a + state_len)` of `[pre-step state | conv input of
    the request's p rows]`, and every index-K and position slot written by a row of ordinal
    `>= a` to its pre-verify contents.

    The targets stage during the forward (`begin`, then `stage_ple_verify` before the fused PLE
    launches and `stage_ple_conv_input` after them, and `stage_index` before each attention
    layer's store) and the spec worker calls `commit_speculative_states` after its draft loop,
    inside the same model forward. Everything staged is allocated by the forward and consumed
    by that forward's commit, so a captured graph holds both ends. A step opened with `fused`
    (the decode target's captured steps) stages the PLE pools with one `spec_rollback_stage`
    launch and commits with one `spec_rollback_commit` launch."""

    def __init__(self) -> None:
        self._pending: dict | None = None
        # (base addresses, page strides, plane count) of every attention layer's index-K
        # plane, once the cache manager is bound.
        self._planes: tuple | None = None

    def set_index_planes(self, planes: list) -> None:
        """Bind the attention layers' index-K planes (static for the manager's lifetime)."""
        ptrs, strides = _plane_table(planes)
        self._planes = (ptrs, strides, len(planes))

    # ------------------------------------------------------------- staging ---

    def discard(self) -> None:
        """Drop whatever an earlier forward staged."""
        self._pending = None

    @property
    def index_planes(self) -> tuple | None:
        """`(base addresses, page strides, plane count)` of the bound index-K planes."""
        return self._planes

    def begin_staged(
        self,
        *,
        num_gens: int,
        rows_per_gen: int,
        slots: torch.Tensor,
        pages: torch.Tensor,
        within: torch.Tensor,
        index_all: torch.Tensor,
    ) -> None:
        """Open a generation-only verify step's staging from what `verify_step_setup`
        resolved: the requests' int64 recurrent slots, the rows' int64 QSA pages and offsets,
        and every bound index-K plane at them. The PLE staging and the commit are the
        one-launch ones (`begin(..., fused=True)`)."""
        n, p = num_gens, rows_per_gen
        assert n > 0 and p > 1, (
            f"a verify step stages generation requests carrying drafts; got {n} requests of "
            f"{p} rows"
        )
        assert pages.shape[0] == n * p and index_all.shape[1] == n * p, (
            f"{n} generation requests of {p} rows are {n * p} rows; got {pages.shape[0]} slots"
        )
        self._pending = dict(
            num_contexts=0,
            prefix=0,
            n=n,
            p=p,
            slots=slots,
            pages=pages,
            within=within,
            staged_layers=0,
            position=None,
            index_all=index_all,
            fused=True,
            ngram_stream=None,
        )

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
    ) -> None:
        """Open a verify step's staging. The generation rows follow the `prefix_tokens`
        context rows as `num_gens` runs of `rows_per_gen`, so row `s * p + t` is ordinal `t`
        of generation request `s`. `slots` are those requests' recurrent slots;
        `request_indices` and `logical_positions` the generation rows' own. Each row's QSA
        slot is resolved here, as `qsa_paged_index_store` resolves it, and every index plane's
        slots are gathered before the trunk writes any of them."""
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
        positions = logical_positions.to(torch.long)
        pages = block_table[request_indices.to(torch.long), positions // tokens_per_block]
        pages = pages.to(torch.long)
        within = positions % tokens_per_block
        ptrs, strides, count = self._planes
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

    def stage_ple_verify(
        self, ngram_pool: torch.Tensor, conv_pool: torch.Tensor, input_ids: torch.Tensor
    ) -> None:
        """A verify step's PLE staging, before either pool is written: both pools, the
        generation slots' pre-step conv rows, and the n-gram candidates rebuilt from the slots'
        pre-step id history and the rows' ids. Row `t`'s history after the step is the last
        `context` ids of `[history | ids of rows 0..t]`."""
        pending = self._pending
        prefix, n, p = pending["prefix"], pending["n"], pending["p"]
        slots = pending["slots"]
        pending["ngram_pool"] = ngram_pool
        pending["conv_pool"] = conv_pool
        if pending["fused"]:
            pending["ngram_stream"], pending["conv_prev"] = spec_rollback_stage(
                ngram_pool, conv_pool, slots, input_ids[prefix : prefix + n * p], p
            )
            return
        pending["conv_prev"] = conv_pool[slots]
        context = ngram_pool.shape[1]
        ids = torch.reshape(input_ids[prefix : prefix + n * p], [n, p]).to(ngram_pool.dtype)
        stream = torch.cat([ngram_pool[slots], ids], 1)
        # Window k of the unfold is the history after k rows.
        pending["ngram_candidates"] = stream.unfold(1, context, 1)[:, 1:]

    def stage_ple_conv_input(self, conv_input: torch.Tensor) -> None:
        """The generation rows' conv input (`[T, C]`, token-major), as the fused PLE launch
        wrote it into a buffer the forward allocated for this staging alone. A mixed batch's
        rows are copied out, so its context prefix is not kept alive through the draft loop."""
        pending = self._pending
        prefix, n, p = pending["prefix"], pending["n"], pending["p"]
        conv_x = torch.reshape(conv_input[prefix:], [n, p, -1])
        pending["conv_x"] = conv_x.clone() if prefix else conv_x

    def stage_index(self, position_cache: torch.Tensor) -> None:
        """Before an attention layer's index store: on the step's first attention layer (the
        first to write the shared plane) the generation rows' position slots. The index-K
        slots of every layer were gathered when the step was opened."""
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


#: Switch, on by default (bitwise; a row table): at a dep4 / dep8 rank on SM103
#: (`_dp_decode_rank`) the engine keeps its opted-in steady one-model MTP decode prepare at each
#: attention-DP rank (`RuntimeHooks.steady_decode_prepare`): a decode-only graph step whose rank
#: batch is the previous step's -- the same requests on the same slots, verifying the previous
#: step's device drafts, and the same CUDA-graph padding rows -- or whose requests all join on
#: that batch's metadata, writes its decoder inputs, the graph's static inputs and the KV lengths
#: in one launch between the two replays, and still makes the full pass's padding and count
#: exchange (so ranks that stage and ranks that do not keep their collectives paired). The QSA
#: metadata advances by its steady decode step (`RuntimeHooks.qsa.steady_decode`): no prompt /
#: KV-length uploads, the block offsets copied and unscaled only when the KV manager's block-table
#: key changed. The world-8 group is the same rule (one attention-DP group of the row table's
#: size), not a separate switch. Measured on an earlier attention-DP form of the
#: engine's prepare: dep4 -0.022..-0.050 ms a step, dep8 graph 12 -0.006,
#: 98.7% of decode-only graph steps staged at dep8. `0`: the full pass on every step.
DP_STEPEDGE_ENV = "TRTLLM_QWEN4_EXP_DP_STEPEDGE"
#: Switch, on by default where `DP_STEPEDGE_ENV` engages (bitwise): the replay-gap cuts of the
#: steady step (`RuntimeHooks.qsa.steady_replay_gap`). The eager QSA row mapping is not launched
#: (the replayed forward rewrites all four of its outputs before any reader: the overlap
#: scheduler's KV-length correction and `on_update_kv_lens`, or the decode prologue), and a
#: changed block table is copied and unscaled in one Triton launch (`triton_qsa_copy_block_offsets`,
#: the native copy's stores) where the KV manager's copy is the plain one. Measured on top of
#: `DP_STEPEDGE_ENV`: dep4 -0.013..-0.021 ms at graphs 5-7, dep8 about -0.011 at graphs 8-10 and
#: -0.013 at 12. `0`: the steady step launches the mapping and the native copy.
DP_RGAP_ENV = "TRTLLM_QWEN4_EXP_DP_RGAP"
#: Switch, on by default (bitwise; a row table): at a tep4 rank on SM103 (`_tep4_layout`) the QSA
#: metadata of the engine's steady one-model MTP decode prepare advances by its steady decode
#: step (`RuntimeHooks.qsa.steady_decode`) instead of its full `prepare()`: no prompt / KV-length
#: uploads, the block offsets copied and unscaled only when the block-table key changed. That is
#: the QSA half of the measured tep4 steady step (the gap between replays 82.5 -> 26.1 us, host
#: p50 -0.039..-0.050 ms at 1-4 requests, P90 interactivity +0.7..+1.2% end to end); its other half,
#: the one-launch input stager, is the engine's at every one-group layout already. `0`: the
#: metadata's full prepare() on staged steps.
TEP_STEPEDGE_ENV = "TRTLLM_QWEN4_EXP_TEP_STEPEDGE"


def _is_sm103(device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == archs.SM_103


def _host_step_forms(core, device=None) -> dict:
    """The decode-step host forms of this core's layout (`DP_STEPEDGE_ENV`, `DP_RGAP_ENV`,
    `TEP_STEPEDGE_ENV`), read once when the shell declares its runtime hooks: `dp_steady` (the
    steady prepare at an attention-DP rank), `qsa_steady` (the QSA steady decode step) and
    `qsa_replay_gap`. A switch set to 1 where its row table does not reach warns once. Decided
    from the layout and the GPU only, so every rank of a group decides alike."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dp = _dp_decode_rank(core, device)
    tep = _tep4_layout(core) and _is_sm103(device)
    dp_steady = dp and _switch(DP_STEPEDGE_ENV, default=True)
    replay_gap = dp_steady and _switch(DP_RGAP_ENV, default=True)
    tep_steady = tep and _switch(TEP_STEPEDGE_ENV, default=True)
    for env, serves, why in (
        (DP_STEPEDGE_ENV, dp, "a dep4 / dep8 rank on SM103"),
        (DP_RGAP_ENV, dp_steady, f"a dep4 / dep8 rank on SM103 with {DP_STEPEDGE_ENV} on"),
        (TEP_STEPEDGE_ENV, tep, "a tep4 rank on SM103"),
    ):
        if not serves and os.environ.get(env, "").strip() and _switch(env):
            logger.warning(f"{env} is set but ignored: it serves {why}")
    return {
        "dp_steady": dp_steady,
        "qsa_steady": dp_steady or tep_steady,
        "qsa_replay_gap": replay_gap,
    }


def _runtime_hooks(spec_config=None, core=None) -> RuntimeHooks:
    """The engine hooks of a speculative shell, as the FP8 target's: the spec loop on the
    execution stream, and the decode prologue -- a generation-only verify step's position and
    KV-length corrections and the QSA row mapping in one launch during input preparation
    (`qsa_spec_step_metadata`), bitwise what it replaces; the decode forward reads the corrected
    positions it leaves (`take_step_positions`). Under `CTX_COLDIDX_ENV`, the cold index codec.
    With `core`, the decode-step host forms of its layout (`_host_step_forms`), recorded on the
    core as `host_step_forms` for its decode-forms line."""
    forms = _host_step_forms(core) if core is not None else {}
    if core is not None:
        core.host_step_forms = forms
    cold_page_codec = None
    if _switch(CTX_COLDIDX_ENV):
        draft_len = getattr(spec_config, "max_draft_len", None)

        def cold_page_codec(kv_cache_config):
            # Imported here: the provider module reaches the executor's resource managers.
            from ..qsa_cold_pages import qsa_cold_pages_provider

            return qsa_cold_pages_provider(
                kv_cache_config, index_key_fp8=_switch(INDEX_KEY_FP8_ENV), max_draft_len=draft_len
            )

    return RuntimeHooks(
        loop_on_execution_stream=True,
        decode_prologue=SpecStepMetadataPrologue(),
        cold_page_codec=cold_page_codec,
        steady_decode_prepare=forms.get("dp_steady", False),
        qsa=QsaRuntimePolicy(
            steady_decode=forms.get("qsa_steady", False),
            steady_replay_gap=forms.get("qsa_replay_gap", False),
        ),
    )


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


def _fp8_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`weight` [rows, cols] bf16 -> (e4m3 [rows, cols], fp32 [1, rows]): one scale a row,
    amax / 448, so each row's largest magnitude maps to e4m3's largest finite value."""
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    w8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for lo in range(0, weight.shape[0], 16384):
        hi = min(lo + 16384, weight.shape[0])
        chunk = weight[lo:hi].float()
        scale[lo:hi] = (chunk.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
        w8[lo:hi] = (chunk / scale[lo:hi, None]).to(torch.float8_e4m3fn)
    return w8, scale.reshape(1, -1).contiguous()


def _head_mid(weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> MidWeight:
    """`mid_rows_mm`'s view of a per-row-scaled e4m3 head copy (`_fp8_rowwise`)."""
    return MidWeight(weight_fp8, weight_scale.reshape(-1))


class _Fp8Head:
    """A vocabulary projection over an FP8 copy of `lm_head`: up to `_FP8_GEMV_HEAD_ROWS` rows on
    `fp8_gemv` over the packed block-scaled copy -- or, given the NVFP4 copy `nv`, `nvfp4_gemv`
    over it --, up to `_MID_HEAD_ROWS` on `mid_rows_mm` over the per-row-scaled copy, more rows as
    per-token e4m3 activations times that copy, scaled in cuBLASLt's epilogue; bf16 out
    (`_WidenedHead` widens it as it does the bf16 head's). The draft head always (at tp1), the
    verify head under the dense FP8 mode."""

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, packed, nv) -> None:
        self.weight_fp8, self.weight_scale, self.packed, self.nv = (
            weight_fp8,
            weight_scale,
            packed,
            nv,
        )
        self.mid = _head_mid(weight_fp8, weight_scale)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows = hidden_states.shape[0]
        if rows <= _FP8_GEMV_HEAD_ROWS and _gemv_ready(hidden_states):
            if self.nv is not None:
                return nvfp4_gemv_op(hidden_states, self.nv)
            if self.packed is not None:
                return fp8_gemv_op(hidden_states, self.packed)
        if rows <= _MID_HEAD_ROWS and _gemv_ready(hidden_states):
            return mid_rows_mm_op(hidden_states, self.mid)
        x8, xs = torch.ops.tensorrt_llm.quantize_e4m3_activation(hidden_states)
        return torch._scaled_mm(
            x8,
            self.weight_fp8.t(),
            scale_a=xs.float().reshape(-1, 1),
            scale_b=self.weight_scale,
            out_dtype=hidden_states.dtype,
        )


class _WideningLogitsProcessor(LogitsProcessor):
    """The shell's logits processor (its row gather, its fp32 logits) over `_WidenedHead`, at
    tp1; with `verify_head` set (the dense FP8 mode) the target's own head is that `_Fp8Head`.

    `widen=False` (with `return_context_logits`, the speculative paths) returns the head's own
    bf16 logits: for a consumer that only takes their row argmax, which is the same token over
    bf16 as over their exact fp32 widening (same order, ties, NaN and +-0)."""

    verify_head = None
    #: Set by the shell's forward each engine step: the step's draft logits feed only the
    #: greedy draft sampler's argmax (`_spec_argmax_only`).
    draft_argmax_only = False

    def forward(
        self,
        hidden_states,
        lm_head,
        attn_metadata,
        return_context_logits: bool = False,
        widen: bool = True,
    ) -> torch.Tensor:
        if self.verify_head is not None and not isinstance(lm_head, _Fp8Head):
            lm_head = self.verify_head
        if not widen and return_context_logits:
            return lm_head(hidden_states)
        return super().forward(
            hidden_states, _WidenedHead(lm_head), attn_metadata, return_context_logits
        )


def _spec_argmax_only(spec_worker, spec_metadata) -> tuple[bool, bool]:
    """`(verify, draft)`: whether this step's verify / draft logits are consumed only by a row
    argmax, so they can stay bf16.

    Both need no guided decoder (it edits the logits). The draft logits then feed only the draft
    sampler, an argmax unless rejection sampling runs on a non-greedy batch. The verify logits
    are also the step's `logits` output and the occurrence penalties' and relaxed acceptance's
    input; the one-model speculative sampler admits no request asking for logits or log
    probabilities, so they stay bf16 only for an all-greedy batch with the penalties off (a
    deploy-time flag) and strict acceptance. Every input is a deploy-time constant or
    `is_all_greedy_sample`, part of the CUDA graph key."""
    if getattr(spec_worker, "guided_decoder", None) is not None:
        return False, False
    draft = not spec_metadata.wants_advanced_draft_sampling
    verify = (
        spec_metadata.is_all_greedy_sample
        and not getattr(spec_metadata, "enable_penalty", False)
        and not getattr(
            getattr(spec_worker, "spec_config", None), "use_relaxed_acceptance_for_thinking", False
        )
    )
    return verify, draft


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


class ModelingV2Qwen38FlashNextNvfp4Sm10xDefault(
    DecoderModelForCausalLM[Qwen4ExpDefaultModelingV2, PretrainedConfig]
):
    """Qwen3.8-Flash-Next-NVFP4 at tp1, tp2, tep2, dep2, tp4, tep4, dep4, or dep8 across nodes."""

    # With a `speculative_config`, one-engine speculative decoding takes the engine's fast paths
    # (greedy and temperature-only sampling CUDA graph variants, the overlap loop on the
    # execution stream, hoisted position ids, the MTP-Eagle draft loop's step-0 epilogue, the
    # recorded steady decode-step prepare) at every layout, as the FP8 target does.
    spec_decode_fast_paths: bool = False
    runtime_hooks: RuntimeHooks | None = None
    # Keeps a requested NVFP4 KV pool, which the engine otherwise turns into FP8 on sm_107.
    supports_nvfp4_kv_cache = True

    #: PLE seeds a fresh request's n-gram history and conv state in its prefill only, so the
    #: engine must not run a one-token prompt as a decode row.
    resets_recurrent_state_in_context = True

    # A context chunk that reaches the prompt end takes the Mamba snapshot before it inside
    # the same forward: the GDN layers and the PLE layer split their state at the snapshot
    # point (`PrefillTarget._gdn_inforward`, `_ple_block`). A positive value turns the path
    # on in the cache manager; 1 leaves the fewest chunk tokens before the snapshot point at
    # the manager's own floor (the GDN conv window and the PLE windows).
    inforward_snapshot_min_main = 1
    # The longest tail the split runs: FlashInfer's in-place path. A snapshot further from the
    # prompt end (`additional_snapshot_offsets_from_end`) keeps the two-iteration path.
    inforward_snapshot_max_tail = _GDN_INPLACE_MAX_TOKENS

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Prefer KV cache manager V2: PLE state (`ple_layer_cache`) and QSA's index and
        position planes exist only there."""
        return "V2"

    @staticmethod
    def check_deployment(model_config: ModelConfig, llm_args: TorchLlmArgs, layout: str) -> None:
        """Raise `NoTarget` for a checkpoint or an engine choice this target does not run.

        `route` calls it once the GPU, the parallel layout (`layout`: routing's `layout`) and the
        draft path match, before it returns this class. Where the constructor, the bind or the
        first forward would refuse the deployment, or a same-shape checkpoint would compute
        something else, `auto` then falls back to the built-in implementation.
        """
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
        if shape != _TEXT_SHAPE:
            raise NoTarget(f"text shape {shape}; the target is Qwen3.8-Flash-Next's {_TEXT_SHAPE}")
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
        if vshape != _VISION_SHAPE:
            raise NoTarget(
                f"vision shape {vshape}; the target is Qwen3.8-Flash-Next's {_VISION_SHAPE}"
            )
        differs = {k: v for k, v in _text_settings(text).items() if v != _TEXT_SETTINGS[k]}
        if differs:
            raise NoTarget(
                f"the checkpoint has Qwen3.8-Flash-Next's shape but declares {differs}; the "
                f"target implements { ({k: _TEXT_SETTINGS[k] for k in differs}) }"
            )

        # Read here so a misspelled value fails before the build.
        _ple_host.host_table_mode()
        mapping = model_config.mapping
        one_rank = mapping.world_size == 1
        spec = model_config.spec_config
        prefill_graph = str(llm_args.prefill_cuda_graph_backend.value)
        # The draft loop runs the checkpoint's single MTP module as one linear chain, verified
        # at one width, `max_draft_len + 1` rows a request.
        if spec is not None:
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
                    "acceptance_rate_threshold), but this target's verify step runs at the one "
                    "width max_draft_len + 1"
                )
        if mapping.enable_attention_dp and prefill_graph == "breakable":
            raise NoTarget(
                f"prefill_cuda_graph_backend='breakable' with attention DP ({layout}): the engine "
                "pads every rank to one prefill bucket, so a rank holding only generation "
                "requests is handed more rows than its batch, which the target's decode step "
                "refuses"
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
        # The CuteDSL expert pipeline runs every expert at once (one rank only), where it is built.
        requested = str(model_config.moe_backend_requested).upper()
        cute_dsl = one_rank and archs.fast_path("cute_dsl_moe", torch.cuda.get_device_capability())
        moe_backends = [
            name for name, runs in _TRUNK_MOE_BACKENDS.items() if cute_dsl or runs != "CUTEDSL"
        ]
        if requested not in moe_backends:
            raise NoTarget(
                f"moe_config.backend={model_config.moe_backend_requested!r}; at {layout} the "
                f"target runs the trunk's routed experts on {', '.join(moe_backends)}"
            )
        state_dtype = effective_mamba_ssm_cache_dtype(
            model_config, llm_args.kv_cache_config.mamba_ssm_cache_dtype
        )
        state_dtypes = _ONE_RANK_SSM_STATE_DTYPES if one_rank else _SSM_STATE_DTYPES
        if state_dtype not in state_dtypes:
            served = " or ".join(str(dtype).removeprefix("torch.") for dtype in state_dtypes)
            raise NoTarget(
                f"a {state_dtype} recurrent-state cache; at {layout} the target serves {served} "
                "(kv_cache_config.mamba_ssm_cache_dtype)"
            )
        if llm_args.moe_config.load_balancer is not None and mapping.moe_ep_size > 1:
            raise NoTarget(
                f"moe_config.load_balancer (EPLB) at {layout}: the target runs fixed contiguous "
                "expert windows and registers no layer with the balancer, so the placement "
                "would do nothing"
            )
        strategy = AllReduceStrategy(int(model_config.allreduce_strategy))
        if _pool.attention_tp_size(mapping) > 1 and strategy in (
            AllReduceStrategy.UB,
            AllReduceStrategy.LOWPRECISION,
        ):
            raise NoTarget(
                f"allreduce_strategy={strategy.name} at {layout}: the target calls the all-reduce "
                "op directly, without the buffers the stock AllReduce module sets up for it"
            )
        # As the stock factory: only an attention-DP group has a MoE round trip to override.
        comm_method = os.environ.get(MOE_COMM_METHOD_ENV, "ALLGATHER").upper()
        if mapping.enable_attention_dp and mapping.dp_size > 1 and comm_method != "ALLGATHER":
            raise NoTarget(
                f"{MOE_COMM_METHOD_ENV}={comm_method} at {layout}: the target runs the "
                "attention-DP MoE round trip as ALLGATHER (the default)"
            )
        if _dp_moe_comm() != "nccl" and layout not in _LAMPORT_LAYOUTS.values():
            raise NoTarget(
                f"{DP_MOE_COMM_ENV}={_dp_moe_comm()} at {layout}: the Lamport round trip serves "
                f"{' and '.join(_LAMPORT_LAYOUTS.values())} workers only"
            )
        if getattr(llm_args.multimodal_config, "vision_gpu_normalize", False):
            raise NoTarget(
                "multimodal_config.vision_gpu_normalize: the input processor then ships uint8 "
                "pixels, and the vision tower runs at the bf16 activation dtype"
            )
        if prefill_graph == "piecewise":
            raise NoTarget(
                "prefill_cuda_graph_backend='piecewise' makes the engine torch.compile the model, "
                "which the target does not implement; 'breakable' runs the prefill step as a "
                "breakable CUDA graph"
            )
        mm_config = llm_args.multimodal_config
        if mm_config is not None and mm_config.encoder_attention_backend:
            raise NoTarget(
                "multimodal_config.encoder_attention_backend="
                f"{mm_config.encoder_attention_backend}; the target's vision tower runs its own "
                "attention and does not apply it"
            )
        if spec is None:
            return
        if not is_gdn_replay_enabled():
            raise NoTarget(
                "MTP with TRTLLM_USE_GDN_REPLAY=0: the target verifies its Gated-DeltaNet rows "
                "on the cached replay only, not on the legacy per-step recurrent states; leave "
                "TRTLLM_USE_GDN_REPLAY unset"
            )
        if requested == "CUTEDSL" and spec.max_draft_len > _CUTEDSL_MAX_DRAFT_LEN:
            raise NoTarget(
                f"max_draft_len={spec.max_draft_len} with moe_config.backend=CUTEDSL: the CuteDSL "
                f"expert pipeline is certified at max_draft_len up to {_CUTEDSL_MAX_DRAFT_LEN}"
            )

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None
        # Adopt the engine-normalized text tower (`Qwen4ExpTextConfig`) as the model's config:
        # everything that sizes the pools reads `model_engine.model.model_config.
        # pretrained_config`. The swap is in place because the engine keeps its own reference
        # to this `ModelConfig`.
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
            Qwen4ExpDefaultModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # `lm_head` is outside the quantization scheme: a dense bf16 `LMHead`.
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"expected a bf16 lm_head; the shell built {self.lm_head.weight.dtype}"
        )
        self.one_rank = model_config.mapping.world_size == 1
        if self.one_rank:
            self.logits_processor = _WideningLogitsProcessor()
        # No separate draft checkpoint: the MTP module's weights are the target's own.
        self.draft_config = None
        # The speculative branch, built only when `speculative_config` asked for it.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # Must match what the resource manager decided: left at `get_spec_worker`'s
            # default, the draft step would be handed a manager that maps no draft layer.
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.spec_decode_fast_paths = True
            self.runtime_hooks = _runtime_hooks(self.spec_config, self.model)
            # Under attention TP the TP group's key exchange, as the FP8 target builds it: the
            # worker then takes the target tokens of greedy and temperature-only batches, and
            # the greedy draft tokens, from this rank's vocabulary shard (argmax or Gumbel-max
            # in one exchange) instead of the all-gather. Collective over the TP group (every
            # rank constructs the model in step); None at one rank, under attention DP, without
            # peer memory or with `TLLM_SPEC_VOCAB_PARALLEL_ARGMAX=0`.
            key_exchange = None
            if not self.one_rank and vocab_parallel_argmax_enabled():
                from tensorrt_llm._torch.distributed import ll_exchange

                key_exchange = ll_exchange.create_tp_key_exchange(
                    model_config.mapping, model_config.max_num_tokens
                )
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
                key_exchange=key_exchange,
                spec_decode_fast_paths=self.spec_decode_fast_paths,
            )
            assert self.spec_worker is not None, (
                f"the engine resolved no spec worker for spec_dec_mode "
                f"{self.spec_config.spec_dec_mode!r}"
            )
            if getattr(self.spec_worker, "key_exchange", None) is not None:
                self.draft_model.mtp_layers[0].worker = self.spec_worker
            self.spec_worker.set_draft_model(self.draft_model)
            # The PLE pools and the QSA side caches are written per verify row and nothing the
            # runtime owns rolls them back; the worker calls this after its draft loop, inside
            # the same forward.
            self.spec_worker.register_auxiliary_state_handler(self.model.spec_rollback)
        # Media placeholder ids for the executor; built lazily (`__init__` runs under meta
        # init).
        self._mm_token_ids_value = (self.model.image_token_id, self.model.video_token_id)
        self._mm_token_ids_cache: torch.Tensor | None = None
        # Replicated on every rank; built once the weights are real (`post_load_weights`).
        self.vision_tower: _vision.VisionTower | None = None
        # `DP_HEAD_SHARD_ENV` (opt-in): the attention-DP heads over vocabulary slices, built
        # after load.
        self._dp_head_shard: _DpHeadShard | None = None

    @property
    def samples_from_vocab_shards(self) -> bool:
        """With `DP_HEAD_SHARD_ENV`'s verify head the engine exchanges the attention-DP ranks'
        sampling variants (greedy, temperature-only) and captures the temperature-only one, so
        the members of a group decide alike."""
        shard = self._dp_head_shard
        return shard is not None and bool(shard.verify_rows)

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """Placeholder ids whose rows vision embeddings replace. This checkpoint's
        placeholders are in vocabulary, so the executor's out-of-vocabulary fallback would
        select nothing."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """Tensors under `multimodal_data` the executor moves to the device. Absent, it moves
        everything, putting the `(t, h, w)` grids the tower reads on the host there too."""
        return list(_vision.DEVICE_PATHS)

    def validate_request(self, request) -> None:
        """Admission check, run by the executor on every new request: a media request the
        vision tower would assert on fails here as its own error (`vision.validate_request`);
        a text request returns at once."""
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
        """One engine step (`_step`). With `DP_MOE_COMM_ENV=lamport_step` the whole step --
        trunk, heads and every draft pass -- is one region of the Lamport launch indices."""
        args = (
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            lora_params,
        )
        steps = self.model._lamport_steps
        if steps is None:
            return self._step(*args, **kwargs)
        with steps.region():
            return self._step(*args, **kwargs)

    def _step(
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
        """One engine step: the core's `begin_step`, then the base forward with every
        argument it was handed (`resource_manager` stays in `**kwargs`).

        With a spec worker the shell owns the logits gather (one-model modes gather the rows
        the sampler verifies, `spec_metadata.gather_ids`) and hands the worker the trunk's
        ungathered stream rows, which the MTP module reads; `position_ids` goes in the
        engine's layout."""
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
        assert not self.one_rank or self.model.draft_head is not None, (
            "MTP at tp1 runs its draft steps on the FP8 draft head `_build_heads` makes, but "
            "the core has none"
        )
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
        head_rows = _head_rows(self.model, spec_metadata.gather_ids, final)
        shard = self._dp_head_shard
        shard_tokens = None
        if shard is not None:
            shard.begin_step(spec_metadata)
            shard_tokens = shard.verify_tokens(
                self.spec_worker, spec_metadata, attn_metadata, head_rows
            )
        if shard_tokens is not None:
            logits = _ShardVerifyLogits(shard_tokens, self.model.vocab)
        elif isinstance(self.logits_processor, _WideningLogitsProcessor):
            verify_argmax_only, self.logits_processor.draft_argmax_only = _spec_argmax_only(
                self.spec_worker, spec_metadata
            )
            logits = self.logits_processor.forward(
                head_rows,
                self.lm_head,
                attn_metadata,
                True,
                widen=not verify_argmax_only,
            )
        elif getattr(self.spec_worker, "key_exchange", None) is not None:
            # This rank's vocabulary shard for a greedy or temperature-only batch (the worker
            # marks `spec_metadata.sharded_target_logits`), else the gathered fp32 logits.
            logits = self.spec_worker.target_logits(
                head_rows,
                self.lm_head,
                self.logits_processor,
                attn_metadata,
                spec_metadata,
                self.draft_model,
            )
        else:
            logits = self.logits_processor.forward(
                head_rows,
                self.lm_head,
                attn_metadata,
                True,
            )
        # The draft head decides from the same batch whether its logits may stay sharded.
        self.draft_model.mtp_layers[0].spec_metadata = spec_metadata
        # A prefill CUDA graph pads the step's rows; the worker reshapes generation rows per
        # request, so it takes the batch's own.
        if attn_metadata.padded_num_tokens is not None:
            n = attn_metadata.num_tokens
            input_ids, hc = input_ids[:n], hc[:n]
            position_ids = position_ids[..., :n]
        outputs = self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hc,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )
        if shard_tokens is not None and isinstance(outputs, dict):
            # The engine keeps the step's `logits` output (and slices it on an idle rank); nothing
            # reads its values, since the one-model sampler admits no request asking for logits.
            if isinstance(outputs.get("logits"), _ShardVerifyLogits):
                outputs["logits"] = shard.placeholder[: shard_tokens.shape[0]]
        return outputs

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        self.vision_tower = _vision.VisionTower(self.model.w, self.model.vision_config)
        self.vision_tower.warm_up()
        if self.one_rank:
            self._build_heads(self.lm_head.weight.data)
        elif self.spec_worker is not None and self._dp_head_shard is None:
            # Collective where set (every rank of the attention-DP group builds it).
            self._dp_head_shard = _dp_head_shard_build(self, self.lm_head.weight.device)

    def _build_heads(self, weight: torch.Tensor) -> None:
        """The tp1 heads' copies of `weight`: the draft head's FP8 copies (per row, and packed
        for `fp8_gemv` unless its NVFP4 copy replaces it, `_MTP_HEAD_NVFP4_ENV`), and under the
        dense FP8 mode the verify head over the same per-row copy with its packed copy or its
        NVFP4 one (`_TARGET_HEAD_NVFP4_ENV`); every decode variant compiled now."""
        dense_fp8 = _switch(_DENSE_FP8_ENV)
        draft = self.model.mtp_enabled
        if not (draft or dense_fp8):
            return
        rowwise = _fp8_rowwise(weight)
        draft_nv = draft and _switch(_MTP_HEAD_NVFP4_ENV)
        target_nv = dense_fp8 and _switch(_TARGET_HEAD_NVFP4_ENV)
        packed = (
            pack_from_bf16(weight)
            if (draft and not draft_nv) or (dense_fp8 and not target_nv)
            else None
        )
        nv = None
        if draft_nv or target_nv:
            nv = nvfp4_pack_from_bf16(weight)
            if weight.is_cuda:
                nvfp4_gemv_precompile(nv)
        if weight.is_cuda:
            mid_rows_precompile(_head_mid(*rowwise), _FP8_GEMV_HEAD_ROWS + 1, _MID_HEAD_ROWS)
        if draft:
            self.model.draft_head = _Fp8Head(*rowwise, packed, nv if draft_nv else None)
        if dense_fp8:
            self.logits_processor.verify_head = _Fp8Head(
                *rowwise, packed, nv if target_nv else None
            )
        if target_nv:
            print(
                f"target head NVFP4 ({_TARGET_HEAD_NVFP4_ENV}=1): decode rows <= "
                f"{_FP8_GEMV_HEAD_ROWS} on nvfp4_gemv over lm_head {tuple(nv.data.shape)}",
                flush=True,
            )
