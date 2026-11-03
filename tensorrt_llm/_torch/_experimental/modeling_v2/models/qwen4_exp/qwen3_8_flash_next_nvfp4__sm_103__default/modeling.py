# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 (hf-fc694b5_orig) / sm_103 / default.

One target for the layouts `routing.py` admits in one node: one rank (tp1, every head and
expert); at four ranks tp4 (attention TP4, routed experts split by intermediate), tep4
(attention TP4, experts EP4) and dep4 (attention DP, experts EP4); at two ranks dep2 (attention
DP, experts EP2), the layout of both disaggregated-serving engines, which is dep4 at half the
group. The core holds the static
structure (this rank's share of the weights, derived operands, per-manager pool views);
`PrefillTarget` (any batch with context rows) and `DecodeTarget` (generation only) run a step
over it, passing every argument to every op at the call site.

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
  (`_expert_sum_op`). The router and the bf16 shared expert are replicated and run on this
  rank's rows, so the block output `routed + sigmoid(gate) * shared` is formed by the next
  hyper-connection site on the reduced routed rows.
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

**QSA sparse attention.** The index plane is maintained on every step so a request can
cross `dense_seq_len_threshold` later without rebuilding its prefix; above it the sparse
core (index scores, block top-k and expansion, sparse GQA) replaces the dense op, and the
main K/V append becomes this layer's own call. The raw index key is stored before anything
normalizes it: the compression applies the norm and the rotation itself.

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

import math
import os
import re
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    MAX_CERTIFIED_TOKENS as _MOE_MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_nvfp4_pages_to_bf16 import (
    qsa_nvfp4_pages_to_bf16,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (
    qsa_paged_index_scores,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_store import (
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_kv_store import (
    qsa_paged_kv_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_kv_store_nvfp4 import (
    qsa_paged_kv_store_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_decode import (
    qsa_paged_sparse_gqa_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_decode import (
    supports as _sparse_decode_supports,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4 import (
    qsa_paged_sparse_gqa_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_verify import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_verify,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_verify import (  # noqa: E501
    takes_grouped as _verify_takes_grouped,
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_sparse_gqa_rowtile import (
    qsa_sparse_gqa_rowtile as one_rank_sparse_gqa_rowtile,
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_attn_decode_prologue import (  # noqa: E501
    SCORE_ROWS as _PROLOGUE_SCORE_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_attn_decode_prologue import (  # noqa: E501
    qwen4_exp_attn_decode_prologue as one_rank_attn_decode_prologue,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qwen4_exp_qk_norm_rope import (
    qwen4_exp_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.spec_rollback_commit import (
    spec_rollback_commit,
    spec_rollback_stage,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.verify_step_setup import (
    _BLOCK_S as _VERIFY_SETUP_MAX_SEQS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.verify_step_setup import (
    verify_step_setup,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import (
    MAX_ROWS as _GEMV_MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import (
    TILING as _GEMV_TILING,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.bf16_gemv_pdl import (
    bf16_gemv_pdl,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.fp8_gemv import (
    fp8_gemv_conv1d_op,
    fp8_gemv_conv1d_verify_op,
    fp8_gemv_op,
    pack_from_bf16,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.fp8_per_tensor_mm import (
    fp8_per_tensor_mm,
    fp8_per_tensor_weight,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.low_m_bf16_mm import (
    low_m_bf16_mm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    MidWeight,
    mid_rows_mm_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    precompile as mid_rows_precompile,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    nvfp4_gemv_op,
    nvfp4_split_gemv_conv1d_op,
    nvfp4_split_gemv_conv1d_verify_op,
    nvfp4_split_gemv_op,
    precompile_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    pack_from_bf16 as nvfp4_pack_from_bf16,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    pack_split_from_bf16 as nvfp4_pack_split_from_bf16,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.gemm.nvfp4_gemv import (
    precompile as nvfp4_gemv_precompile,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_first_cute import (
    qwen4_exp_hc_decode_first_cute,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_mix_fp4_cute import (
    qwen4_exp_hc_decode_mix_cute,
    qwen4_exp_hc_decode_mix_fp4_cute,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_rows_tc import (
    MAX_ROWS as _HC_DECODE_ROWS_MAX,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_decode_rows_tc import (
    qwen4_exp_hc_decode_rows_first,
    qwen4_exp_hc_decode_rows_mix,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_first_rescale_tc import (
    qwen4_exp_hc_first_rescale_tc,
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
    qwen4_exp_hc_up_gate_mean_rstd_fp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_post_inject import (
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import (  # noqa: E501
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.cute_dsl_nvfp4_grouped_gemm import (
    cute_dsl_nvfp4_grouped_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_permuted import (
    fp4_block_scale_moe_renorm_permuted,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (
    fp4_block_scale_moe_renorm_routed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp4_moe_renorm_cute_dsl import (
    fp4_moe_renorm_cute_dsl,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp8_block_scale_moe_runner import (
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fused_sigmoid_gate_mul_add import (
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.moe_sort import moe_sort
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.moe_unpermute_inplace import (
    moe_unpermute_inplace,
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_triton_update import (
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.cu_seqlens_from_seq_lens import (
    cu_seqlens_from_seq_lens,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.flashinfer_chunk_gated_delta_rule import (
    INPLACE_MAX_TOKENS as _GDN_INPLACE_MAX_TOKENS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.flashinfer_chunk_gated_delta_rule import (
    flashinfer_chunk_gated_delta_rule,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_gated_norm_pdl import (
    MAX_ROWS as _GDN_DECODE_FUSED_MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_gated_norm_pdl import (
    gdn_decode_gated_norm_pdl,
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
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import (
    FLASHINFER_REPLAY_RING_SIZE,
    flashinfer_cached_replay_update,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._torch.utils import ActivationType, is_gdn_replay_enabled
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import pool as _pool
from . import vision as _vision
from . import weights as _weights
from .ops.attention.qsa_fp8_pages_to_bf16__prefill import qsa_fp8_pages_to_bf16
from .ops.attention.qsa_sparse_gqa_rowtile__prefill import qsa_sparse_gqa_rowtile
from .ops.attention.qwen4_exp_attn_decode_prologue__decode import qwen4_exp_attn_decode_prologue
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import MAX_ROWS as _HC_TC_FIRST_ROWS
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import precompile as hc_tc_first_precompile
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import qwen4_exp_hc_decode_tc_first
from .ops.hc.qwen4_exp_hc_up_gate_mean_fp4__moe import qwen4_exp_hc_up_gate_mean_fp4
from .ops.ssm.gdn_cached_replay_cute__verify import gdn_cached_replay_cute
from .ops.ssm.gdn_cached_replay_cute__verify import precompile as gdn_replay_precompile

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The GPU architecture this target is certified for; a direct instantiation could
# bypass routing.
_SM = (10, 3)

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
_QSA_POSITION_AXES = _pool.QSA_POSITION_AXES
# The compressed index keys rotate at coordinate 0 alone (no mRoPE section), as the index
# query does.
_QSA_MROPE_SECTION = None
# The draft experts' activation scales are fp32, not the UE8M0 power-of-two form.
_MTP_MOE_USE_UE8M0 = False
# Rows per draft FP8 runner call: its 128-expert windows are certified up to one 8192-row
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
#: tp1; 0 (default): the FP8 block-scale runner.
_MTP_MOE_NVFP4_ENV = "TRTLLM_MODELING_V2_MTP_MOE_NVFP4"

#: The NVFP4 draft MoE's two activation global scales are static, `448 * 6 / A` with `A` this factor
#: times the largest calibrated amax of the trunk's routed experts (its `input_scale`s, FC1 input and
#: FC2 input; the draft layer carries none). A global scale only has to keep the e4m3 block scales
#: inside their range, so the margin costs nothing until blocks are ~1e-3 of `A`.
_MTP_MOE_NVFP4_FC1_MARGIN = 8.0
_MTP_MOE_NVFP4_FC2_MARGIN = 4.0

_PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"

#: The opt-in switches certified at tp1; every other layout runs the checkpoint's bf16 paths.
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
# Whether one rank keeps that second copy (about 1.3 GiB a layer). No engine arg owns it, hence
# an env var read at construction (`_switch`, default on); off drops the copy and every launch
# runs on trtllm-gen.
_MOE_PREFILL_COPY_ENV = "TRTLLM_MODELING_V2_MOE_PREFILL_COPY"


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


def _ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is declared in host memory (the default)."""
    return _switch(_PLE_HOST_OFFLOAD_ENV, default=True)


def _moe_layer_scalars(w, i: int, start: int, count: int) -> dict[str, torch.Tensor]:
    """Layer `i`'s routed-expert scalars for the trtllm-gen runner over this rank's window
    `[start, start + count)`, built from the checkpoint's `input_scale` / `weight_scale_2`
    rows (declared over the whole routing space on every rank).

    The checkpoint stores modelopt reciprocals (`input_scale = 1 / g_act`, `weight_scale_2 =
    1 / g_w`), so a quantizer's global scale is `1 / input_scale` and the runner's scalars
    are products of the two."""
    isc1, isc1_up = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc1_up"]
    assert torch.equal(isc1, isc1_up), (
        f"layer {i}: expert gate/up input_scale differ; the fused FC1 reads one activation "
        "quantization for both halves"
    )
    assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
        f"layer {i}: expert gate/up weight_scale_2 differ; the fused FC1 carries one alpha "
        "per expert for both halves"
    )
    # One activation quantization feeds all experts, on every rank.
    assert torch.equal(isc1.min(), isc1.max()), (
        f"layer {i}: the routed experts do not share one gate_proj.input_scale, so a single "
        "activation quantization cannot feed them all"
    )
    e_isc2, ws2_2 = w[f"l{i}_e_isc2"], w[f"l{i}_e_ws2_2"]
    gate1 = isc1 * w[f"l{i}_e_ws2_1"]
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
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
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
        # At one rank the CuteDSL-layout copy of the experts for prefill-sized launches
        # (`_moe_prefill_copy`); `weights.py` declares it from this.
        self.moe_prefill_copy = _moe_prefill_copy(
            self.one_rank, self.trunk_moe_backend, model_config.max_num_tokens
        )
        # The draft experts' NVFP4 copy at one rank (`_mtp_moe_nvfp4`), built once: it frees
        # the FP8 experts, and the shell's post-load walk reaches `post_load_weights` again.
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
        self.ple_table_dtype = _weights.PLE_TABLE_DTYPE
        self.ple_host_offload = _ple_host_offload_enabled()
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
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
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
        table_rows = w["ple_table"].shape[0]
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
            # The prefill CuteDSL MoE's output memset, beside its FC1 (`fp4_moe_renorm_cute_dsl`):
            # its own stream, so the join before FC2 does not also wait for the shared expert.
            self._moe_memset_stream = torch.cuda.Stream(device=device)
            self.tp_sum = self._tp_sum_op()
            self.expert_sum = self._expert_sum_op()
        # A decode step's bf16 projection entry, and its GDN in_proj's: at tp1 the one-rank
        # dispatch (`_one_rank_projection`), at the other layouts the shapes' own.
        one_rank = self.one_rank
        self.decode_mm = _one_rank_projection if one_rank else _decode_projection
        # At one rank a trtllm-gen decode MoE leaves its finalize to the next site
        # (`DecodeTarget._moe`).
        self.moe_fold = one_rank and self.trunk_moe_backend == "TRTLLM"
        # At one rank the hyper-connection sites' fast forms by row count (the decode GEMV and
        # tcgen05 rows pairs, the prefill rstd sites); the other layouts do not take them.
        self.hc_fast_sites = one_rank
        self.decode_in_mm = (
            _one_rank_projection
            if one_rank
            else (lambda mat_a, mat_b: low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL))
        )
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _build_draft_one_rank(self, device) -> None:
        """At one rank the draft layer's operands for `DecodeTarget._mtp_layer`: the fused
        `[q | k | v | gate | index | pad]` attention input (`weights._attn_in_w`), with
        `_MTP_MOE_NVFP4_ENV` its routed experts in NVFP4, and with `_MTP_DENSE_FP8_ENV` the
        packed FP8 copies of its dense projections; None / empty at every other layout."""
        self._mtp_attn_in = None
        self._mtp_gemv8 = {}
        if not self.one_rank:
            return
        w = self.w
        p = f"l{self.mtp_layer_idx}_mtp_"
        attn_in = _weights._attn_in_w(
            (w[f"{p}q_w"], w[f"{p}k_w"], w[f"{p}v_w"], w[f"{p}idx_w"]), self
        )
        self._mtp_attn_in = attn_in.t()
        if device.type != "cuda":
            return
        if _switch(_MTP_MOE_NVFP4_ENV):
            if self._mtp_moe_nvfp4 is None:
                self._mtp_moe_nvfp4 = _mtp_moe_nvfp4(self)
            self._mtp = self._mtp._replace(**self._mtp_moe_nvfp4)
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
        the engine's `allreduce_strategy`), which every rank gets the same bits back from;
        otherwise nothing to sum."""
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

        def round_trip(run, x: torch.Tensor, pad_rows: int) -> torch.Tensor:
            rows = x.shape[0]
            padded = x if not pad_rows else nn.functional.pad(x, [0, 0, 0, pad_rows])
            gathered = allgather(padded, None, group, process_group)
            mine = reducescatter(run(gathered), None, group, process_group)
            return mine if not pad_rows else mine[:rows]

        return round_trip

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
        kind = _weights.memory_kind(table.data_ptr())
        expected = (
            _weights.CUDA_MEMORY_TYPE_HOST
            if self.ple_host_offload
            else _weights.CUDA_MEMORY_TYPE_DEVICE
        )
        assert kind == expected, (
            f"the n-gram table was declared with {_PLE_HOST_OFFLOAD_ENV}="
            f"{'1' if self.ple_host_offload else '0'}, but the driver reports cudaMemoryType "
            f"{kind} for the pointer the forward gathers from"
        )
        mib = 1024.0 * 1024.0
        free, total = torch.cuda.mem_get_info()
        print(
            f"ple n-gram table: {tuple(table.shape)} {table.dtype}, "
            f"{table.numel() * table.element_size() / mib:.1f} MiB, "
            f"{'HOST-resident (device-addressable)' if self.ple_host_offload else 'DEVICE-resident'}"
            f"; after load: allocated {torch.cuda.memory_allocated() / mib:.1f} MiB, "
            f"driver-used {(total - free) / mib:.1f} MiB of {total / mib:.1f} MiB",
            flush=True,
        )

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
        if self.mtp_enabled and state_dtype == torch.bfloat16:
            # Every CuTe replay kernel a verify step of up to `max_num_requests` requests
            # launches (`_gdn_verify`), compiled now: the captured decode graphs do not reach
            # every launch mapping, and a compile in a serving step stalls it 5-10 s.
            gdn_replay_precompile(
                self.max_verify_rows,
                self.num_k_heads,
                self.num_v_heads,
                md.max_num_requests,
                pdl=_GDN_REPLAY_PDL,
                partition_from=CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
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
        bound = _Bound(cm, rank_pools)
        if self.spec_rollback is not None:
            self.spec_rollback.set_index_planes(
                [rank_pools.index_cache[i] for i in self.attn_layers]
            )
        if self.mtp_enabled:
            bound = bound._replace(**self._bind_step_buffers(cm, g, mm, rank_pools))
        self._bound = bound

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
        _pool.check_replay_state(
            history, width, meta.prev_num_accepted_tokens, meta.cache_buf_idx, ssm0
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
            _pool.check_replay_history(g, i, history, ssm0.shape[0], *old[i][:3])
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
    # With `_MTP_MOE_NVFP4_ENV` (one rank), `fc1_w` .. `fc2_s` are the NVFP4 copy in the trunk's
    # layout and these its activation global scale and the runner's three scalars.
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


def _verify_state(core, mm, num_decodes: int) -> tuple:
    """What a verify step's `_gdn_verify` reads besides the layer: the rows
    `update_mamba_states` promotes the conv windows from, and the replay's work items. Work
    items go to the kernel from `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests on, and the
    kernel leaves the history commit to the manager's all-layer commit exactly when the
    manager runs one after the step -- the stock Gated-DeltaNet layer's two conditions. The
    FlashInfer ring replay folds in-kernel and reads no work items."""
    assert core._bound.int_idx is not None, "the cache-manager bind has not run"
    int_idx = core._bound.int_idx[:num_decodes]
    if core._bound.replay.ring is not None:
        return int_idx, None, None, False
    partitioned = num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    all_layer = partitioned and bool(
        getattr(core._bound.cm, "use_gdn_cached_replay_all_layer_commit", False)
    )
    work_items = mm.replay_work_items[:num_decodes] if partitioned else None
    n_writes = mm.replay_n_writes if partitioned else None
    return int_idx, work_items, n_writes, all_layer


def _gdn_verify(core, mq, a, b, i, L, si_d, n, p, verify, mq_out=None) -> torch.Tensor:
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
    epilogue (`fp8_gemv_conv1d_verify`, `nvfp4_split_gemv_conv1d_verify`; `mq` is then None)."""
    int_idx, work_items, n_writes, all_layer = verify
    pools = core._bound.pools
    replay = core._bound.replay
    ssm = pools.ssm[i]
    cute = ssm.dtype == torch.bfloat16
    # Only the CuTe replay is launched as the conv's programmatic dependent.
    pdl = _GDN_REPLAY_PDL and cute and replay.ring is None
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
    if cute:
        replay_op = gdn_cached_replay_cute
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
    id reads row 0 instead of past the table; every valid id embeds as before."""
    return nn.functional.embedding(input_ids.clamp_min(0), embed_tokens)


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
    return _index_query(core, index_scratch)


def _mtp_moe_block(core, M: _MtpWeights, x, pad_rows: int = 0, routed_only=False) -> torch.Tensor:
    """The draft layer's MoE: renormalized top-k over the whole routing space, the FP8
    block-scale routed experts over this rank's window (1x128 activation scales, pre-routed),
    and the replicated bf16 shared expert in the trunk's form -- one GEMM forms gate|up and
    the per-token gate logit -- combined as `routed + sigmoid(gate logit) * shared`. The
    activation scales come back transposed `[hidden / 128, tokens]` and go straight to the
    runner.

    The window's partial is completed as the trunk's is: under attention TP summed over the
    TP group; under attention DP by the round trip over the group's gathered rows
    (`core.expert_sum`), `pad_rows` padding this rank's (`_draft_pad_rows`). The routing,
    the quantization and the runner are row-wise and run on the rows the window sees, in
    chunks of at most `_MTP_MOE_MAX_ROWS`: one chunk at a TP rank's rows, several over the
    rows an attention-DP group gathers."""

    def run(rows: torch.Tensor) -> torch.Tensor:
        parts = []
        for lo in range(0, rows.shape[0], _MTP_MOE_MAX_ROWS):
            chunk = rows[lo : lo + _MTP_MOE_MAX_ROWS]
            logits = cublas_mm(chunk, M.router)
            topk_ids, topk_w = renorm_moe_routing_op(
                logits, topk=core.topk, output_dtype=chunk.dtype
            )
            xq, xsf = fp8_quantize_1x128(chunk, use_ue8m0=_MTP_MOE_USE_UE8M0)
            parts.append(
                fp8_block_scale_moe_runner(
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
                    local_expert_offset=core.mtp_expert_offset,
                    local_num_experts=core.mtp_local_experts,
                    routed_scaling_factor=None,
                    routing_method_type=_ROUTING_METHOD_RENORM,
                    topk_weights=topk_w,
                    topk_ids=topk_ids,
                    act_type=_ACT_TYPE_SWIGLU,
                    gemm1_clamp_limit=None,
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    if M.moe_gscale is not None:
        # At one rank under `_MTP_MOE_NVFP4_ENV` the experts are their NVFP4 copy.
        routed = _mtp_routed_nvfp4(core, M, x)
    elif core.dp_size == 1:
        routed = core.tp_sum(run(x))
    else:
        routed = core.expert_sum(run, x, pad_rows)
    if routed_only:
        return routed
    gug = cublas_mm(x, M.sh_gug)
    shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), M.sh_dn)
    gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
    return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)


def _mtp_routed_nvfp4(core, M: _MtpWeights, x) -> torch.Tensor:
    """The draft layer's routed experts on their NVFP4 copy (`_mtp_moe_nvfp4`, one rank):
    the activation quantized at the static FC1 input global scale, then the trunk's
    logits-routed trtllm-gen runner over every expert, finalized."""
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
    )


def _mtp_moe_nvfp4(core) -> dict:
    """The draft layer's routed experts requantized to NVFP4 (`_MTP_MOE_NVFP4_ENV`), as the
    `_MtpWeights` fields `_mtp_routed_nvfp4` reads; frees the FP8 copies. The runner's scalars
    follow `_moe_layer_scalars` with `1 / g_in` and `1 / g_mid` standing for the checkpoint's
    `input_scale`s and `1 / g_w` for its `weight_scale_2`: `out1_gate = 1 / (g_in * g_w1[e])`,
    `out1 = out1_gate * g_mid`, `out2 = 1 / (g_mid * g_w2[e])`. The activation scales are static
    (`_MTP_MOE_NVFP4_FC1_MARGIN` / `_FC2_MARGIN`)."""
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"
    keys = [f"{p}{k}" for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")]
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
    (`qwen4_exp_hc_inject_norm`, `_hc_mixer`)."""
    x, normed = qwen4_exp_hc_inject_norm(
        x, core.tp_sum(cublas_mm(attn, M.o)), inject, M.mhc_norm, core.eps, core.hc_count
    )
    mixed, inject = _hc_mixer(core, normed, M.mhc_first, M.mhc_up, True)
    return qwen4_exp_post_inject(x, _mtp_moe_block(core, M, mixed, pad_rows), inject)


def _decode_projection(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """A generation-only step's dense bf16 projection at every layout but tp1 (the attention
    input and output projections and the GDN output projection of `DecodeTarget`, which the
    engine captures in CUDA graphs; tp1 takes `_one_rank_projection`).

    Up to `_GEMV_MAX_ROWS` rows of the o / GDN out projection at a tp4/tep4 rank (K=1536) run
    `bf16_gemv_pdl`: its weight streams into registers before it waits on the producer.
    Everything else runs `low_m_bf16_mm`, whose table picks FlashInfer's low-M kernels at the
    rows and shapes it names (a tp4/tep4 rank's, and the whole layer's that a dep4 / dep2 rank
    holds) and cuBLAS elsewhere. Both launch with PDL per `_DECODE_PROJ_PDL` and read the
    activation only after `griddepcontrol.wait`."""
    if mat_a.shape[0] <= _GEMV_MAX_ROWS and mat_b.shape[0] == _GEMV_TP4_K:
        return bf16_gemv_pdl(mat_a, mat_b, _DECODE_PROJ_PDL)
    return low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL)


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


def _hc_first(core, normed, first, with_inject):
    """A hyper-connection mixer's first half: the first projection over
    `[down; block_inject; 0]` (fp32 logits), then the SiLU and injection activations.
    Returns `(low, inject)`; `inject` is None without `with_inject`. Up to
    `_HC_TC_FIRST_MAX_ROWS` rows (a per-capture constant) it is one tcgen05 launch; larger
    steps run cuBLAS's fp32 logits and the activation kernel."""
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
    """One hyper-connection mixer from the site's normed stream: `_hc_first`, then the up
    projection with the sigmoid gate and the stream mean in its epilogue. Returns
    `(mixed, inject)`; `inject` is None without `with_inject` (the stack-final mixer)."""
    low, inject = _hc_first(core, normed, first, with_inject)
    return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count), inject


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
        return _decode_router if captured else cublas_mm
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
    instead, in one call. On the CuteDSL backend (one rank) every launch is
    `_cute_dsl_routed`, in one call."""
    rows = x.shape[0]
    if core.trunk_moe_backend == "CUTEDSL":
        xq, xsf = fp4_quantize(x, L.moe_gscale, _SF_VEC, False, False) if fp4 is None else fp4
        routed_out = torch.empty([rows, core.hidden], dtype=x.dtype, device=x.device)
        return _cute_dsl_routed(core, L, router_mm(x, L.router), xq, xsf, routed_out)
    if L.fc1_w_cd is not None and _CUTE_DSL_MOE_MIN_ROWS <= rows <= _CUTE_DSL_MOE_MAX_ROWS:
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
    gug = cublas_mm(x, L.sh_gug)
    shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
    return shared, gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]


def _moe_block(
    core, x, L: _LayerWeights, pad_rows: int, fp4=None, router_mm=cublas_mm, decode_gemm=None
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
    entry, under which the shared expert reads the dense FP8 mode's copies when they exist."""
    main = torch.cuda.current_stream()
    side = core._side_stream
    # Fork: the side stream reads `x` only after everything queued on `main` so far. What it
    # allocates is consumed on `main` after the join below.
    side.wait_stream(main)
    with torch.cuda.stream(side):
        shared, gate_logits = _shared_expert(core, x, L, decode_gemm)
    routed = core.expert_sum(
        lambda rows: _routed_experts(core, rows, L, fp4, router_mm), x, pad_rows
    )
    main.wait_stream(side)
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
    core, scores, visible_blocks, logical, seq_lens_q, topk_indices, row_starts, out, live_cols=None
):
    """The selected token indices of `scores`' rows into `out` `[rows, token_topk +
    compress_ratio - 1]`: the canonical block top-k and its expansion in one launch, or, for
    a score table that kernel does not serve, the selection and the expansion apart (the
    rewrite bounded by `live_cols`, `_select`)."""
    if _topk_expand_supports(scores):
        qsa_topk_expand_decode(
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
            scores = qsa_paged_index_scores(
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


def _sparse_gqa_rowtile(
    core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages
):
    """The row-tile pass over a context launch, or None where it does not serve it. A bf16
    pool is read in place. Over an fp8 pool, request slot `s`'s first `ceil((p_s + 1) / tpb)`
    pages (`p_s` the highest position the launch's rows hold there) are widened, exactly, into
    a bf16 scratch pool addressed through a table of `block_table`'s shape; `fp8_pages` is
    their host count (`_fp8_context_pages`), read back from the device when not given. A
    scratch past `_FP8_ROWTILE_MAX_SCRATCH` bytes declines (None)."""
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
    """Sparse GQA over the layer's paged pool. `query_positions` is a context row's bound on
    its scan (the prefill pair with `compress_ratio`), or None for decode rows. A context
    launch of `_ROWTILE_MIN_ROWS` rows or more over a bf16 or fp8 pool takes the row-tile
    pass (`fp8_pages`: see `_sparse_gqa_rowtile`). Over an NVFP4 pool context rows take the
    prefill kernel and decode rows the split-K decode kernel."""
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
    # `_live_score_columns`: a bound on every row's selection window.
    live_cols: int | None = None


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
        # The packed `[q | k | v]` as one compact copy: the dense op reads it flat, and the
        # sparse GQA reads its query heads from it.
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
                    step.fp8_pages,
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
        rows gather fp8 vectors from one table (`ple_ngram_embed`); the key/value projections,
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
        if step.stage:
            core.spec_rollback.stage_ple_verify(ngram_pool, conv_states, step.input_ids)
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
        )
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
        hc_out[num_tokens:].zero_()
        normed_out[num_tokens:].zero_()

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
            _verify_state(core, mm, num_decodes) if is_verify else None,
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
        if (
            sparse
            and core.quant_mode == QM_FP8_KV_CACHE
            and num_prefill_tokens >= _ROWTILE_MIN_ROWS
        ):
            fp8_pages = _fp8_context_pages(md, md.tokens_per_block)
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
            nvfp4_context=nvfp4_context,
            live_cols=_live_score_columns(core, md, batch),
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
            scores = qsa_paged_index_scores(
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
        if fused_prologue and not stage and num_tokens <= _PROLOGUE_SCORE_ROWS:
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

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage):
        """One gated GQA layer plus its QSA index branch for a generation-only batch, at this
        rank's heads.

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
        proj = _dense_mm(
            x, L.attn_in, L.attn_in8, core.decode_mm, "in", L.attn_in_g, mid=L.attn_in_m
        )
        qkv_view, gate, _ = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        pools = core._bound.pools
        # Over a bf16 or e4m3 pool the sparse path's norm + rope, packed copy, K/V append,
        # index store and compression are one launch; it writes the packed buffer itself.
        prologue = sparse and pools.kv[i].k_scale is None
        if stage:
            core.spec_rollback.stage_index(pools.position_cache)
        if prologue:
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
            # Both branches leave the normalized, rotated K/V in the packed copy.
            k_rows, v_rows = torch.split(
                torch.reshape(qkv, [qkv.shape[0], -1, core.head_dim])[:, core.heads_q :],
                core.heads_kv,
                1,
            )
            if not self._kv_probe.observe(i, k_rows, v_rows):
                self._kv_probe = None

        if sparse:
            kv = pools.kv[i]
            if not prologue:
                _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            scores = qsa_paged_index_scores(
                q=_index_query(core, index_scratch),
                index_cache=core._bound.pools.index_cache[i],
                block_table=block_table,
                query_positions=logical,
                request_indices=req_idx,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=False,
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
            attn = _sparse_gqa(
                core,
                kv,
                _packed_q(core, qkv),
                block_table,
                selected,
                req_idx,
                tokens_per_block,
                None,
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
        fused_sigmoid_mul_inplace(out, gate)
        return core.tp_sum(_dense_mm(out, L.o, L.o8, core.decode_mm, "out", L.o_g, mid=L.o_m))

    def _hc_site(self, hc, block_out, inject, norm, first, up, with_inject, normed=None, moe=None):
        """One hyper-connection site: the previous block's injection (none at the stack's first
        site and after the PLE add), this site's grouped norm and its mixer. Returns `(hc, mixed,
        inject)`. At one rank (`hc_fast_sites`) the row count, a per-capture constant, picks the
        form: one row runs the CuTe GEMV pair, up to `_HC_DECODE_ROWS_MAX` rows the tcgen05 rows
        pair, larger batches `qwen4_exp_hc_inject_norm` and `_hc_mixer`, which every batch takes
        at the other layouts. `normed`, when given, is this site's grouped
        norm of `hc` already computed (the PLE tail's, no injection pending), which only the
        inject-norm route reads (the first kernels of the other two compute it themselves).

        After a MoE block `block_out` is its routed output and `moe` the `(gate_logits, shared,
        topk_weights, topk_rows)` `_moe` left uncombined: the site's first kernel forms the
        block output itself; with the top-k pair `block_out` is the runner's permuted FC2 rows
        and that kernel also forms their finalize."""
        core = self.core
        gate_logits, shared, topk_w, topk_rows = moe if moe is not None else (None,) * 4
        n_cols = core.hc_lowrank + (core.hc_count if with_inject else 0)
        fast = core.hc_fast_sites
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
        mixed, inject = _hc_mixer(core, normed, first, up, with_inject)
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
    ) -> torch.Tensor:
        """The draft layer at one rank on the decode machinery: the front end's two projections,
        both hyper-connection sites as `_hc_site` (the front end's per-stream `x_h + x_e` is the
        first site's injection at unit weight, the attention output the second's), the attention
        as `_mtp_attention`, the MoE as `_mtp_moe`. Returns the stream with the MoE output
        injected."""
        core = self.core
        M, G = core._mtp, core._mtp_gemv8
        gemm = core.decode_mm if md.is_cuda_graph else cublas_mm
        e = _draft_embedding(input_ids, embed_tokens)
        x_e = _draft_mm(
            flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e, G.get("fc_e"), gemm
        )
        _, h_n = qwen4_exp_hc_inject_norm(
            hidden_states, None, None, M.pre_h_norm, core.eps, core.hc_count
        )
        x_h = torch.reshape(
            _draft_mm(
                torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h, G.get("fc_h"), gemm
            ),
            [rows, core.stream],
        )
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
        )
        x, mixed, inject = self._hc_site(x, attn, inject, M.mhc_norm, M.mhc_first, M.mhc_up, True)
        return qwen4_exp_post_inject(x, self._mtp_moe(mixed), inject)

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
    ) -> torch.Tensor:
        """The draft layer's gated GQA and QSA index branch over the draft manager's planes
        (`draft`), as `_attention_one_rank` runs a trunk layer: one input projection, the norm +
        rope of both planes in one launch at the module's rope base, the index store and
        compression on the side stream, the K/V append, the index scores and the one-launch
        top-k + expansion, the gated sparse GQA, the output projection. Under IndexShare a draft
        decode step (`reuse`) skips the index branch and rebuilds its selection from the
        draft-extend step's capture; the draft-extend step (`share`) captures it."""
        core = self.core
        M = core._mtp
        dt, dev = x.dtype, x.device
        proj = _draft_mm(x, core._mtp_attn_in, core._mtp_gemv8.get("attn_in"), gemm)
        qkv_view, gate, index_view = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        q_dst, k_dst, v_dst = _attn_in_heads(core, proj)
        index_scratch = torch.empty([rows, core.index_scratch_width], dtype=dt, device=dev)
        q_index = _index_query(core, index_scratch)
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
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
        main.wait_stream(side)
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
                scores = qsa_paged_index_scores(
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

    def _mtp_moe(self, x) -> torch.Tensor:
        """The draft layer's MoE at one rank: the FP8 block-scale routed experts (or with
        `_MTP_MOE_NVFP4_ENV` the NVFP4 ones on the trunk's logits-routed runner) and the bf16
        shared expert (its dense projections on `_draft_mm`), combined as `routed +
        sigmoid(gate logit) * shared`; the NVFP4 path runs the shared expert on the side
        stream."""
        core = self.core
        M, G = core._mtp, core._mtp_gemv8

        def shared_expert():
            gu = _draft_mm(x, M.sh_gu, G.get("sh_gu"), cublas_mm)
            shared = _draft_mm(
                silu_and_mul_rows(gu, core.shared_inter), M.sh_dn, G.get("sh_dn"), cublas_mm
            )
            return shared, cublas_mm(x, M.sh_gate)

        if M.moe_gscale is None:
            routed = _mtp_moe_block(core, M, x, routed_only=True)
            shared, gate_logits = shared_expert()
            return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared, gate_logits = shared_expert()
        routed = _mtp_routed_nvfp4(core, M, x)
        main.wait_stream(side)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def _hc_moe_site(self, hc, block_out, inject, L: _LayerWeights):
        """`_hc_site` at an MoE site (always with injection). Under attention TP (or at one
        rank) it also returns the routed experts' NVFP4 input (linear scales at the layer's FC1
        input global scale), written by the mixer's own last launch; under attention DP the
        window quantizes the group's gathered rows itself and `fp4` is None. Returns `(hc,
        mixed, inject, fp4)`."""
        core = self.core
        if core.dp_size > 1:
            hc, mixed, inject = self._hc_site(
                hc, block_out, inject, L.mhc_norm, L.mhc_first, L.mhc_up, True
            )
            return hc, mixed, inject, None
        n_cols = core.hc_lowrank + core.hc_count
        fast = core.hc_fast_sites
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

    def _moe(self, x, L: _LayerWeights, pad_rows: int, fp4, fold: bool, captured: bool) -> tuple:
        """One decode MoE block, left uncombined for the next site: `(routed, (gate_logits,
        shared, topk_weights, topk_rows))`. With `fold` (one rank, where the window is the whole
        layer and needs no sum) the runner skips its finalize: `routed` is its permuted FC2 rows
        and the top-k pair its routing weights and index map, which the next site's first kernel
        gathers; otherwise the pair is None and `routed` the finalized rows (`_moe_block`).
        `captured` is the step's CUDA-graph state, which picks the router (`_decode_router_mid`)."""
        core = self.core
        router_mm = _decode_router_mid(core, L, captured)
        if not fold:
            routed, gate_logits, shared = _moe_block(
                core, x, L, pad_rows, fp4, router_mm, core.decode_mm
            )
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

    def _gdn_block(self, x, i, L, si_all, verify, gen_p, captured=False) -> torch.Tensor:
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
        replace."""
        core = self.core
        pools = core._bound.pools
        num_tokens = x.shape[0]
        is_verify = verify is not None
        nv = (
            L.in_proj_nv
            if L.in_proj_nv is not None and num_tokens <= _INPROJ_NVFP4_MAX_ROWS and _gemv_ready(x)
            else None
        )
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
            )
            return self._gdn_epilogue(out, z, L, num_tokens)
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
        return core.tp_sum(_dense_mm(o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m))

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """A verify step's gated RMS norm (sigmoid gate, plain non-gemma weight) and the
        output projection, whose per-rank partial is summed over the TP group."""
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
        return core.tp_sum(_dense_mm(o, L.out, L.out8, core.decode_mm, "out", L.out_g, mid=L.out_m))

    def _ple_block(self, hc, input_ids, ple_state, next_norm, stage) -> tuple:
        """The n-gram side path on its decoder layer, added into the stream and normalized
        for the layer's attention-side site. Returns `(hc, normed)`. The same steps as the
        prefill target's `_ple_block`, over up to `rows` rows a sequence; every generation
        sequence carries its history by construction, so no carry flags are passed. A verify
        step (`stage`) stages what `SpeculativeRollback` rewrites the two pools from: both
        pools' pre-step rows before either launch publishes, and the conv input the second
        launch writes out."""
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
        )
        conv_input = torch.empty_like(hc) if stage else None
        out = ple_gate_conv_inject_norm(
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

        h = nn.functional.embedding(input_ids, core.w["embed"])
        num_tokens = h.shape[0]
        dt = h.dtype
        dev = h.device
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
        pools = core._bound.pools
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
            pos3 = _copy_positions(position_ids)
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

        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        layers, S = self._layers, self._stack
        # Each block's output and injection go into the next site's first kernel (`_hc_site`),
        # after an MoE block with its combine (and at one rank its finalize, `fold`); before
        # the PLE layer the pending block is added here, the PLE tail taking the site's norm.
        block_out = inject = moe = None
        for i, L in enumerate(layers):
            normed = None
            if i == core.ple_layer:
                if block_out is not None:
                    if moe is not None:
                        block_out = fused_sigmoid_gate_mul_add(block_out, moe[0], moe[1])
                    hc = qwen4_exp_post_inject(hc, block_out, inject)
                    block_out = inject = moe = None
                hc, normed = self._ple_block(hc, input_ids, ple_state, L.ahc_norm, is_verify)
            hc, mixed, inject = self._hc_site(
                hc, block_out, inject, L.ahc_norm, L.ahc_first, L.ahc_up, True, normed, moe
            )
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step, is_verify
                )
            else:
                block_out = self._gdn_block(mixed, i, L, si_all, verify, gen_p, md.is_cuda_graph)
            hc, mixed, inject, fp4 = self._hc_moe_site(hc, block_out, inject, L)
            block_out, moe = self._moe(
                mixed,
                L,
                pad_rows,
                fp4,
                core.moe_fold and i + 1 != core.ple_layer,
                md.is_cuda_graph,
            )
        # The stack-final site with no injection weight: its mix *is* the final norm.
        hc, final, _ = self._hc_site(
            hc, block_out, inject, S.fhc_norm, S.fhc_first, S.fhc_up, False, moe=moe
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
        x, mixed, inject = _mtp_front(core, M, embed_tokens, input_ids, hidden_states)
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
        if sparse:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            seq_lens_q = md.qsa_sequence_lengths[:rows]
            selected = torch.empty(
                [rows, core.token_topk + core.compress_ratio - 1],
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
                scores = qsa_paged_index_scores(
                    q=q_index,
                    index_cache=draft.index_cache,
                    block_table=block_table,
                    query_positions=logical,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=False,
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
                )
                if share:
                    self._capture_selection(md, topk_indices, visible_blocks, gen_seqs, p)
            attn = torch.reshape(
                _sparse_gqa(
                    core,
                    kv,
                    _packed_q(core, qkv),
                    block_table,
                    selected,
                    req_idx,
                    tokens_per_block,
                    None,
                ),
                [rows, core.q_width],
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
        fused_sigmoid_mul_inplace(attn, gate)
        return _mtp_tail(core, M, x, attn, inject, pad_rows)


# ------------------------------------------------------------ the MTP module ---


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

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        """The module's own head: its stack-final hyper-connection mixer (a gated mixer with
        its own parameters, not a plain RMS norm) over the grouped norm, then the trunk's
        `lm_head` through the shell's logits processor (vocab-parallel and gathered under
        attention TP). The norm and the mixer are the trunk's stack-final site's."""
        core = self.core
        M = core._mtp
        _, normed = qwen4_exp_hc_inject_norm(
            hidden_states, None, None, M.fhc_norm, core.eps, core.hc_count
        )
        final, _ = _hc_mixer(core, normed, M.fhc_first, M.fhc_up, False)
        draft_head = getattr(core, "draft_head", None)
        if draft_head is None:
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


class ModelingV2Qwen38FlashNextNvfp4Sm103Default(
    DecoderModelForCausalLM[Qwen4ExpDefaultModelingV2, PretrainedConfig]
):
    """Qwen3.8-Flash-Next-NVFP4 at tp1, tp4, tep4, dep4 or dep2 on sm_103."""

    # At tp1 one-engine speculative decoding takes the engine's fast paths (greedy and
    # temperature-only sampling CUDA graph variants, the overlap loop on the execution stream,
    # hoisted position ids, the MTP-Eagle draft loop's step-0 epilogue, the recorded steady
    # decode-step prepare), validated there; off at every other layout.
    spec_decode_fast_paths: bool = False
    runtime_hooks: RuntimeHooks | None = None

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Prefer KV cache manager V2: PLE state (`ple_layer_cache`) and QSA's index and
        position planes exist only there."""
        return "V2"

    @staticmethod
    def check_deployment(model_config: ModelConfig, llm_args: TorchLlmArgs, layout: str) -> None:
        """Raise `NoTarget` for a checkpoint or an engine choice this target does not run.

        `route` calls it once the GPU, the parallel layout (`layout`, a key of routing's
        `_LAYOUTS`) and the draft path match, before it returns this class. Where the
        constructor, the bind or the first forward would refuse the deployment, or a same-shape
        checkpoint would compute something else, `auto` then falls back to the built-in
        implementation.
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
        # The CuteDSL expert pipeline runs every expert at once: one rank only.
        requested = str(model_config.moe_backend_requested).upper()
        moe_backends = [
            name for name, runs in _TRUNK_MOE_BACKENDS.items() if one_rank or runs != "CUTEDSL"
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
            if self.one_rank:
                self.spec_decode_fast_paths = True
                self.runtime_hooks = RuntimeHooks(loop_on_execution_stream=True)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
                spec_decode_fast_paths=self.spec_decode_fast_paths,
            )
            assert self.spec_worker is not None, (
                f"the engine resolved no spec worker for spec_dec_mode "
                f"{self.spec_config.spec_dec_mode!r}"
            )
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
        if isinstance(self.logits_processor, _WideningLogitsProcessor):
            verify_argmax_only, self.logits_processor.draft_argmax_only = _spec_argmax_only(
                self.spec_worker, spec_metadata
            )
            logits = self.logits_processor.forward(
                nn.functional.embedding(spec_metadata.gather_ids, final),
                self.lm_head,
                attn_metadata,
                True,
                widen=not verify_argmax_only,
            )
        else:
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
        self.vision_tower = _vision.VisionTower(self.model.w, self.model.vision_config)
        self.vision_tower.warm_up()
        if self.one_rank:
            self._build_heads(self.lm_head.weight.data)

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
