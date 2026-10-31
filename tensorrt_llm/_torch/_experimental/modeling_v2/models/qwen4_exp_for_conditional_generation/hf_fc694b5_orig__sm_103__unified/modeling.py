# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 (hf-fc694b5_orig) / sm_103 / unified.

One target for the layouts `routing.py` admits in one node: at four ranks tp4 (attention
TP4, routed experts split by intermediate), tep4 (attention TP4, experts EP4) and dep4
(attention DP, experts EP4); at two ranks dep2 (attention DP, experts EP2), the layout of both
disaggregated-serving engines, which is dep4 at half the group. The core holds the static
structure (this rank's share of the weights, derived operands, per-manager pool views);
`PrefillTarget` (any batch with context rows) and `DecodeTarget` (generation only) run a step
over it, passing every argument to every op at the call site.

The forward is the single-rank target's general path (`qwen4_exp/...__tp1/modeling.py`),
cut per rank. What the layout settles is settled once, at construction:

* **Attention and Gated-DeltaNet heads** are this rank's share (`weights.rank_dims`,
  `pool.pool_geometry`): at attention TP4 six q heads, one of the two KV heads, four GDN key
  and twelve value heads. The fused attention input projection row is
  `[q | k | v | gate | index q, k | v slot]`, 4352 wide; the QSA indexer is replicated.
* **The mixer output is a partial sum** under attention TP: o_proj and out_proj read this
  rank's heads. One plain sum (`comm/allreduce`, op `NONE`) completes it before the
  hyper-connection inject, which every rank runs on the same reduced rows. Under attention DP
  a rank holds every head and serves its own requests, so there is nothing to sum.
* **Routed experts** run on the trtllm-gen NVFP4 runner, pre-routed by
  `renorm_moe_routing_op` over the whole routing space. A rank holds an EP window of
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
and then the mixer: the first projection (`cublas_mm`, fp32 out), `qwen4_exp_hc_lowrank_act`
and `qwen4_exp_hc_up_gate_mean`. The MoE site's routed input is that mixer output quantized
to NVFP4 with linear block scales at the layer's FC1 input global scale (`fp4_quantize`).
There is **no** `model.language_model.norm.weight`: the stack-final mixer is the final norm.

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

Not assembled here: the legacy per-step recurrent verify (`TRTLLM_USE_GDN_REPLAY=0`); its
refusal names what is missing.

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
from typing import NamedTuple

import torch
from flashinfer.topk import TopKTieBreak
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.fused_sigmoid_mul_inplace import (
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.fused_sigmoid_mul_inplace import (
    fused_sigmoid_mul_inplace as mtp_fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.silu_and_mul_rows import (
    silu_and_mul_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_top_k_ragged_transform import (
    flashinfer_top_k_ragged_transform,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope as mtp_fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_expand_block_indices import (
    qsa_expand_block_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_mtp_reuse_topk import (
    qsa_mtp_reuse_topk,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_index_scores import (
    qsa_paged_index_scores,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_index_store import (
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_kv_store import (
    qsa_paged_kv_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_kv_store_nvfp4 import (
    qsa_paged_kv_store_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (
    qsa_paged_sparse_gqa_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4_decode import (
    qsa_paged_sparse_gqa_nvfp4_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_prefill_compress import (
    qsa_prefill_compress,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_rollback_planes import (
    plane_table as _plane_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_rollback_planes import (
    qsa_rollback_gather,
    qsa_rollback_restore,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_expand_decode import (
    qsa_topk_expand_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_expand_decode import (
    supports as _topk_expand_supports,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.spec_rollback_commit import (
    spec_rollback_commit,
    spec_rollback_stage,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import (
    cublas_mm as mtp_cublas_mm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_gated_residual import (
    qwen4_exp_gated_residual,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_inject_norm import (
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_lowrank_act import (
    qwen4_exp_hc_lowrank_act,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_up_gate_mean import (
    qwen4_exp_hc_up_gate_mean,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_post_inject import (
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    MAX_CERTIFIED_TOKENS as _MOE_MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp8_block_scale_moe_runner import (
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_sigmoid_gate_mul_add import (
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_gemma_rmsnorm import (
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ple.ple_gate_conv_inject_norm import (
    ple_gate_conv_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ple.ple_ngram_embed import (
    ple_ngram_embed,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (
    fp8_quantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_triton_update import (
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.cu_seqlens_from_seq_lens import (
    cu_seqlens_from_seq_lens,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.flashinfer_chunk_gated_delta_rule import (
    flashinfer_chunk_gated_delta_rule,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.gdn_decode_rows_gated_norm import (
    gdn_decode_rows,
    gdn_decode_rows_gated_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.gdn_prefill_conv_post import (
    gdn_prefill_conv_post,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.quantization.mode import QuantAlgo

from . import pool as _pool
from . import vision as _vision
from . import weights as _weights
from .ops.comm.allgather__r6 import allgather
from .ops.comm.allreduce__r5 import NONE as _SUM_ONLY
from .ops.comm.allreduce__r5 import allreduce, allreduce_group_args
from .ops.comm.reducescatter__r7 import reducescatter

# The GPU architecture this target is certified for; a direct instantiation could
# bypass routing.
_SM = (10, 3)

# `moe_config.backend` request -> the routed-expert backend the trunk runs.
_TRUNK_MOE_BACKENDS = {"AUTO": "TRTLLM", "TRTLLM": "TRTLLM"}

# Rows per request of the widest verify step (`max_draft_len + 1`), and of the widest
# one without the GDN cached replay (the per-step recurrent states).
_MAX_VERIFY_WIDTH = _pool.MAX_VERIFY_WIDTH
_MAX_LEGACY_VERIFY_WIDTH = _pool.MAX_LEGACY_VERIFY_WIDTH

# Recurrent-state cache dtypes the target serves.
_SSM_STATE_DTYPES = (torch.bfloat16,)

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
_ROUTING_METHOD_RENORM = 1
_ACT_TYPE_SWIGLU = 0
_PAD_SLOT_ID = -1
_SOFTPLUS_THRESHOLD = 20.0
_L2_NORM_EPS = 1e-6
_PLE_CLAMP_MIN = 1e-6
_QSA_POSITION_AXES = _pool.QSA_POSITION_AXES
# The compressed index keys rotate at coordinate 0 alone (no mRoPE section), as the index
# query does.
_QSA_MROPE_SECTION = None
# The canonical selection: above the k-th score, then the highest-index (most recent) ties.
_TIE_BREAK_LARGE = int(TopKTieBreak.LARGE)
# The draft experts' activation scales are fp32, not the UE8M0 power-of-two form.
_MTP_MOE_USE_UE8M0 = False
# Rows per draft FP8 runner call: its 128-expert windows are certified up to one 8192-row
# chunk (`fp8_block_scale_moe_runner`). The rows an attention-DP group gathers (up to
# its size times a rank's) run in chunks of it.
_MTP_MOE_MAX_ROWS = 8192

_PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"
_PLE_HOST_OFFLOAD_OFF = frozenset({"0", "false", "no", "off"})


def _ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is declared in host memory (the default)."""
    value = os.environ.get(_PLE_HOST_OFFLOAD_ENV, "1").strip().lower()
    return value not in _PLE_HOST_OFFLOAD_OFF


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


class _Replay(NamedTuple):
    """The GDN cached replay's history buffers and bookkeeping of one manager."""

    old: dict  # GDN layer -> (old_x, old_B, old_dt, old_dA_cumsum)
    cache_buf_idx: torch.Tensor  # [slots] int32: the history half each slot writes next
    prev_num_accepted: torch.Tensor  # [slots] int32: tokens pending since the checkpoint
    history: int  # rows a history half holds
    width: int  # the one verify width a step may carry per generation request


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


class Qwen4ExpUnifiedModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` is passed in because the text sub-config adopted as
        `model_config.pretrained_config` carries neither the vision half nor the media
        token ids."""
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
        # at every layout, tp4 included: their 128x128 weight blocks do not take the trunk's
        # 160-row intermediate split (`weights.rank_dims`, `mtp_local_experts`). Under
        # attention DP each rank drafts for its own requests, and the window runs over the
        # rows the group gathers, as the trunk's does (`_mtp_moe_block`).
        # Rows per generation request of the widest verify step (`max_draft_len + 1`); 1
        # without MTP.
        self.max_verify_rows = spec_config.max_draft_len + 1 if self.mtp_enabled else 1

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
        # weight table keys it there); its rope base is its own block's, else the trunk's.
        self.mtp_layer_idx = self.num_layers
        mtp_block = getattr(cfg, "mtp", None)
        self.mtp_theta = (
            float(mtp_block.get("rope_theta", self.theta))
            if isinstance(mtp_block, dict)
            else self.theta
        )

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
        # The CuteDSL-layout copy of the experts for prefill-sized launches is not kept.
        self.moe_prefill_copy = False

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
        assert d.expert_split in _weights.LOADED_EXPERT_SPLITS, d.expert_split
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
        # The tensor-parallel sum of a block's partial output (`_tp_sum_op`), and the routed
        # experts' completion over this rank's rows (`_expert_sum_op`).
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
        projection is one GEMM. The checkpoint's parameters are rebound as row views of the
        padded tensors."""
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
        self._moe_scalars = [
            _moe_layer_scalars(w, i, self.expert_offset, self.local_experts)
            for i in range(self.num_layers)
        ]
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
        self._cos_sin = self._index_rope_table(self.theta, device)
        if self.mtp_enabled:
            # The draft layer's index plane rotates at the module's own rope base, like its
            # attention rope.
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
        self._fuse_hc_first_projections()
        # The CPU-only load checks build the targets over host weights; they launch nothing.
        if device.type == "cuda":
            self._report_ple_residency()
            self._side_stream = torch.cuda.Stream(device=device)
            self.tp_sum = self._tp_sum_op()
            self.expert_sum = self._expert_sum_op()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _tp_sum_op(self):
        """The sum that completes a block's per-rank partial output, settled once: under
        attention TP the plain all-reduce over the TP group (`comm/allreduce`, op `NONE`),
        which every rank gets the same bits back from; otherwise nothing to sum."""
        if self.attn_tp_size == 1:
            return lambda x: x
        group = allreduce_group_args(self.mapping)

        def tp_sum(x: torch.Tensor) -> torch.Tensor:
            return allreduce(
                x,
                None,
                None,
                scale=None,
                bias=None,
                op=_SUM_ONLY,
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
        assert self.quant_mode != QM_NVFP4_KV_CACHE or md.use_paged_context_fmha, (
            "an NVFP4 KV cache is read by the paged-context attention kernels only, which the "
            "engine enables for chunked prefill, KV cache block reuse or speculative decoding"
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
            f"modeling_v2 unified qwen4_exp: rank {self.mapping.rank} binds "
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
            f"modeling_v2 unified qwen4_exp: rank {self.mapping.rank} binds the draft layer to "
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


def _layer_views(core: Qwen4ExpUnifiedModelingV2) -> list[_LayerWeights]:
    """Every trunk layer's `_LayerWeights`, read off the realized weights."""
    w = core.w
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
                # The expert stacks are uint8 storage; the runner reads the packed nibbles as
                # uint8 and the block scales as e4m3.
                fc1_w=w[f"l{i}_fc1_w"],
                fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                fc2_w=w[f"l{i}_fc2_w"],
                fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                out1_scale=scalars["output1_scale_scalar"],
                out1_gate_scale=scalars["output1_scale_gate_scalar"],
                out2_scale=scalars["output2_scale_scalar"],
                sh_gug=w[f"l{i}_sh_gu"].t(),
                sh_dn=w[f"l{i}_sh_dn"].t(),
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


def _stack_views(core: Qwen4ExpUnifiedModelingV2) -> _StackWeights:
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
    `num_hidden_layers`), as the views the ops read: GEMM operands are `.t()` views and
    grouped-norm weights their per-stream slices."""

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
    fhc_norm: tuple
    fhc_down: torch.Tensor
    fhc_up: torch.Tensor


def _mtp_views(core: Qwen4ExpUnifiedModelingV2) -> _MtpWeights:
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"

    def slices(key: str) -> tuple:
        return tuple(torch.split(w[key], core.hidden, dim=0))

    return _MtpWeights(
        pre_e_norm=w[f"{p}pre_e_norm"],
        pre_h_norm=slices(f"{p}pre_h_norm"),
        fc_e=w[f"{p}fc_e"].t(),
        fc_h=w[f"{p}fc_h"].t(),
        ahc_norm=slices(f"{p}ahc_norm"),
        ahc_down=w[f"{p}ahc_down"],
        ahc_up=w[f"{p}ahc_up"],
        ahc_inject=w[f"{p}ahc_inject"],
        mhc_norm=slices(f"{p}mhc_norm"),
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
        sh_gug=w[f"{p}sh_gu"].t(),
        sh_dn=w[f"{p}sh_dn"].t(),
        fhc_norm=slices(f"{p}fhc_norm"),
        fhc_down=w[f"{p}fhc_down"],
        fhc_up=w[f"{p}fhc_up"],
    )


# ------------------------------------------------------------ shared blocks ---


def _grouped_norm(core, x: torch.Tensor, weights: tuple) -> torch.Tensor:
    """`Qwen4ExpTextRMSNorm(group_size=hidden)` of the `[T, stream]` residual `x`: an
    independent Gemma RMS norm per `hidden`-wide stream (a column slice at the stream's row
    stride), concatenated back."""
    parts = torch.split(x, [core.hidden] * core.hc_count, 1)
    return torch.cat(
        [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(weights, parts)], 1
    )


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
    manager runs one after the step -- the stock Gated-DeltaNet layer's two conditions."""
    assert core._bound.int_idx is not None, "the cache-manager bind has not run"
    int_idx = core._bound.int_idx[:num_decodes]
    partitioned = num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    all_layer = partitioned and bool(
        getattr(core._bound.cm, "use_gdn_cached_replay_all_layer_commit", False)
    )
    work_items = mm.replay_work_items[:num_decodes] if partitioned else None
    n_writes = mm.replay_n_writes if partitioned else None
    return int_idx, work_items, n_writes, all_layer


def _gdn_verify(core, mq, a, b, i, L, si_d, n, p, verify) -> torch.Tensor:
    """The recurrent half of a verify step: `n` generation sequences of `p` rows, written so
    that the accepted prefix, unknown until the sampler has run, can be kept.

    The conv writes each row's window to the runtime's per-step buffer
    (`update_mamba_states` promotes the accepted one) besides advancing the pool; the cached
    replay rebuilds each slot's state from its checkpoint and pending history, runs the `p`
    rows and appends them to the history, so rejected rows cost nothing to undo. `mq` is the
    `[n * p, conv_dim]` q|k|v column view of the input projection, `a` and `b` its raw gate
    columns (contiguous): the kernel applies `A_log`, `dt_bias` and the q/k L2 norm itself.
    Returns `[1, n * p, num_v_heads, head_v_dim]`."""
    int_idx, work_items, n_writes, all_layer = verify
    pools = core._bound.pools
    replay = core._bound.replay
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
        launch_dependent_kernels=False,
    )
    # Back to token-major: one contiguous copy, which the kernel reads q/k/v from directly.
    mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
    old_x, old_B, old_dt, old_dA = replay.old[i]
    qv, kv, vv = torch.split(mq_out, [core.key_dim, core.key_dim, core.value_dim], 1)
    out = fused_recurrent_gated_delta_rule_cached_replay_update(
        q=torch.reshape(qv, [n, p, core.num_k_heads, core.head_k_dim]),
        k=torch.reshape(kv, [n, p, core.num_k_heads, core.head_k_dim]),
        v=torch.reshape(vv, [n, p, core.num_v_heads, core.head_v_dim]),
        g=torch.reshape(a, [n, p, core.num_v_heads]),
        beta=torch.reshape(b, [n, p, core.num_v_heads]),
        ssm_states=pools.ssm[i],
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
        use_all_layer_commit=all_layer,
        output=None,
    )
    return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])


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


def _mtp_front(core, M: _MtpWeights, embed_tokens, input_ids, hidden_states) -> tuple:
    """The draft layer's front end and its attention-side hyper-connection. `fc_embedding`
    over the next token's normalized embedding is added to every stream of `fc_hidden`
    (one `[hidden, hidden]` matrix applied to each normalized stream of the trunk's
    residual); then the gated mixer. Returns `(x, mixed, inject)`: the layer's residual,
    the mixed attention input and its injection weights."""
    rows = hidden_states.shape[0]
    e = nn.functional.embedding(input_ids, embed_tokens)
    x_e = mtp_cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
    h_n = _grouped_norm(core, hidden_states, M.pre_h_norm)
    x_h = torch.reshape(
        mtp_cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
        [rows, core.stream],
    )
    x = torch.add(x_h, torch.cat([x_e] * core.hc_count, -1))
    mixed, inject = qwen4_exp_gated_residual(
        _grouped_norm(core, x, M.ahc_norm), M.ahc_down, M.ahc_up, M.ahc_inject, core.hc_count
    )
    return x, mixed, inject


def _mtp_qkv(core, M: _MtpWeights, mixed, pos3) -> tuple:
    """The draft layer's q (with its interleaved output gate), k and v projections at this
    rank's heads, packed into one compact `[T, qkv_width]` row and normalized and rotated in
    place at the module's rope base. Returns `(qkv, gate, k, v)`: `gate` is the
    `[T, heads_q, head_dim]` gate half of the q projection, `k` and v` views of `qkv`."""
    rows = mixed.shape[0]
    qg = mtp_cublas_mm(mixed, M.q)
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
    k_dst.copy_(torch.reshape(mtp_cublas_mm(mixed, M.k), [rows, core.heads_kv, core.head_dim]))
    v_dst.copy_(torch.reshape(mtp_cublas_mm(mixed, M.v), [rows, core.heads_kv, core.head_dim]))
    mtp_fused_qk_norm_rope(
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
    qk_index = mtp_cublas_mm(mixed, M.idx)
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


def _mtp_moe_block(core, M: _MtpWeights, x, pad_rows: int = 0) -> torch.Tensor:
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

    if core.dp_size == 1:
        routed = core.tp_sum(run(x))
    else:
        routed = core.expert_sum(run, x, pad_rows)
    gug = cublas_mm(x, M.sh_gug)
    shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), M.sh_dn)
    gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
    return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)


def _mtp_tail(core, M: _MtpWeights, x, attn, inject, pad_rows: int) -> torch.Tensor:
    """The draft layer after its attention core: the gated output projection's partial summed
    over the TP group and injected, the MoE-side hyper-connection, the MoE and its injection.
    `attn` is the `[T, q_width]` gated attention output; `pad_rows` is the MoE round trip's
    padding under attention DP (`_draft_pad_rows`)."""
    x = qwen4_exp_post_inject(x, core.tp_sum(cublas_mm(attn, M.o)), inject)
    mixed, inject = qwen4_exp_gated_residual(
        _grouped_norm(core, x, M.mhc_norm), M.mhc_down, M.mhc_up, M.mhc_inject, core.hc_count
    )
    return qwen4_exp_post_inject(x, _mtp_moe_block(core, M, mixed, pad_rows), inject)


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


def _hc_mixer(core, normed, first, up, with_inject):
    """One hyper-connection mixer from the site's normed stream: the first projection over
    `[down; block_inject; 0]` (fp32 logits), the SiLU and injection activations, then the up
    projection with the sigmoid gate and the stream mean in its epilogue. Returns
    `(mixed, inject)`; `inject` is None without `with_inject` (the stack-final mixer)."""
    low, inject = qwen4_exp_hc_lowrank_act(
        cublas_mm(normed, first[:, : core.hc_first_cols], None, torch.float32),
        core.hc_lowrank,
        core.hc_count,
        with_inject,
        normed.dtype,
    )
    return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count), inject


def _routed_experts(core, x, L: _LayerWeights) -> torch.Tensor:
    """This rank's partial of the routed experts over every row of `x`: Renormalize routing
    over the whole routing space (`renorm_moe_routing_op`), the NVFP4 activation with linear
    block scales at the layer's FC1 input global scale, and the trtllm-gen runner over this
    rank's expert window, pre-routed, finalized. A slot routed outside the window adds
    nothing, so the window outputs sum to the layer's. Under the intermediate split (tp4)
    the window is every expert at this rank's `expert_inter`-wide slice, and the slices'
    outputs sum to the layer's the same way.

    Every step is row-wise, so `x` runs in chunks of at most the runner's certified rows (a
    host decision on `T`): one chunk at a TP rank's rows, several over the rows an
    attention-DP group gathers."""
    rows = x.shape[0]
    parts = []
    for lo in range(0, rows, _MOE_MAX_ROWS):
        chunk = x[lo : min(rows, lo + _MOE_MAX_ROWS)]
        # The routing reads its logits as a dense row-major buffer; cublas_mm allocates them
        # contiguous.
        logits = cublas_mm(chunk, L.router)
        topk_ids, topk_w = renorm_moe_routing_op(
            logits, topk=core.topk, output_dtype=torch.bfloat16
        )
        xq, xsf = fp4_quantize(chunk, L.moe_gscale, _SF_VEC, False, False)
        parts.append(
            fp4_block_scale_moe_runner(
                routing_logits=None,
                routing_bias=None,
                hidden_states=xq,
                # The runner reads the chunk's linear block scales flat.
                hidden_states_scale=torch.reshape(xsf, [-1]).view(torch.float8_e4m3fn),
                gemm1_weights=L.fc1_w,
                gemm1_weights_scale=L.fc1_s,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=L.fc2_w,
                gemm2_weights_scale=L.fc2_s,
                gemm2_bias=None,
                output1_scale_scalar=L.out1_scale,
                output1_scale_gate_scalar=L.out1_gate_scale,
                output2_scale_scalar=L.out2_scale,
                num_experts=core.num_experts,
                top_k=core.topk,
                n_group=None,
                topk_group=None,
                intermediate_size=core.expert_inter,
                local_expert_offset=core.expert_offset,
                local_num_experts=core.local_experts,
                routed_scaling_factor=None,
                routing_method_type=_ROUTING_METHOD_RENORM,
                do_finalize=True,
                act_type=_ACT_TYPE_SWIGLU,
                topk_weights=topk_w,
                topk_ids=topk_ids,
            )[0]
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _moe_block(core, x, L: _LayerWeights, pad_rows: int) -> tuple:
    """One MoE layer: NVFP4 routed experts plus the replicated bf16 shared expert, left
    uncombined. Returns `(routed, gate_logits, shared)`: the next `qwen4_exp_hc_inject_norm`
    forms `routed + sigmoid(gate_logits) * shared` inside its kernel. `routed` is every
    window's contribution to this rank's rows (`core.expert_sum`): under attention TP the
    window partials' sum over the TP group, the same bits on every rank; under attention DP
    the round trip over the group's gathered rows, `pad_rows` padding this rank's.

    The shared expert and its per-token gate logit run on the core's side stream,
    concurrently with the routed path, and join it before this returns. `x` is the bf16
    MoE-site mixer output."""
    main = torch.cuda.current_stream()
    side = core._side_stream
    # Fork: the side stream reads `x` only after everything queued on `main` so far. What it
    # allocates is consumed on `main` after the join below.
    side.wait_stream(main)
    with torch.cuda.stream(side):
        # One GEMM forms gate|up and the gate logit (`sh_gu`'s extra rows).
        gug = cublas_mm(x, L.sh_gug)
        shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
        gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
    routed = core.expert_sum(lambda rows: _routed_experts(core, rows, L), x, pad_rows)
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


def _select(core, scores, row_starts, row_ends, topk_indices) -> None:
    """Each row's canonical top-`block_topk` groups of its window into `topk_indices`: the
    groups strictly above the budget's last score, then the highest-index ties, ascending,
    `-1` last -- a function of the scores alone, which the expansion and the sparse GQA
    reduce in. QSA's row starts are zero, so window-local and table indices agree."""
    selected = flashinfer_top_k_ragged_transform(
        scores,
        torch.zeros_like(row_starts),
        row_ends - row_starts,
        core.block_topk,
        False,
        _TIE_BREAK_LARGE,
        False,
        row_starts,
    )
    topk_indices.copy_(selected)


def _selection(core, scores, visible_blocks, logical, seq_lens_q, topk_indices, row_starts, out):
    """The selected token indices of `scores`' rows into `out` `[rows, token_topk +
    compress_ratio - 1]`: the canonical block top-k and its expansion in one launch, or, for
    a score table that kernel does not serve, the selection and the expansion apart."""
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


def _sparse_gqa(core, kv, q, block_table, selected, request_indices, tpb, query_positions):
    """Sparse GQA over the layer's paged pool. `query_positions` is a context row's bound on
    its scan (the prefill pair with `compress_ratio`), or None for decode rows. Over an NVFP4
    pool context rows take the prefill kernel and decode rows the split-K decode kernel."""
    if kv.k_scale is None:
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
    """A fresh contiguous int32 `[3, N]` copy of the step's mRoPE positions (the engine hands
    a slice of a larger buffer, whose row stride the rope rejects)."""
    pos3 = torch.empty(
        [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
        dtype=torch.int32,
        device=position_ids.device,
    )
    pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))
    return pos3


def _step_args(md: TrtllmAttentionMetadata, num_contexts, num_ctx_tokens) -> dict:
    """thop_attention's batch state off the prepared metadata. Tensors are engine-owned
    persistent buffers refreshed in place; Python ints are per-capture constants."""
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


class PrefillTarget:
    """Any batch with context rows (`num_contexts > 0`), possibly with generation rows
    beside them."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2) -> None:
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks.
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)
        # The step `prepare` built, read by the breaks.
        self._step: _PrefillStep | None = None

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
        return _step_args(md, md.num_contexts, md.num_ctx_tokens)

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
        proj = cublas_mm(x, L.attn_in)
        gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
        _attention_rope(core, proj, index_scratch, L, pos3, pos)
        eager_on_graph(proj, proj_qkv, index_scratch, attn_out, i, inner=self._attention_core)
        fused_sigmoid_mul_inplace(attn_out, gate)
        return core.tp_sum(cublas_mm(attn_out, L.o))

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
        # The packed `[q | k | v]` as one compact copy: the dense op reads it flat, and the
        # sparse GQA reads its query heads from it.
        qkv = proj_qkv[:num_tokens]
        qkv.copy_(proj[:, : core.qkv_width])
        if step.stage:
            core.spec_rollback.stage_index(
                core._bound.pools.index_cache[i], core._bound.pools.position_cache
            )
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
            for lo, hi, context_rows in parts:
                scores = qsa_paged_index_scores(
                    q=q_index[lo:hi],
                    index_cache=core._bound.pools.index_cache[i],
                    block_table=block_table,
                    query_positions=logical[lo:hi],
                    request_indices=req_idx[lo:hi],
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=context_rows,
                )
                _selection(
                    core,
                    scores,
                    visible_blocks[lo:hi],
                    logical[lo:hi],
                    seq_lens_q[lo:hi],
                    topk_indices[lo:hi],
                    topk_row_starts[lo:hi],
                    selected[lo:hi],
                )
            # Context rows with the prefill pair (a bound on a row's scan), generation rows
            # without.
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
        proj = cublas_mm(x, L.in_proj)
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
                    a_d.contiguous(),
                    b_d.contiguous(),
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
        return core.tp_sum(cublas_mm(o, L.out))

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
        _, normed = qwen4_exp_hc_inject_norm(hc, None, None, layers[0].ahc_norm, core.eps, hc_count)
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                eager_on_graph(hc, ple_hc, ple_normed, inner=self._ple_block)
                hc, normed = ple_hc, ple_normed
            mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, proj_qkv, index_scratch, attn_out, pos3, pos
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_out)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, block_out, inject, L.mhc_norm, core.eps, hc_count
            )
            mixed, inject = _hc_mixer(core, normed, L.mhc_first, L.mhc_up, True)
            next_norm = layers[i + 1].ahc_norm if i + 1 < len(layers) else S.fhc_norm
            # The MoE block output is formed inside the injection: routed + sigmoid(gate) *
            # shared, rounded once.
            routed, gate_logits, shared = _moe_block(core, mixed, L, pad_rows)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, routed, inject, next_norm, core.eps, hc_count, shared, gate_logits
            )
            del routed, gate_logits, shared
        # The stack-final mixer with no injection weight is the final norm.
        final, _ = _hc_mixer(core, normed, S.fhc_first, S.fhc_up, False)
        if not core.mtp_enabled:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the `[T, stream]` stream
        # it was collapsed from. They leave packed in one tensor, because a breakable prefill
        # graph's body returns one tensor (`split_spec_output`).
        return torch.cat([final, hc], -1)

    # ------------------------------------------------------------- the draft ---

    def _capture_selection(self, md, topk_indices, visible_blocks) -> None:
        """IndexShare: stash, per request, the selection of the row the draft loop continues
        from, with that row's complete-group count, in the draft layer's row of the shared
        buffers: a context request's last row of this chunk, and a generation request's last
        accepted row (`num_accepted - 1` past its first)."""
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
        (draft-extend) step of a batch with context rows, eager. Every row runs the whole
        draft layer; the worker gathers one row per request from the result.

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
        if sparse:
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
            for lo, hi, context_rows in parts:
                scores = qsa_paged_index_scores(
                    q=q_index[lo:hi],
                    index_cache=draft.index_cache,
                    block_table=block_table,
                    query_positions=logical[lo:hi],
                    request_indices=req_idx[lo:hi],
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=context_rows,
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
        mtp_fused_sigmoid_mul_inplace(attn, gate)
        return _mtp_tail(core, M, x, attn, inject, pad_rows)


# ------------------------------------------------------------------- decode ---


class DecodeTarget:
    """Generation-only batches (no context rows): one token per sequence, every op inside
    the captured decode graph."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2) -> None:
        self.core = core
        # See PrefillTarget.__init__.
        self._contract_pending = step_contract_enabled()
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
        return _step_args(md, num_contexts=0, num_ctx_tokens=0)

    # --------------------------------------------------------------- blocks ---

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage):
        """One gated GQA layer plus its QSA index branch for a generation-only batch, at this
        rank's heads.

        The four input projections are one GEMM against the fused operand; the norm + rope of
        both planes runs on it; one compact copy of its leading `[q | k | v]` feeds the dense
        op and the sparse GQA's query heads, the sparse K/V append reads k and v as views of
        it, and the index branch reads the raw key from it. The index plane is maintained on
        every step, at both ends of the threshold; above it the dense core is replaced and the
        main K/V append becomes this layer's own call. A verify step's rows (`stage`) come
        `predicted_tokens_per_seq` a request, each scored and attended on its own, and their
        index slots are staged for `SpeculativeRollback` before the store."""
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
        proj = cublas_mm(x, L.attn_in)
        qkv_view, gate, _ = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        _, k_dst, v_dst = _attn_in_heads(core, proj)
        _attention_rope(core, proj, index_scratch, L, pos3, pos)
        if stage:
            core.spec_rollback.stage_index(
                core._bound.pools.index_cache[i], core._bound.pools.position_cache
            )
        _index_plane(core, proj, i, L, (req_idx, logical, block_table, tokens_per_block, coords))
        # The packed `[q | k | v]` as one compact copy: the dense op reads it flat, and the
        # sparse GQA reads its query heads from it.
        qkv.copy_(qkv_view)

        if sparse:
            kv = core._bound.pools.kv[i]
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
        return core.tp_sum(cublas_mm(out, L.o))

    def _gdn_block(self, x, i, L, si_all, verify, gen_p) -> torch.Tensor:
        """One Gated-DeltaNet layer over a generation-only batch, one token per sequence, at
        this rank's heads: one GEMM for all four input projections (q|k|v, z, a and b are
        column views of its output), the conv step over the q|k|v columns in place, then the
        recurrence and the gated norm as one streaming launch that computes each (row,
        v-head) on its own, so over a bf16 pool a row's output and state do not depend on its
        batch. The output projection's per-rank partial is summed over the TP group.

        A verify step (`verify`, `gen_p` rows a sequence) takes `_gdn_verify` and the gated
        norm epilogue instead."""
        core = self.core
        pools = core._bound.pools
        proj = cublas_mm(x, L.in_proj)
        if verify is not None:
            mixed_qkv, z, a, b = torch.split(proj, core.in_proj_split, 1)
            num_tokens = x.shape[0]
            out = _gdn_verify(
                core,
                mixed_qkv,
                a.contiguous(),
                b.contiguous(),
                i,
                L,
                si_all,
                num_tokens // gen_p,
                gen_p,
                verify,
            )
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
        return core.tp_sum(cublas_mm(o, L.out))

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
        return core.tp_sum(cublas_mm(o, L.out))

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
        pos3 = _copy_positions(position_ids)
        pos = pos3[0]
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
        # Under speculation `query_start_loc_long` has one row per generation sequence: with
        # `gen_p > 1` the PLE segmentation is rebuilt from `seq_lens_cuda`.
        ple_qsl = (
            mm.query_start_loc_long
            if gen_p == 1
            else cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        )
        ple_state = (pools.ple_state[0], pools.ple_state[1], si_all, ple_qsl, gen_p)

        # Staging for the speculative rollback (see PrefillTarget.prepare); every row here is
        # a generation row. A captured step stages and commits the PLE pools in one launch
        # each.
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
        _, normed = qwen4_exp_hc_inject_norm(hc, None, None, layers[0].ahc_norm, core.eps, hc_count)
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                hc, normed = self._ple_block(hc, input_ids, ple_state, L.ahc_norm, is_verify)
            mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step, is_verify
                )
            else:
                block_out = self._gdn_block(mixed, i, L, si_all, verify, gen_p)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, block_out, inject, L.mhc_norm, core.eps, hc_count
            )
            mixed, inject = _hc_mixer(core, normed, L.mhc_first, L.mhc_up, True)
            next_norm = layers[i + 1].ahc_norm if i + 1 < len(layers) else S.fhc_norm
            routed, gate_logits, shared = _moe_block(core, mixed, L, pad_rows)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, routed, inject, next_norm, core.eps, hc_count, shared, gate_logits
            )
            del routed, gate_logits, shared
        # The stack-final mixer with no injection weight: its mix *is* the final norm.
        final, _ = _hc_mixer(core, normed, S.fhc_first, S.fhc_up, False)
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
        pos3, pos, coords = _draft_positions(position_ids)
        M = core._mtp
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block

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
        mtp_fused_sigmoid_mul_inplace(attn, gate)
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

    def __init__(self, core: Qwen4ExpUnifiedModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        """The module's own head: its stack-final hyper-connection mixer (a gated mixer with
        its own parameters, not a plain RMS norm) over the grouped norm, then the trunk's
        `lm_head` through the shell's logits processor (vocab-parallel and gathered under
        attention TP)."""
        core = self.core
        M = core._mtp
        final, _ = qwen4_exp_gated_residual(
            _grouped_norm(core, hidden_states, M.fhc_norm),
            M.fhc_down,
            M.fhc_up,
            None,
            core.hc_count,
        )
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

    def __init__(self, core: Qwen4ExpUnifiedModelingV2, lm_head, logits_processor) -> None:
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

    def stage_index(self, index_cache: torch.Tensor, position_cache: torch.Tensor) -> None:
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


class ModelingV2HfFc694b5OrigSm103Unified(
    DecoderModelForCausalLM[Qwen4ExpUnifiedModelingV2, PretrainedConfig]
):
    """Qwen3.8-Flash-Next-NVFP4 at tp4, tep4, dep4 or dep2 on sm_103."""

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Prefer KV cache manager V2: PLE state (`ple_layer_cache`) and QSA's index and
        position planes exist only there."""
        return "V2"

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
            Qwen4ExpUnifiedModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # `lm_head` is outside the quantization scheme: a dense bf16 `LMHead`.
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"expected a bf16 lm_head; the shell built {self.lm_head.weight.dtype}"
        )
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
