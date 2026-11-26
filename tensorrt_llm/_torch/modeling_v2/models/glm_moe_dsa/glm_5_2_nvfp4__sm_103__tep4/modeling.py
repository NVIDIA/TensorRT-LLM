# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: glm-5.2-nvfp4 / sm_103 / tep4 -- self-contained modeling code.

GLM-5.2, modelopt NVFP4 export: 78 layers, hidden 6144, 64 query heads (16 per rank), layers
0-2 dense and 3-77 MoE (256 routed experts at top-8 plus a shared expert). Attention is MLA
with DeepSeek Sparse Attention: an indexer on 21 of the layers selects 2048 KV positions per
query row, and the layers in between reuse the most recent selection. Only the routed experts
are NVFP4; everything else is bf16 and the latent KV pool is fp8-e4m3. Layer 78 is the bf16
MTP module (`MTPLayer`), live only under a `speculative_config`.

`tep4` shards the attention heads, the dense and shared MLPs and `lm_head` four ways and the
routed experts four ways, so every layer ends in an all-reduce (`ops/comm/`).

Most kernels are the checkpoint-specialised CuTe DSL kernels in `kernels/`, reached through
the `cute_dsl_*` entries in `ops/` (projections, norms, indexer, absorbed BMMs, sparse MLA,
the all-reduce, the routed experts' router, top-k, routing, quantize and grouped GEMMs), and
around them the logits head (the `lm_head` shard GEMM and a Lamport all-gather that widens to
fp32), the spec worker's greedy picks (`spec_worker.py`), and the row ops the fused kernels leave over (hidden-
width norms and adds, row lookups, copies). A call site whose row count or layout has no
compiled kernel falls back to the stock op. Both attention
phases run the specialised sparse-MLA kernels: the context phase on the sm103 prefill kernel,
the decode phase on the decode kernel.
The context step's projections, absorbed BMMs and rope/KV append run on the prefill
counterparts of the decode kernels; on a step where the engine skipped the indexer for the
context rows, the context attention reads the causal window through the block table.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._torch.speculative import (
    MTPEagleWorker,
    get_spec_worker,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from . import weights as _weights
from .decode import DecodeTarget
from .forward_ops import (
    _DECODE_MAX_ROWS,
    _MLA_ROPE_MTP_WIDTH,
    _MOE_DECODE_WIDTHS,
    _MOE_MAX_T,
    _MOE_SF_VEC,
    _MOE_TOP_K,
    DRAFT,
    PREFILL,
    Attn,
    Comm,
    Ctx,
    Gemm,
    Indexer,
    Mlp,
    Moe,
    Mtp,
    Rows,
)
from .kernels.common import cuda_stream as _cuda_stream
from .kernels.moe import moe_fc1_kblock as _fc1_kblock
from .ops.activation import cute_dsl_swiglu_mxf8 as _swiglu8_op
from .ops.activation import cute_dsl_swiglu_rows as _swiglu_rows_op
from .ops.activation.cute_dsl_dense_swiglu import compile as compile_dense_swiglu
from .ops.attention import cute_dsl_mha_kv_gather_mxf8 as _kv_gather8_op
from .ops.attention import cute_dsl_mha_prefill as _mha_pre_op
from .ops.attention import cute_dsl_mha_prefix_gather as _mha_prefix_op
from .ops.attention import cute_dsl_mla_rope_append_prefill as _ctx_rope_op
from .ops.attention import cute_dsl_mla_rope_generation as _mla_rope_op
from .ops.attention import cute_dsl_sparse_mla_decode as _fmha_dec_op
from .ops.attention import cute_dsl_sparse_mla_decode_union as _fmha_union_op
from .ops.attention import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .ops.attention.cute_dsl_indexer_k_cache_update import build as build_ix_k_cache
from .ops.attention.cute_dsl_indexer_q_side_fused import build as build_ix_q_side
from .ops.attention.cute_dsl_indexer_q_side_k_cache import build as build_ix_qk
from .ops.attention.cute_dsl_indexer_topk_decode import build as build_ix_topk_decode
from .ops.attention.cute_dsl_indexer_topk_prefill import build as build_ix_topk_prefill
from .ops.attention.cute_dsl_req_index_to_base import build as build_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_global import build as build_req_index_to_global
from .ops.comm import cute_dsl_allgather_widen as _allgather_op
from .ops.comm._lamport import build_workspace as build_lamport_workspace
from .ops.comm.cute_dsl_allgather_widen import cute_dsl_allgather_widen
from .ops.elementwise import cute_dsl_copy as _copy_op
from .ops.elementwise import cute_dsl_hidden_add as _hidden_add_op
from .ops.elementwise import cute_dsl_rows_select as _rows_sel_op
from .ops.embedding import cute_dsl_hidden_row_gather as _row_gather_op
from .ops.embedding.cute_dsl_embedding import build as build_embedding
from .ops.embedding.cute_dsl_hidden_row_gather import cute_dsl_hidden_row_gather
from .ops.gemm import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .ops.gemm import cute_dsl_dense_up_swiglu_mxf8 as _dense_up8_op
from .ops.gemm import cute_dsl_indexer_pre_proj_split as _ix_pre_split_op
from .ops.gemm import cute_dsl_mtp_eh_proj as _mtp_eh
from .ops.gemm import cute_dsl_projection as _projection_op
from .ops.gemm import cute_dsl_projection_mxf8 as _mxf8_op
from .ops.gemm import cute_dsl_projection_mxf8_decode as _dec8_op
from .ops.gemm import cute_dsl_q_b_ix_wq_b as _qbix_op
from .ops.gemm import cute_dsl_qkva_ix_fold as _ix_fold_op
from .ops.gemm import cute_dsl_shared_gate_up_swiglu as _shared_gu_swiglu_op
from .ops.gemm import cute_dsl_shared_up_swiglu_mxf8 as _shared_up8_op
from .ops.gemm.cute_dsl_projection import cute_dsl_projection
from .ops.moe import cute_dsl_moe_l2_prefetch as _l2pf_op
from .ops.moe import cute_dsl_moe_merge as _moe_merge_op
from .ops.moe import cute_dsl_moe_nvfp4_experts as _moe_experts_op
from .ops.moe import cute_dsl_moe_prefill_topk_routing_rep as _moe_topk_pre_rep_op
from .ops.moe import cute_dsl_moe_router as _moe_router_op
from .ops.moe import cute_dsl_moe_routing as _moe_routing_op
from .ops.moe import cute_dsl_moe_sigmoid_topk as _topk_rows_op
from .ops.moe import cute_dsl_moe_topk_routing as _moe_topk_op
from .ops.moe import cute_dsl_moe_topk_routing_rep as _moe_topk_rep_op
from .ops.moe import cute_dsl_moe_topk_routing_wide_rep as _moe_topk_wide_rep_op
from .ops.moe import cute_dsl_moe_zero as _moe_zero_op
from .ops.moe import cute_dsl_mtp_moe_experts as _mtp_experts_op
from .ops.norm import cute_dsl_hidden_add_rmsnorm as _hidden_add_norm_op
from .ops.norm import cute_dsl_hidden_rmsnorm as _hidden_norm_op
from .ops.norm import cute_dsl_mtp_eh_norm_cat_mxf8 as _eh8_op
from .ops.norm import cute_dsl_qkv_a_rmsnorm_q8 as _qkv_a_norm8_op
from .ops.norm.cute_dsl_mtp_norm_shard import compile as compile_mtp_norm_shard
from .ops.norm.cute_dsl_qkv_a_rmsnorm import compile as compile_qkv_a_rmsnorm
from .ops.quantization import cute_dsl_act_quant_mxf8_rows as _aq_rows_op
from .ops.quantization import cute_dsl_moe_quantize_nvfp4 as _moe_quant_op
from .ops.sampling import cute_dsl_argmax as _argmax_op
from .prefill import PrefillTarget
from .spec_worker import Glm52MTPEagleWorker
from .wide_decode import WideDecodeTarget

_QUANT_MODE_FP8_KV = 128
# How often rank 0 logs the route counters, in decode forwards (eager forwards and graph
# captures: a replay does not re-enter Python). Context steps always report.
_ROUTE_REPORT_EVERY = 16

# The fp8 context attention's dequantization scales are powers of two from strict bounds on
# its operands (`_mha_fp8_scales`); the bound's margin covers bf16 rounding of the normalized
# inputs and the e4m3 cached-prefix rows (each within 1/16 of its bf16 value).
_MHA_FP8_MARGIN = 1.1
# Extra headroom of the Q bound when q_b runs in MXF8: both of its operands are e4m3, each
# element within 1/16 of its bf16 value, so the Cauchy-Schwarz bound grows by (17/16)^2.
_MHA_FP8_Q_MXF8 = (17.0 / 16.0) ** 2
# The same headroom on the K / V bounds when the expansion GEMMs run in MXF8: the margin
# covered e4m3 latent rows (1/16 per element); the e4m3 weight adds that factor once more.
_MHA_FP8_KV_MXF8 = 17.0 / 16.0


# The sm103 prefill kernel's two scale arrays. The context path has no op that writes them,
# so they are the values the decode path's `mla_rope_generation` was measured to write:
# `[1] == [0] * log2(e)`, a natural-log softmax scale of 1/16. Both are functions of
# q_scaling and the MLA dims only, so they are constants for this checkpoint. The query is
# cast to e4m3 unscaled: `[0]` is the kernel's internal dequant factor, not the quantizer's.
_FMHA_BMM1_SCALE = (0.0625, 0.09016843885183334)
_FMHA_BMM2_SCALE = (1.0,)
# The indexer's LayerNorm epsilon, which is not the model's rms_norm_eps.
_INDEXER_K_NORM_EPS = 1e-6
# `head_dim + 4`: the fused write needs the fp8 key and its fp32 scale in one slot.
_INDEXER_PER_TOKEN_MIN = 132
# The context indexer projection's fp32 output row: 128 key columns, then 32 head weights.
_IX_PROJ_COLS = 160
# The fused shared-expert gate|up + SwiGLU kernel is compiled for one row count per runner
# (`_shared_gu_swiglu_op`): an MTP3 / MTP5 engine also builds it at its decode row counts of
# 1..`_SHARED_FUSED_UP_SEQS` sequences without a cell (MTP5 6 / 12 / 18, MTP3 12), served
# where `Gemm.cell_up` holds. Wider counts take the two-launch path on the projection cells.
_SHARED_FUSED_UP_SEQS = 3
# The decode logits width (the engine's max_seq_len) at or below which the generation rows keep our
# decode top-k although the engine's config asks for GVR (`_dec_topk_gvr`;
# `TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`). Single-B300 CUDA graph of 20 calls (us per call, ours /
# GVR, every set exact): GVR's tuning is frozen from the width, so on a 16384-wide engine at 192
# rows (MTP5 x 32) it costs 5.2 / 12.5 at KV 2k / 4k against our 1.5 / 5.5, where a realtext
# step (ISL 1k + OSL 8k) spends most of its decode; at 172032 GVR takes 8-17 (42 at 192 rows)
# against our 58-95 at KV 160k. Only the width and the row count are fixed at capture.
_DEC_TOPK_GVR_ABOVE = 16384


# MTP draft-loop index share: the engine's DSA indexer semantics (`Indexer.sparse_attn_indexer`
# with `mtp_index_share`), for the MTP layer's indexer, the only one a draft step runs.
#
# The engine's MTP-Eagle linear draft loop marks the metadata `in_mtp_draft_loop`, hands it the
# trunk's accepted counts (`mtp_num_accepted`) and sets `indexer_skip_topk` from the second draft
# step on (`eagle3._forward_linear_draft_loop`); it does so for every DSA metadata, and each
# indexer's own `mtp_index_share` decides whether to act. With it on, draft step 0 stashes one
# selection row per request in `shared_topk_indices` (a context request's last row, a generation
# request's last accepted row), and the later steps copy those rows instead of scoring and
# selecting. After step 0 the loop rewrites the metadata in place to one row per request in
# request order (`num_contexts = 0`), so stash row `r` is request `r`'s: contexts first, then
# generations. The keys still project and append on every step: the engine's
# `pre_indexer_proj` and `_update_k_cache` run unconditionally.


def _engine_sparse_params(model_config: ModelConfig, layer_idx: int | None):
    """Layer `layer_idx`'s DSA sparse params as the engine's attention module builds them
    (`to_sparse_params(pretrained_config, layer_idx)`), or None without a sparse config."""
    sparse_cfg = getattr(model_config, "sparse_attention_config", None)
    if sparse_cfg is None or not hasattr(sparse_cfg, "to_sparse_params"):
        return None
    return sparse_cfg.to_sparse_params(
        pretrained_config=getattr(model_config, "pretrained_config", None), layer_idx=layer_idx
    )


def _engine_mtp_index_share(model_config: ModelConfig) -> bool:
    """The engine's `mtp_index_share` for the MTP layer's indexer: its sparse params, i.e. the
    LLM arg `index_share_for_mtp_iteration`, else the checkpoint's field of that name (GLM-5.2
    NVFP4's config.json sets it true)."""
    cfg = getattr(model_config, "pretrained_config", None)
    params = _engine_sparse_params(model_config, getattr(cfg, "num_hidden_layers", None))
    return bool(getattr(params, "mtp_index_share", False))


def _dec_topk_gvr_serves(max_seq_len: int) -> bool:
    """Whether the GVR decode top-k serves an engine of this logits width (`max_seq_len`, fixed
    at capture): wider than `_DEC_TOPK_GVR_ABOVE` (`TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`)."""
    above = int(os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE", str(_DEC_TOPK_GVR_ABOVE)))
    return max_seq_len > above


def _dec_topk_gvr_choice(
    params, index_topk: int, *, sm_version: int, dsl_available: bool, env: str | None
) -> bool | None:
    """The decode top-k engine of the generation rows: `gvr_deterministic_topk` where they take
    the stock engine's self-sampling GVR top-k, None where they keep ours.

    Ours by default, under the engine's GVR config too: on long engines our decode top-k splits
    each long row across a CTA cluster, which matches GVR at 64k-160k KV and keeps our one-CTA
    speed at short KV, where GVR's tuning frozen from the engine width is 2-3x slower.
    `TRTLLM_GLM52_DEC_TOPK_GVR=1` (`env`) restores the stock engine's self-sampling GVR top-k
    where the stock DSA indexer would run it on this checkpoint's layers: the engine's
    `enable_heuristic_topk` with the self-sampling engine (`use_self_sampling_gvr`: its
    `use_self_sampling_topk`, CuTe DSL, sm_100 / sm_103, `index_topk` 2048 here, compress ratio
    1). The temporal-hint engine (`use_self_sampling_topk=False`) needs the previous step's
    selection as caller-owned state, which this target does not keep: ours there."""
    if env != "1" or params is None or not getattr(params, "enable_heuristic_topk", False):
        return None
    if index_topk != 2048 or not use_self_sampling_gvr(
        enable_heuristic_topk=True,
        use_self_sampling_topk=bool(getattr(params, "use_self_sampling_topk", True)),
        index_topk=index_topk,
        compress_ratio=1,
        is_cute_dsl_available=dsl_available,
        sm_version=sm_version,
    ):
        return None
    return bool(getattr(params, "gvr_deterministic_topk", False))


_SHARED_INTER = 512
# The widest wide decode step that takes the expert replicas (`_build_moe_wide_rep`): the wide
# top-k's 65..128 range. 4-GPU decode chain with the real all-reduce on real routing (two trunk
# MoE layers with replicas): -6.5 / -5.8 us per step at 128 rows (MTP3), -8.1 / -6.4 / -4.2 us at
# 68 / 96 / 126; at 192 rows (MTP5) -2.7 / +2.8 / +0.4 us over 320 / 800 / 1600 replays a side,
# no gain past the replica top-k's +0.7 us, so 129..256 keep the wide top-k.
_WIDE_REP_MAX_ROWS = 128


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(x.float().clamp_min(2.0**-100))))


def _fold_pow2(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`w [H, rows, K] / s[h]` per head, as a new contiguous tensor in `w`'s dtype. `s` holds
    powers of two, so the division is exact; checked, since an underflow would not be."""
    s = s.float().to(w.device).view(-1, 1, 1)
    out = (w.float() / s).to(w.dtype).contiguous()
    if not torch.equal(out.float() * s, w.float()):
        raise ValueError("fp8 context attention: a power-of-two weight scale was not exact")
    return out


def _try_build(what: str, build):
    """`build()`, or None with a warning: the caller's call sites then keep the stock op."""
    try:
        return build()
    except Exception as e:  # noqa: BLE001 -- that op stays on the stock path
        logger.warning(f"modeling_v2 GLM-5.2: {what} build failed ({e}); using the stock op")
        return None


class GlmMoeDsaTep4ModelingV2(DecoderModel):
    # Off unless `__init__` reads them on (`TRTLLM_GLM52_MOE_DEC_UNPAD*`): a core built without
    # `__init__` (the MoE unit tests) keeps the padded decode path.
    _moe_dec_unpad = False
    _moe_dec_unpad_signal = False
    _moe_topk_signaled = False

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.tp_rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.quant_mode = _QUANT_MODE_FP8_KV
        # TRTLLM_GLM52_DENSE_TOPK_ON_DEVICE (default 1): when every context sequence is shorter
        # than the index top-k, the engine skips the indexer and builds a dense causal table of
        # `num_ctx_tokens x index_topk` entries. Its host build (a CPU kernel plus a copy) costs
        # ~97 ms of `_prepare_inputs` per 8k-token context step on B300 with the GPU idle; the
        # metadata's `dense_topk_on_device` builds the same integer table on the GPU.
        self._dense_topk_on_device = os.environ.get("TRTLLM_GLM52_DENSE_TOPK_ON_DEVICE", "1") == "1"
        self._mixed_base = os.environ.get("TRTLLM_GLM52_MIXED_BASE", "1") == "1"
        # TRTLLM_GLM52_CTX_HOST_DIET (default 1; 0 restores the per-call paths): host time
        # only, same kernels and arguments. A context step reads its per-layer pool addressing
        # from python tables built once per step (`PrefillTarget._diet_tables`); the launch
        # wrappers read raw stream handles (`kernels/common/cuda_stream.py`); eager calls of the
        # decode-cell GEMMs a mixed step's generation rows take (q_b, K-B, V-B) launch TVM-FFI
        # twins of their kernels; the context rope/append and the generation sparse MLA launch a
        # call signature their `prepare_runner` passed before without its contract walk.
        # Read once, at import, by `kernels/common/cuda_stream.py`: the launch wrappers and
        # this target must agree on it.
        self._ctx_host_diet = _cuda_stream.DIET
        # TRTLLM_GLM52_FMHA_UNION (default 1): MTP3 / MTP5 pure decode steps take the
        # union decode attention (`cute_dsl_sparse_mla_decode_union`); 0 keeps the per-row one.
        self._union_on = os.environ.get("TRTLLM_GLM52_FMHA_UNION", "1") == "1"

        self.num_layers = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.heads = cfg.num_attention_heads // mapping.tp_size
        self.nope = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora = cfg.kv_lora_rank
        self.q_lora = cfg.q_lora_rank
        self.qk_dim = self.nope + self.rope_dim
        self.lat_dim = self.kv_lora + self.rope_dim
        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = (rope_cfg or {}).get("rope_theta", getattr(cfg, "rope_theta", None))
        self.theta = float(theta)

        self.idx_heads = cfg.index_n_heads
        self.idx_dim = cfg.index_head_dim
        self.idx_topk = cfg.index_topk
        # The indexer's whole softmax scale: the scorers take none, so the q side folds it into
        # `weights` together with the q quantization scale.
        self.idx_weight_scale = self.idx_dim**-0.5 * self.idx_heads**-0.5
        # The checkpoint's layout, read as stock's model_config reads it (GLM-5.2: interleaved).
        # The two layouts score alike up to ~32k tokens; past that the wrong one selects the
        # wrong keys, which a short-prompt accuracy check cannot see.
        self.indexer_is_neox = not getattr(cfg, "indexer_rope_interleave", False)

        self.dense_layers = cfg.first_k_dense_replace
        # Each dense layer's MXF8 gate|up and down (`Mlp.dense`); set by `_build_proj_mxf8`.
        self._dense_w8: list = [None] * self.dense_layers
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scale = float(cfg.routed_scaling_factor)
        self.moe_inter = cfg.moe_intermediate_size
        self.local_experts = self.num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank
        # Decode expert replicas (`weights.DEC_HOT`): every rank also holds each trunk MoE
        # layer's `dec_replicas` hottest experts in slots 64.. of its routed stacks; 0 = off.
        self.dec_replicas = _weights.dec_replicas()
        self.dec_hot = (
            {i: _weights.dec_hot(i, self.dec_replicas) for i in _weights.DEC_HOT}
            if self.dec_replicas
            else {}
        )
        # Context expert replicas (`TRTLLM_GLM52_CTX_EXPERT_REPLICAS=R`, 0 = off): a context
        # step's top-k splits the rows of each trunk MoE layer's first R replica experts
        # across the EP ranks by producer CTA, levelling the ranks' grouped-GEMM loads (the
        # slowest rank sets the all-reduce wait). Runs on the decode replicas' slots, so R is
        # at most `dec_replicas`.
        # Default: the decode replica count (TRTLLM_GLM52_DEC_EXPERT_REPLICAS=0 turns both off).
        self.ctx_replicas = int(
            os.environ.get("TRTLLM_GLM52_CTX_EXPERT_REPLICAS", str(self.dec_replicas))
        )
        if not 0 <= self.ctx_replicas <= self.dec_replicas:
            raise ValueError(
                "TRTLLM_GLM52_CTX_EXPERT_REPLICAS must be in 0..TRTLLM_GLM52_DEC_EXPERT_REPLICAS "
                f"({self.dec_replicas}), got {self.ctx_replicas}"
            )
        # Wide decode expert replicas (`TRTLLM_GLM52_WIDE_DEC_EXPERT_REPLICAS`: "1", the
        # default, or "0"): an MTP3 / MTP5 decode step of 65..256 rows moves each trunk MoE
        # layer's active hot experts whole to the least loaded EP ranks, on the context
        # replicas' slots, prefill pair and hot ids (`_build_moe_wide_rep`; needs
        # `ctx_replicas` > 0).
        self.wide_dec_replicas = os.environ.get("TRTLLM_GLM52_WIDE_DEC_EXPERT_REPLICAS", "1") == "1"
        # The wide decode top-k's schedule (`TRTLLM_GLM52_WIDE_TOPK_CLUSTER16`: "1", the
        # default, or "0"): the MTP3 / MTP5 decode steps' fused top-k (and its replica
        # variant) on a 16-CTA cluster, half the token-warps per SM, with the trimmed winner
        # lookup; "0" keeps the portable 8-CTA kernel as it was. Same ids, weights, counts
        # and tile layout either way (route `moe_topk_wide_cl16` counts the 16-CTA launches).
        cl16 = os.environ.get("TRTLLM_GLM52_WIDE_TOPK_CLUSTER16", "1") == "1"
        self.wide_topk_schedule = (16, True) if cl16 else (8, False)
        self.mtp_enabled = getattr(model_config, "spec_config", None) is not None
        # The routed experts' FC1 weight layout (`TRTLLM_GLM52_MOE_FC1_KBLK`, `moe_fc1_kblock`),
        # read once here: the loader (`weights._fc1_w`), the MTP layer's quantized experts and
        # every compiled FC1 kernel follow this one value.
        self.fc1_kblk = _fc1_kblock.enabled()
        # The same for FC2 (`TRTLLM_GLM52_MOE_FC2_KBLK`): the loader (`weights._fc2_w`), the
        # MTP layer's quantized FC2 and every compiled FC2 kernel follow this one value.
        self.fc2_kblk = _fc1_kblock.fc2_enabled()
        # Draft tokens per step (MTP-Eagle linear chain: 1, 3 or 5, see routing.py); 0 without
        # speculation. A generation step carries max_draft_len + 1 query rows per sequence.
        self.max_draft_len = (
            int(getattr(model_config.spec_config, "max_draft_len", None) or 1)
            if self.mtp_enabled
            else 0
        )
        # The generation widths (query rows per sequence) the width-specialised decode builds
        # serve: MTP1's 2, plus max_draft_len + 1 at MTP3 / MTP5 (verify and draft step 0).
        # Draft steps 1..D-1 carry 1 row per sequence.
        self.mtp_widths = (_MLA_ROPE_MTP_WIDTH,) + (
            (self.max_draft_len + 1,) if self.max_draft_len > 1 else ()
        )
        # Whether the MTP layer returns its head-normalized output. The engine's linear draft
        # loop feeds the MTP layer's return value back in as the next draft step's hidden state,
        # and stock's MTP layer returns `shared_head.norm(...)`: an un-normalized feed-back
        # collapsed MTP3 / MTP5 acceptance at draft positions 2.. (3.27 vs stock 3.98 at MTP3).
        # MTP1 never feeds back, so it keeps the norm inside `shared_head` (unchanged bytes).
        self._mtp_out_normed = self.max_draft_len > 1
        # The engine's draft-loop index share for the MTP layer's indexer (`_engine_mtp_index_share`;
        # the GLM-5.2 checkpoint turns it on): draft steps after the first reuse draft step 0's
        # selection (`_mtp_share_*`). Only a loop of more than one draft step reads the stash.
        self.mtp_index_share = self.mtp_enabled and _engine_mtp_index_share(model_config)
        self._mtp_share_stash = self.mtp_index_share and self.max_draft_len > 1
        # With TRTLLM_GLM52_DEC_TOPK_GVR=1 and the engine's sparse config asking for the GVR
        # decode top-k (`enable_heuristic_topk`, `_dec_topk_gvr_choice`), the generation rows'
        # top-k runs the stock engine's self-sampling GVR engine through its `TopK` module, as
        # the stock DSA indexer does; otherwise (None) ours, the cluster split on long engines.
        # `_derive_first_step` keeps ours on engines no wider than `_DEC_TOPK_GVR_ABOVE`
        # (`_dec_topk_gvr_serves`) and warms GVR's launchers.
        gvr_det = _dec_topk_gvr_choice(
            _engine_sparse_params(model_config, 0),
            self.idx_topk,
            sm_version=get_sm_version(),
            dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            env=os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR"),
        )
        # Whether it serves this engine's width (`_derive_first_step`).
        self._dec_topk_gvr_live = False
        self._dec_topk_gvr = (
            None
            if gvr_det is None
            else TopK(
                self.idx_topk,
                decode_implementation=TopKImplementation.CUTE_DSL_GVR,
                gvr_self_sampling=True,
                gvr_deterministic=gvr_det,
            )
        )
        # An MTP3 / MTP5 decode step's rows above 64 (verify and draft step 0 carry 32 x
        # (max_draft_len + 1)) are decode rows (`WideDecodeTarget`). Its rows without an MTP
        # cell of their own (a batch of fewer than 32 sequences: 65..191 rows, 96 / 144 at 24
        # requests) run on the next cell up at their exact row count (`Gemm.wide_cell`); False
        # sends them to the bf16 fallbacks instead.
        self._wide_bucket = True
        # TRTLLM_GLM52_DEC_CELL_UP (default 1; 0 restores cuBLAS there): an MTP3 / MTP5
        # decode step's (`Phase.wide`, graph-captured) row counts up to 64 without a decode
        # cell of their own (MTP5 at 1-3 requests: 6 / 12 / 18; MTP3 at 3: 12) take the same
        # rule as an eager draft step (`Gemm.cell_up`): the next cell up at their exact rows.
        # Measured at MTP5 b=2 (12 rows): the indexer projection there ran as `x.float()`
        # + an fp32 SIMT cuBLAS GEMM + split-K reduce + copies (27 us per indexer layer).
        self._dec_cell_up = os.environ.get("TRTLLM_GLM52_DEC_CELL_UP", "1") == "1"
        # Whether `DecodeTarget` runs with that rule (`Phase(up="decode")`).
        self.dec_cell_up = self.max_draft_len > 1 and self._dec_cell_up
        # Whether this engine builds the wide decode cells, the kernels `WideDecodeTarget`
        # runs a pure decode step above 64 rows on (`Phase.wide`): the projections' MTP
        # cells, the wide decode MXF8 o_proj, the folded qkva / q_b GEMMs, the indexer's fused
        # q side + K update, and the routed experts' wide front end on the exact rows. Every
        # engine builds them: an MTP1 or non-speculative engine's decode steps reach them above
        # 32 / 64 sequences, where they beat `DecodeTarget`'s context-sized kernels (GB300,
        # TPOT: no speculation 96 / 128 / 192 rows -17 / -6 / -15%, MTP1 96..256 rows -6 to
        # -20%). TRTLLM_GLM52_WIDE_DECODE=0 keeps such an engine's decode steps on
        # `DecodeTarget` at any row count; an engine drafting three or five tokens always
        # builds them (its decode steps reach 32 x 4 / 6 rows).
        self.wide_cells = (
            self.max_draft_len > 1 or os.environ.get("TRTLLM_GLM52_WIDE_DECODE", "1") == "1"
        )
        # TRTLLM_GLM52_MOE_ACC_PADDED (default 1; 0 restores the add after FC2): at a decode
        # width up to 64 that routing pads to a power of two (MTP5 at 1-3 requests: 6 / 12 /
        # 18 -> 8 / 16 / 32) the shared expert also writes the routed slot's first rows and
        # FC2 accumulates onto them (`Moe.begin_layer`), as at the power-of-two widths: no
        # clear, and no separate shared + routed add on the critical chain after FC2.
        self._moe_acc_padded = os.environ.get("TRTLLM_GLM52_MOE_ACC_PADDED", "1") == "1"
        # TRTLLM_GLM52_MOE_DEC_UNPAD (default 1; 0 restores the padding): a trunk MoE layer at a
        # decode width up to 64 that is not a power of two (MTP5 at 1-3 requests: 6 / 12 / 18)
        # runs its grouped GEMMs on the fused top-k's own maps at the exact row count
        # (`Moe.dec_unpadded`), as the power-of-two widths do. Padding it cost seven launches
        # per layer between the top-k and FC1 (two id / weight fills and cats, two staging
        # copies, the routing kernel: 8.3 us of the critical chain at 12 rows on B300)
        # and dropped the decode replicas' maps for the routing kernel's home mapping.
        self._moe_dec_unpad = os.environ.get("TRTLLM_GLM52_MOE_DEC_UNPAD", "1") == "1"
        # TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL (default 1): on an MTP3 / MTP5 engine the unpadded
        # trunk widths that are multiples of `max_draft_len + 1` and not powers of two also get
        # a top-k handle that releases FC1 early (`k.topk_sig`), as the power-of-two widths'
        # do: FC1 follows the top-k directly there. The MTP layer keeps the plain handles.
        self._moe_dec_unpad_signal = os.environ.get("TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL", "1") == "1"
        # TRTLLM_GLM52_MOE_L2_PREFETCH (default 1; 0 skips building the decode L2 prefetch).
        self._moe_l2_prefetch = os.environ.get("TRTLLM_GLM52_MOE_L2_PREFETCH", "1") == "1"
        # The fused shared-expert cells at those row counts (`_SHARED_FUSED_UP_SEQS`).
        self._shared_gu_swiglu_up: dict = {}
        self._ix_pre_split_rag = None
        # The MTP3 / MTP5 decode cells' K-B writing the e4m3 q-base beside the rope kernel
        # (`Attn.gen_query`), by row count; empty keeps the bf16 K-B and the full rope there.
        self._kb8_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_IX_FOLD (default 1): on an MTP3 / MTP5 engine an indexer layer's
        # wide decode step (rows above 64) runs the qkva projection and the indexer's key /
        # head-weight projection as one GEMM over the two weights stacked in one buffer
        # (`Gemm.qkva_ix_fold`); 0 keeps the two GEMMs. Set up in `post_load_weights`: the
        # stacked weights by layer and the fold cells by row count; empty keeps the two.
        self._ix_fold_on = os.environ.get("TRTLLM_GLM52_WIDE_IX_FOLD", "1") == "1"
        self._ix_fold_w: dict[int, torch.Tensor] = {}
        self._qkva_ix_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_QB_IXQB (default 1): likewise q_b and the indexer's wq_b (both read
        # q_a) as one GEMM over their two weights stacked `[2, 4096, 2048]` (`Gemm.qb_ixqb`), where
        # the step runs the indexer; 0 keeps the two GEMMs. Stacked weights by layer, cells by
        # row count, both set up in `post_load_weights`.
        self._qbix_on = os.environ.get("TRTLLM_GLM52_WIDE_QB_IXQB", "1") == "1"
        self._qbix_w: dict[int, torch.Tensor] = {}
        self._qbix_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_IX_QK (default 1): there, too, the indexer's q side (rope + e4m3
        # pack of its q, the head weights) and this step's K-cache update run as one launch
        # (`cute_dsl_indexer_q_side_k_cache`: each kernel's CTAs in one grid, the same bytes);
        # 0 keeps the two launches. Both pool layouts' launchers, built with the K update's.
        self._ix_qk_on = os.environ.get("TRTLLM_GLM52_WIDE_IX_QK", "1") == "1"
        self._ix_qk_compiled: dict = {}

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        self._derive_pending = True
        self._derived_for = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._idx_cos_sin: torch.Tensor | None = None
        self._pool_layer_id: list[int] = []
        self._stride_factor = 0
        self._max_ctx_len = 0
        self._mtp: dict | None = None
        self._ctx_rope_compiled: _ctx_rope_op.CompiledMlaRopeAppendPrefill | None = None
        # The non-absorbed context attention (`PrefillTarget._attn_mha`): its MHA, its
        # cached-prefix gather and the two kv_b expansion runners; empty or None keeps every
        # context step on the absorbed path.
        self._mha_pre: _mha_pre_op.CompiledMhaPrefill | None = None
        self._mha_prefix: _mha_prefix_op.CompiledMhaPrefixGather | None = None
        self._kvb_expand: dict = {}
        # The fp8 form of that attention (`_build_context_mha_fp8`) and its q_b runner.
        self._mha_fp8 = False
        self._q8 = None
        # qkva / o_proj in MXF8 for the context rows (`_build_proj_mxf8`).
        self._proj_mxf8: dict = {}
        # MTP3 / MTP5: the trunk o_proj's wide decode MXF8 GEMM and the V-B writing its
        # operand (`_build_dec_mxf8`); empty keeps those rows on the context MXF8 GEMM.
        self._dec8w: dict = {}
        self._vb8w: dict = {}
        self._fmha_dec_compiled: _fmha_dec_op.CompiledSparseMlaDecode | None = None
        # MTP3 / MTP5 only: the decode attention's grid for max_draft_len + 1 rows per sequence.
        self._fmha_dec_wide: _fmha_dec_op.CompiledSparseMlaDecode | None = None
        # MTP3 / MTP5 pure decode: the union decode attention's handle (None: per-row only).
        self._fmha_union: _fmha_union_op.CompiledSparseMlaDecodeUnion | None = None
        # The ops the Targets call (`forward_ops`). They read the weights and the kernels this
        # core builds at load, and keep the per-step state the kernels share across calls: the
        # routed experts' operand slots alternate per layer (trunk) or per call (MTP).
        self.op = SimpleNamespace(
            rows=Rows(self),
            comm=Comm(self),
            gemm=Gemm(self),
            attn=Attn(self),
            indexer=Indexer(self),
            mlp=Mlp(self),
            moe=Moe(self),
            mtp=Mtp(self),
            ctx=Ctx(self),
        )
        # Context steps on the dense-causal context attention, and the others by reason.
        self.dense_causal_steps = 0
        self.dense_causal_misses: dict[str, int] = {}
        # The dense-causal group table's sync-free uploader, built at first use.
        self._dc_uploader = None  # `PrefillTarget._dense_causal_absorbed`
        # The row ops outside the fused kernels and the logits head's local GEMM, built in
        # `post_load_weights` and `build_logits_head`; None keeps a call site's stock op.
        self._rows = SimpleNamespace(
            norm=None,
            add=None,
            add_norm=None,
            gather=None,
            topk=None,
            swiglu=None,
            copy=None,
            argmax=None,
        )
        self._lm_head = None
        self._lm_head_runners: dict = {}
        # Per-route counters, by the name a smoke asserts on: `route_hits[name]` counts the
        # calls that ran the CuTe DSL route, `route_misses[name]` those its call site sent to
        # the stock op. Python-side, so they count eager forwards and each CUDA graph's
        # capture, not its replays. Created here, before the weights load, and never
        # replaced: the load-time builds above only fill in what they point at.
        self.route_hits: dict[str, int] = {}
        # Context steps on which the MTP layer computed every row, by reason
        # (`PrefillTarget._mtp_last_plan`).
        self.mtp_last_declines: dict[str, int] = {}
        self.route_misses: dict[str, int] = {}
        self._forwards = 0

    def _rope_tables(self, device, positions: int) -> dict:
        """Two layouts of one table, built in fp64 and rounded once.

        `rotary_cos_sin` is the duplicated interleaved layout the MLA ops read. The indexer
        table is `[positions, rope]` with cos in the first half and sin in the second.
        `positions` is a row count: every row depends only on its own index, so a longer table
        is the same table with more rows.
        """
        half = self.rope_dim // 2
        inv = 1.0 / (self.theta ** (torch.arange(half, dtype=torch.float64) * 2.0 / self.rope_dim))
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos(), ang.sin()
        table = torch.empty(positions, self.rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        self._idx_cos_sin = torch.cat([cos, sin], dim=1).float().to(device).contiguous()
        return {
            "rotary_cos_sin": table.reshape(1, positions * self.rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": self.rope_dim,
            "rope_base": self.theta,
        }

    def _derive_attention(self, i: int) -> SimpleNamespace:
        """One layer's attention operands: column-major GEMM views and the two absorption
        operands of the regrouped kv_b_proj, plus each in the `[H, N, K]` layout the absorbed
        BMM kernels take (K-B's is a copy, made here)."""
        w = self.w
        kvb = w[f"l{i}_kvb"]
        hn = self.heads * self.nope
        k_b = kvb[:hn].reshape(self.heads, self.nope, self.kv_lora)
        v_b_nk = kvb[hn:].reshape(self.heads, self.v_dim, self.kv_lora)
        return SimpleNamespace(
            norm1=w[f"l{i}_norm1"],
            qkva=w[f"l{i}_qkva"].t(),
            q_norm=w[f"l{i}_q_norm"],
            kv_norm=w[f"l{i}_kv_norm"],
            qb=w[f"l{i}_qb"].t(),
            k_b=k_b,
            v_b_t=torch.transpose(v_b_nk, 1, 2),
            k_b_nk=torch.transpose(k_b, 1, 2).contiguous(),
            v_b_nk=v_b_nk,
            o=w[f"l{i}_o"].t(),
            norm2=w[f"l{i}_norm2"],
            # The non-absorbed context attention's weight, set by `_build_context_mha`.
            kvb_aug=None,
            # Its fp8 form, also set there: the K / V expansion weights with the per-head
            # scales folded in (`[4096, 576]`, `[4096, 512]`), q_b with the Q scale folded in
            # (MXF8 or bf16), and the MHA's per-head factor table.
            kvb8=None,
            qb8=None,
            mha_hs=None,
            # qkva / o_proj in MXF8 for the context rows, set in `post_load_weights`.
            qkva8=None,
            o8=None,
            # Decode rows' q_b / o_proj on MXF8 weights (`_build_dec_mxf8`), and the power-of-two
            # Q scale folded into `qb8`.
            qb_dec8=None,
            o_dec8=None,
            mha_sq=None,
        )

    def _kv_b_augmented(self, i: int) -> torch.Tensor:
        """Layer `i`'s kv_b as the non-absorbed context attention's expansion weight,
        `[heads * (qk + v), kv_lora + rope]` = `[8192, 576]` bf16: the K rows of head h are
        `[W_UK_h | 0 ; 0 | I_64]` (the nope part, then the roped k_pe copied exactly by the
        identity block), the V rows `[W_UV_h | 0]`. A new tensor; the absorbed path's views
        of kv_b are untouched."""
        kvb = self.w[f"l{i}_kvb"].detach()
        h, nope, rope, v, lora = self.heads, self.nope, self.rope_dim, self.v_dim, self.kv_lora
        w = torch.zeros(h * (nope + rope + v), lora + rope, dtype=kvb.dtype, device=kvb.device)
        wk = w[: h * (nope + rope)].view(h, nope + rope, lora + rope)
        wk[:, :nope, :lora] = kvb[: h * nope].view(h, nope, lora)
        wk[:, nope:, lora:] = torch.eye(rope, dtype=kvb.dtype, device=kvb.device)
        wv = w[h * (nope + rope) :].view(h, v, lora + rope)
        wv[:, :, :lora] = kvb[h * nope :].view(h, v, lora)
        return w

    def _build_proj_mxf8(self) -> None:
        """The context rows' qkva and o_proj as MXF8 GEMMs: e4m3 operands with e8m0 1x32
        block scales, fp32 accumulation, bf16 out. Each trunk layer's two weights are
        quantized once here (3.3 GB per rank beside the bf16 copies, which the decode cells
        keep); `Gemm.project` quantizes the activation with one launch per call. On B300 at
        T ~ 7-8k quant + GEMM takes 0.61-0.76x the bf16 prefill GEMM. Decode cells (rows
        <= 64) and the MTP layer stay bf16; an entry that fails to build keeps the bf16
        prefill GEMM."""
        self._proj_mxf8: dict[str, _mxf8_op.Mxf8ProjectionRunner] = {}
        for name, attr in (("qkva", "qkva"), ("o_proj", "o")):
            wq = [
                _mxf8_op.quantize_weight(getattr(self._attn[i], attr).t())
                for i in range(self.num_layers)
            ]
            runner = _try_build(f"MXF8 {name}", lambda name=name, w=wq[0]: _mxf8_op.build(name, w))
            if runner is None:
                continue
            for i, w in enumerate(wq):
                setattr(self._attn[i], f"{attr}8", w)
            self._proj_mxf8[name] = runner
        self._build_mlp_mxf8()
        # Each dense layer's MXF8 gate|up and down (`Mlp.dense`), None where they did not build.
        self._dense_w8 = (
            list(self._dense8) if self._dense8 is not None else [None] * self.dense_layers
        )

    def _build_mlp_mxf8(self) -> None:
        """The context rows' MLPs on MXF8 GEMMs, their weights quantized once here.

        Dense layers: gate|up, the SwiGLU writing the down GEMM's operand
        (`cute_dsl_swiglu_mxf8`), down (+0.17 GB per rank). At 7-8k rows 0.82 against 1.61 ms
        per context step in bf16. MoE layers: the shared expert's gate|up with the SwiGLU in
        the GEMM's epilogue (`cute_dsl_shared_up_swiglu_mxf8`, +0.47 GB per rank), on the
        operand the pre-MoE all-reduce writes; its down stays bf16 (the MXF8 down moved the
        output distribution more than the accuracy budget allows). Either part that fails to
        build keeps its bf16 path."""
        dev = torch.cuda.current_device()
        self._dense8: list | None = None
        dense = range(self.dense_layers)
        wq = {
            name: [_mxf8_op.quantize_weight(self.w[f"l{i}_mlp_{key}"]) for i in dense]
            for name, key in (("dense_gate_up", "gu"), ("dense_down", "dn"))
        }
        runners = {
            name: _try_build(f"MXF8 {name}", lambda name=name: _mxf8_op.build(name, wq[name][0]))
            for name in wq
        }
        swiglu = _try_build("SwiGLU into MXF8", lambda: _swiglu8_op.compile(dev))
        if swiglu is not None and all(runners.values()):
            self._proj_mxf8.update(runners)
            self._swiglu8 = swiglu
            self._dense8 = list(zip(wq["dense_gate_up"], wq["dense_down"]))
            # TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI (default 1): gate|up with the SwiGLU into the
            # down GEMM's MXF8 operand in its epilogue (`cute_dsl_dense_up_swiglu_mxf8`, its own
            # interleaved weight, +0.11 GB per rank): one launch instead of two and no bf16
            # [T, 6144].
            dense_sw = (
                _try_build("dense gate|up + SwiGLU MXF8", lambda: _dense_up8_op.compile(dev))
                if os.environ.get("TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI", "1") == "1"
                else None
            )
            if dense_sw is not None:
                self._dense_sw8 = dense_sw
                self._dense8 = [
                    (gu8, dn8, _dense_up8_op.prepare_weight(self.w[f"l{i}_mlp_gu"]))
                    for i, (gu8, dn8) in zip(dense, self._dense8)
                ]
        self._shared8 = _try_build("shared gate|up MXF8", lambda: _shared_up8_op.compile(dev))
        self._shared_w8 = [
            _shared_up8_op.prepare_weight(self.w[f"l{i}_mlp_gu"])
            if self._shared8 is not None and i >= self.dense_layers
            else None
            for i in range(self.num_layers)
        ]
        # The MTP module's replicated eh_proj on context rows: its two input norms write the
        # MXF8 operand, then the MXF8 GEMM (draft-only numerics: they move MTP acceptance).
        self._mtp_eh8 = None
        if self.mtp_enabled:
            w8 = _mxf8_op.quantize_weight(self._mtp["eh"].t())
            runner = _try_build("MXF8 mtp_eh", lambda: _mxf8_op.build("mtp_eh", w8))
            norm = _try_build("MTP input norms into MXF8", lambda: _eh8_op.compile(dev))
            if runner is not None and norm is not None:
                self._proj_mxf8["mtp_eh"] = runner
                self._mtp_eh8 = SimpleNamespace(norm=norm, w8=w8)

    def _build_dec_mxf8(self, dev) -> None:
        """Decode rows (1..64) of the trunk layers' q_b and o_proj on MXF8 weights the context
        path already keeps: o_proj on `o8`; q_b on the fp8 attention's `qb8` (`q_b / s_q`)
        with `log2(s_q)` added back to its block scales -- exact, s_q being a power of two,
        and checked against a direct quantization (a mismatch keeps the direct copy). At
        decode sizes these GEMMs read their weight and little else, and e4m3 halves it.

        The producers then write the operands into the launchers' own buffers, so on a pure
        decode step neither GEMM has an activation quant launch in front of it: the q_a
        norm's decode kernel (`cute_dsl_qkv_a_rmsnorm_q8`) for q_b, and the V-B GEMM's side
        output (`build_v_b_mxf8_side`, the o_proj operand's addresses compiled in) for
        o_proj. The MTP layer and a mixed step's generation rows keep the bf16 cells.

        MTP3 / MTP5 (`max_draft_len > 1`): the trunk o_proj's decode rows above 64 (up to
        192) on the wide decode MXF8 GEMM (`_dec8w`, `_dec8_op.build_wide`), its operand
        written by the MTP group's V-B side output (`_vb8w`). Same e4m3 operands and K order
        as the context MXF8 GEMM those rows took before (`Gemm.mxf8_takes`): bitwise its output,
        without the act-quant launch, the weight streamed before the PDL wait.
        TRTLLM_GLM52_WIDE_OPROJ_DEC8=0 keeps those rows on the context MXF8 GEMM."""
        self._dec8: dict = {}
        self._vb8: dict = {}
        self._dec8w: dict = {}
        self._vb8w: dict = {}
        trunk = range(self.num_layers)
        if "o_proj" in self._proj_mxf8:
            runner = _try_build(
                "decode MXF8 o_proj", lambda: _dec8_op.build("o_proj", self._attn[0].o8)
            )
            if runner is not None:
                self._dec8["o_proj"] = runner
                for i in trunk:
                    self._attn[i].o_dec8 = self._attn[i].o8
        qb8 = [self._attn[i].qb8 for i in trunk]
        if all(isinstance(w, _mxf8_op.QuantizedWeight) for w in qb8):
            weights = []
            for i, w in enumerate(qb8):
                shifted = _dec8_op.shift_scales(w, round(math.log2(self._attn[i].mha_sq)))
                direct = _mxf8_op.quantize_weight(self._attn[i].qb.t())
                same = torch.equal(direct.q, shifted.q) and torch.equal(direct.sf, shifted.sf)
                weights.append(shifted if same else direct)
            runner = _try_build("decode MXF8 q_b", lambda: _dec8_op.build("q_b", weights[0]))
            if runner is not None:
                self._dec8["q_b"] = runner
                for i, w in enumerate(weights):
                    self._attn[i].qb_dec8 = w
        if "o_proj" in self._dec8:
            for group in _absorbed_bmm_op.GROUPS:
                vb8 = _try_build(
                    f"V-B with o_proj's MXF8 operand {group}",
                    lambda group=group: _absorbed_bmm_op.build_v_b_mxf8_side(
                        group, dev, self._dec8["o_proj"].operand
                    ),
                )
                if vb8 is None:
                    self._vb8 = {}
                    break
                self._vb8.update(dict.fromkeys(vb8.rows, vb8))
        if (
            "o_proj" in self._dec8
            and self.wide_cells
            and os.environ.get("TRTLLM_GLM52_WIDE_OPROJ_DEC8", "1") == "1"
        ):
            runner = _try_build(
                "wide decode MXF8 o_proj",
                lambda: _dec8_op.build_wide("o_proj", self._attn[0].o8),
            )
            if runner is not None:
                self._dec8w["o_proj"] = runner
                vb8 = _try_build(
                    "MTP V-B with the wide o_proj's MXF8 operand",
                    lambda: _absorbed_bmm_op.build_v_b_mxf8_side(
                        _absorbed_bmm_op.MTP_GROUPS[0], dev, runner.operand
                    ),
                )
                if vb8 is not None:
                    self._vb8w = dict.fromkeys(vb8.rows, vb8)

    def _build_context_mha(self, dev) -> None:
        """The non-absorbed context attention of dense-causal steps: the causal varlen MHA,
        its cached-prefix gather, and the kv_b expansion on the prefill GEMM. The fp8 form
        (`_build_context_mha_fp8`) when it builds, else the bf16 form: the expansion on the
        `kv_b_k` / `kv_b_v` entries against one augmented weight per trunk layer
        (`a.kvb_aug`, 9.4 MB each). The MTP layer keeps the absorbed path. If any part fails
        to build, every context step keeps the absorbed path."""
        self._mha_fp8 = False
        # The fp8 form's MXF8 K / V expansion (gather + GEMM runners) and its MHA handle that
        # writes O as o_proj's MXF8 operand; None where they did not build.
        self._kvb8 = None
        self._mha_q8 = None
        self._mha_prefix = _try_build(
            "context MHA prefix gather", lambda: _mha_prefix_op.compile(dev)
        )
        if self._mha_prefix is None:
            return
        if self._build_context_mha_fp8(dev):
            return
        self._mha_pre = _try_build("context MHA", lambda: _mha_pre_op.compile(dev))
        if self._mha_pre is None:
            return
        aug = [self._kv_b_augmented(i) for i in range(self.num_layers)]
        k_rows = self.heads * self.qk_dim
        expand = {
            half: _try_build(
                f"kv_b expansion {half.upper()}",
                lambda half=half: _projection_op.build_prefill(
                    f"kv_b_{half}", aug[0][:k_rows] if half == "k" else aug[0][k_rows:]
                ),
            )
            for half in ("k", "v")
        }
        if not all(expand.values()):
            return
        for i, w in enumerate(aug):
            self._attn[i].kvb_aug = w
        self._kvb_expand = expand

    def _build_context_mha_fp8(self, dev) -> bool:
        """The fp8 context attention: q from q_b's e4m3 output, K / V from e4m3 expansion
        GEMMs, and the fp8 MHA (e4m3 QK^T and PV, fp32 accumulation, bf16 out). The
        dequantization scales are powers of two from strict bounds (`_mha_fp8_scales`),
        folded into route-only weight copies (exact) and read by the MHA as a per-head
        table, so no context row can leave the e4m3 range. The V expansion reads only the
        latent row's c_kv part (`[4096, 512]` = W_UV; the augmented V rows' rope block is
        zero). q_b runs in MXF8 when the MXF8 projections built (the bf16 copy of the
        folded weight is then not kept), else on the prefill GEMM's e4m3 entry. Against
        the bf16 form at a ~7.3k-row step on B300: -13..-15 % per layer for q_b, rope,
        expansion and MHA; attention output 4.3e-2 .. 5.9e-2 relative RMS against fp32
        (absorbed fp8 chain 6.3e-2 .. 8.1e-2). False (nothing kept) if a part fails.

        With the MXF8 projections built, two more parts when they build: the K / V expansion
        in MXF8 -- one launch per layer writes the `[cached | new]` latent rows as MXF8
        (`cute_dsl_mha_kv_gather_mxf8`: pool bytes at 2^0 for a cached row, the new rows
        block-quantized) and both expansion GEMMs run e4m3 x e4m3 with 1x32 block scales
        against the folded weights quantized once (their bf16 copies not kept; the K / V
        bounds take `_MHA_FP8_KV_MXF8` more headroom) -- as ONE launch (`kv_b_kv8`, the two
        folded weights stacked, V zero-padded to 576; the MHA handles then read K / V as the
        two halves of its `[T_k, 8192]` rows, `kv_strided`) when that entry and the strided
        MHA handle build, with the two launches' output bytes; and the MHA handle whose epilogue
        writes O as o_proj's MXF8 operand (`compile(q8=True)`), which o_proj then reads
        without its own quant launch (measured in a served run, MHA + quant + o_proj
        -11.9 .. -13.5 us per layer at a ~7.3k-row step)."""
        mha = _try_build("context MHA fp8", lambda: _mha_pre_op.compile(dev, fp8=True))
        if mha is None:
            return False
        mxf8 = "qkva" in self._proj_mxf8
        q_extra = _MHA_FP8_Q_MXF8 if mxf8 else 1.0
        h, nope, rope, v, lora = self.heads, self.nope, self.rope_dim, self.v_dim, self.kv_lora
        kvb8 = self._build_kvb8(dev) if mxf8 else None
        if kvb8 is not None and kvb8.merged is not None:
            mha_s = _try_build(
                "context MHA fp8 on strided K / V",
                lambda: _mha_pre_op.compile(dev, fp8=True, kv_strided=True),
            )
            if mha_s is None:
                kvb8.merged = None
            else:
                mha = mha_s
        merged = kvb8 is not None and kvb8.merged is not None
        kv_extra = _MHA_FP8_KV_MXF8 if kvb8 is not None else 1.0
        ranges: list[tuple[float, ...]] = []
        layers = []
        for i in range(self.num_layers):
            a = self._attn[i]
            s_q, s_k, s_v = self._mha_fp8_scales(i, q_extra, kv_extra)
            kvb = self.w[f"l{i}_kvb"].detach()
            w_k = torch.zeros(h, nope + rope, lora + rope, dtype=kvb.dtype, device=kvb.device)
            w_k[:, :nope, :lora] = kvb[: h * nope].view(h, nope, lora)
            w_k[:, nope:, lora:] = torch.eye(rope, dtype=kvb.dtype, device=kvb.device)
            w_v = kvb[h * nope :].view(h, v, lora)
            w_k8 = _fold_pow2(w_k, s_k).view(h * (nope + rope), lora + rope)
            w_v8 = _fold_pow2(w_v, s_v).view(h * v, lora)
            if merged:
                # K rows over V rows zero-padded to 576: rows quantize independently, so the
                # K rows' bytes are the K weight's and the V rows' first 512 columns the V's.
                w_kv = torch.zeros(
                    h * (nope + rope + v), lora + rope, dtype=kvb.dtype, device=kvb.device
                )
                w_kv[: h * (nope + rope)] = w_k8
                w_kv[h * (nope + rope) :, :lora] = w_v8
                w_k8, w_v8 = _mxf8_op.quantize_weight(w_kv), None
                del w_kv
            elif kvb8 is not None:
                w_k8, w_v8 = _mxf8_op.quantize_weight(w_k8), _mxf8_op.quantize_weight(w_v8)
            qb8 = _fold_pow2(a.qb.t().view(h, nope + rope, -1), torch.full((h,), s_q)).view(
                h * (nope + rope), -1
            )
            layers.append((w_k8, w_v8, _mxf8_op.quantize_weight(qb8) if mxf8 else qb8))
            hs = _mha_pre_op.fp8_head_scales(s_q, s_k, s_v).to(kvb.device)
            ranges.append(
                (s_q, float(s_k.min()), float(s_k.max()), float(s_v.min()), float(s_v.max()))
            )
            self._attn[i].mha_hs = hs
            self._attn[i].mha_sq = s_q
        w_k8, w_v8, qb8 = layers[0]
        if kvb8 is not None:
            expand = kvb8.runners
        else:
            expand = {
                "k": _try_build(
                    "kv_b expansion K e4m3",
                    lambda: _projection_op.build_prefill("kv_b_k_e4m3", w_k8),
                ),
                "v": _try_build(
                    "kv_b expansion V e4m3",
                    lambda: _projection_op.build_prefill("kv_b_v512_e4m3", w_v8),
                ),
            }
        q8 = _try_build(
            "q_b e4m3",
            lambda: (
                _mxf8_op.build("q_b_e4m3", qb8)
                if mxf8
                else _projection_op.build_prefill("q_b_e4m3", qb8)
            ),
        )
        if not all(expand.values()) or q8 is None:
            for i in range(self.num_layers):
                self._attn[i].mha_hs = None
            return False
        for i, (w_k8, w_v8, qb8) in enumerate(layers):
            self._attn[i].kvb8 = (w_k8, w_v8)
            self._attn[i].qb8 = qb8
        self._mha_pre, self._kvb_expand, self._q8 = mha, expand, q8
        self._kvb8 = kvb8
        self._mha_fp8 = True
        if "o_proj" in self._proj_mxf8:
            mha_q8 = _try_build(
                "context MHA fp8 with the MXF8 O epilogue",
                lambda: _mha_pre_op.compile(dev, fp8=True, q8=True, kv_strided=merged),
            )
            rows_q8 = _try_build("act quant rows", lambda: _aq_rows_op.compile(dev))
            if mha_q8 is not None and rows_q8 is not None:
                self._mha_q8 = SimpleNamespace(mha=mha_q8, rows=rows_q8)

        def lg(x: float) -> int:
            return int(round(math.log2(x)))

        logger.info(
            "modeling_v2 GLM-5.2: context MHA fp8, q_b "
            f"{'MXF8' if mxf8 else 'bf16'}, K / V expansion "
            f"{('MXF8, one launch' if merged else 'MXF8') if kvb8 is not None else 'bf16 MMA'}, "
            "o_proj operand "
            f"{'from the MHA' if self._mha_q8 is not None else 'quantized'}; log2 scales over "
            f"layers q [{min(lg(r[0]) for r in ranges)}, {max(lg(r[0]) for r in ranges)}] "
            f"k [{min(lg(r[1]) for r in ranges)}, {max(lg(r[2]) for r in ranges)}] "
            f"v [{min(lg(r[3]) for r in ranges)}, {max(lg(r[4]) for r in ranges)}]"
        )
        return True

    def _build_kvb8(self, dev) -> SimpleNamespace | None:
        """The MXF8 K / V expansion's gather and its two GEMM runners (built against zero
        weights: the runners take each layer's weight per call), or None if a part fails."""
        gather = _try_build("MXF8 K/V gather", lambda: _kv_gather8_op.compile(dev))
        if gather is None:
            return None
        h, qk, v, lat, lora = self.heads, self.qk_dim, self.v_dim, self.lat_dim, self.kv_lora
        bf = dict(dtype=torch.bfloat16, device=torch.device("cuda", dev))
        runners = {
            "k": _try_build(
                "MXF8 K expansion",
                lambda: _mxf8_op.build(
                    "kv_b_k8", _mxf8_op.quantize_weight(torch.zeros(h * qk, lat, **bf))
                ),
            ),
            "v": _try_build(
                "MXF8 V expansion",
                lambda: _mxf8_op.build(
                    "kv_b_v8", _mxf8_op.quantize_weight(torch.zeros(h * v, lora, **bf))
                ),
            ),
        }
        if not all(runners.values()):
            return None
        # Both in one launch (`kv_b_kv8`); None keeps the two launches.
        merged = _try_build(
            "MXF8 K | V expansion in one launch",
            lambda: _mxf8_op.build(
                "kv_b_kv8", _mxf8_op.quantize_weight(torch.zeros(h * (qk + v), lat, **bf))
            ),
        )
        d = torch.device("cuda", dev)
        # The gather's pool / index / table operands on a step without a cached prefix: read
        # by nothing (no cached rows, new rows in order), but they must be valid tensors.
        idle = (
            torch.zeros(1, lat, dtype=torch.uint8, device=d),
            torch.zeros(1, dtype=torch.int64, device=d),
            torch.zeros(1, dtype=torch.int32, device=d),
            torch.zeros(1, 1, dtype=torch.int32, device=d),
        )
        return SimpleNamespace(gather=gather, runners=runners, merged=merged, idle=idle)

    def _mha_fp8_scales(
        self, i: int, q_extra: float, kv_extra: float = 1.0
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Layer `i`'s power-of-two dequantization scales `(s_q, s_k [H], s_v [H])` of the fp8
        context attention, from strict bounds so no context row can exceed e4m3's 448: every
        operand is a linear map of an RMS-normalized vector `n * w` with `|n|_2 <= sqrt(dim)`,
        so by Cauchy-Schwarz `|(W (n * w))_d| <= |W_d * w|_2 sqrt(dim)`. Q = q_b(q_a norm) per
        row (nope, and for rope the norm of each rotated pair); K = W_UK c_kv (nope) and the
        roped k_pe of kv_a(norm1(h)); V = W_UV c_kv. `s = 2^ceil(log2(bound * margin / 448))`,
        per tensor for Q, per head for K and V."""
        a, f32 = self._attn[i], torch.float32
        h, nope, rope, v = self.heads, self.nope, self.rope_dim, self.v_dim
        q_lora, kv_lora = self.q_lora, self.kv_lora
        wqb = a.qb.t().detach().to(f32).view(h, nope + rope, q_lora)
        qb = (wqb * a.q_norm.detach().to(f32)).norm(dim=-1) * math.sqrt(q_lora)
        q_pair = torch.sqrt(qb[:, nope::2] ** 2 + qb[:, nope + 1 :: 2] ** 2)
        q_bound = torch.maximum(qb[:, :nope].amax(), q_pair.amax())
        kv = self.w[f"l{i}_kvb"].detach().to(f32)
        wkvn = a.kv_norm.detach().to(f32)
        k_nope = (kv[: h * nope].view(h, nope, kv_lora) * wkvn).norm(dim=-1)
        v_all = (kv[h * nope :].view(h, v, kv_lora) * wkvn).norm(dim=-1)
        pe_rows = a.qkva.t().detach().to(f32)[q_lora + kv_lora : q_lora + kv_lora + rope]
        pe = (pe_rows * a.norm1.detach().to(f32)).norm(dim=-1) * math.sqrt(pe_rows.shape[1])
        pe_pair = torch.sqrt(pe[0::2] ** 2 + pe[1::2] ** 2).amax()
        k_bound = torch.maximum(k_nope.amax(-1) * math.sqrt(kv_lora), pe_pair)
        v_bound = v_all.amax(-1) * math.sqrt(kv_lora)
        s_q = float(_pow2_ceil(q_bound * _MHA_FP8_MARGIN * q_extra / 448.0))
        return (
            s_q,
            _pow2_ceil(k_bound * _MHA_FP8_MARGIN * kv_extra / 448.0),
            _pow2_ceil(v_bound * _MHA_FP8_MARGIN * kv_extra / 448.0),
        )

    def _derive_indexer(self, i: int) -> SimpleNamespace | None:
        if f"l{i}_ix_qb" not in self.w:
            return None
        w = self.w
        return SimpleNamespace(
            fused=w[f"l{i}_ix_fused"].t(),
            fused_bf16=w[f"l{i}_ix_fused_bf16"],
            qb=w[f"l{i}_ix_qb"].t(),
            kn_w=w[f"l{i}_ix_kn_w"],
            kn_b=w[f"l{i}_ix_kn_b"],
        )

    def _stack_ix_fold_weights(self, layers) -> dict[int, torch.Tensor]:
        """For every layer with an indexer: its qkva weight and the indexer's bf16 key /
        head-weight rows (`ix_fused_bf16`) moved into one `[2848, 6144]` buffer
        (`_ix_fold_op.stack_weight`: qkva, 64 zero rows, the indexer rows), the two
        parameters re-pointed at their rows of it -- the same bytes in the same row-major
        layout, so every other reader is unchanged, and the old storage is freed. Returns the
        buffers by layer (`Gemm.qkva_ix_fold`'s weight)."""
        stacked = {}
        for i in layers:
            if f"l{i}_ix_fused_bf16" not in self.w:
                continue
            qkva, ix = self.w[f"l{i}_qkva"], self.w[f"l{i}_ix_fused_bf16"]
            buf = _ix_fold_op.stack_weight(qkva.detach(), ix.detach())
            qkva.data = buf[: int(qkva.shape[0])]
            ix.data = buf[_ix_fold_op.IX_ROW0 :]
            stacked[i] = buf
        return stacked

    def _stack_qbix_weights(self, layers) -> dict[int, torch.Tensor]:
        """For every layer with an indexer: its q_b weight and the indexer's wq_b moved into
        one `[2, 4096, 2048]` buffer (`_qbix_op.stack_weight`), the two parameters re-pointed
        at its two halves (the same bytes and row-major layout: every other reader, the MXF8
        copies included, is unchanged; the old storage is freed). Returns the buffers by
        layer (`Gemm.qb_ixqb`'s weight)."""
        stacked = {}
        for i in layers:
            if f"l{i}_ix_qb" not in self.w:
                continue
            qb, ixqb = self.w[f"l{i}_qb"], self.w[f"l{i}_ix_qb"]
            buf = _qbix_op.stack_weight(qb.detach(), ixqb.detach())
            qb.data = buf[0]
            ixqb.data = buf[1]
            stacked[i] = buf
        return stacked

    def _build_qbix(self, dev: int) -> dict:
        """The q_b | wq_b cells (`_qbix_op.build_all`), kept only where they match both
        projections' MTP cells row for row: then they serve exactly the rows those cells would
        (`Gemm.wide_cell`), with their bytes. Empty (the two GEMMs) otherwise or on a failed
        build."""
        cells = _try_build("q_b | wq_b fold", lambda: _qbix_op.build_all(dev))
        want = set(self._proj_mtp.get("q_b", {}))
        if not cells or set(cells) != want or set(self._proj_mtp.get("ix_qb", {})) != want:
            if cells:
                logger.warning(
                    "modeling_v2 GLM-5.2: q_b | wq_b fold cells do not match the MTP cells; "
                    "the two GEMMs stay separate"
                )
            return {}
        return cells

    def _build_qkva_ix_fold(self, dev: int) -> dict:
        """The fold cells (`_ix_fold_op.build_all`), kept only where they match the qkva
        projection's MTP cells row for row: the fold then serves exactly the rows the qkva
        cells would (`Gemm.wide_cell`), with their bytes. Empty (the two GEMMs) otherwise or if
        the build fails."""
        cells = _try_build("qkva indexer fold", lambda: _ix_fold_op.build_all(dev))
        if not cells or set(cells) != set(self._proj_mtp.get("qkva", {})):
            if cells:
                logger.warning(
                    "modeling_v2 GLM-5.2: qkva indexer fold cells do not match the qkva MTP "
                    "cells; the two GEMMs stay separate"
                )
            return {}
        return cells

    def _derive_moe(self, i: int) -> SimpleNamespace:
        """The routed experts' operands, with the per-tensor scalars folded into the kernels'
        alphas over this rank's window.

        modelopt stores reciprocals: `input_scale = 1/g`. Every routed expert's input_scale
        is 1.0 on this checkpoint, and one quantization of the hidden states feeds every
        expert, so a per-expert value could not be honoured: asserted, not assumed.
        """
        w = self.w
        window = slice(self.expert_offset, self.expert_offset + self.local_experts)
        isc1, isc2 = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc2"]
        if self.dec_replicas:
            # The alphas follow the stacks' slots: this rank's window, then the replicas.
            window = torch.cat(
                [
                    torch.arange(self.expert_offset, self.expert_offset + self.local_experts),
                    torch.tensor(self.dec_hot[i], dtype=torch.long),
                ]
            ).to(isc1.device)
        assert torch.equal(isc1, w[f"l{i}_e_isc1_up"]), f"layer {i}: gate/up input_scale differ"
        assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
            f"layer {i}: gate/up weight_scale_2 differ"
        )
        assert bool((isc1 == 1.0).all()) and bool((isc2 == 1.0).all()), (
            f"layer {i}: routed-expert input_scale is not uniformly 1.0"
        )
        alpha1 = (isc1[window] * w[f"l{i}_e_ws2_1"][window]).contiguous()
        return SimpleNamespace(
            router=w[f"l{i}_router"],
            router_t=w[f"l{i}_router"].t(),
            router_bias=w[f"l{i}_router_bias"],
            fc1_w=w[f"l{i}_fc1_w"],
            fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
            fc2_w=w[f"l{i}_fc2_w"],
            fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
            alpha1=alpha1,
            # `o1[:1] / alpha1[:1]` with `o1 = alpha1 / isc2`: weights only, so made here
            # rather than allocated inside every capture.
            norm_const=((alpha1 / isc2[window])[:1] / alpha1[:1]).contiguous(),
            alpha2=(isc2[window] * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            g1=(1.0 / isc1[:1]).contiguous(),
        )

    def _derive_mtp(self) -> dict:
        """The MTP module's own operands. Its attention and indexer are a trunk layer's at
        index `num_layers`; its experts are bf16."""
        w = self.w
        j = self.num_layers
        k = _mtp_eh.LOCAL_INPUT_SIZE
        # The front end's TP4 K-split: this rank owns columns [r*K, (r+1)*K) of the 12288-wide
        # concat. Held here, outside `w`, so no parameter appears that the weight table did
        # not declare.
        eh_local = nn.Parameter(
            w[f"l{j}_eh"][:, self.tp_rank * k : (self.tp_rank + 1) * k].contiguous(),
            requires_grad=False,
        )
        return {
            "enorm": w[f"l{j}_enorm"],
            "hnorm": w[f"l{j}_hnorm"],
            "eh": w[f"l{j}_eh"].t(),
            "eh_local": eh_local,
            "ehproj": _mtp_eh.build(eh_local),
            # Ranks 0-1 norm halves of e, ranks 2-3 halves of h; the offset is compiled in.
            "norm_shard": compile_mtp_norm_shard((self.tp_rank % 2) * k),
            "head_norm": w[f"l{j}_head_norm"],
            "router": w[f"l{j}_router"],
            "router_t": w[f"l{j}_router"].t(),
            "router_bias": w[f"l{j}_router_bias"],
            "fc1": w[f"l{j}_fc1"],
            "fc2": w[f"l{j}_fc2"],
        }

    def _mtp_moe_fp4(self) -> SimpleNamespace:
        """The MTP module's routed experts (bf16 in this checkpoint: modelopt excludes
        `model.layers.78`) quantized once to NVFP4 in the trunk's grouped-GEMM layout, so the
        MTP layer runs the trunk's routed path (router, fused top-k, FC1, FC2) instead of its
        bf16 pair: at decode 307 + 184 us per call on the bf16 GEMMs against about 105 + 44 us
        for a trunk layer. FC1 is stored in the trunk's row order already (64-row blocks
        alternating up and gate), so each expert quantizes as stored, up and gate sharing
        one global scale `448 * 6 / amax` (modelopt's `1 / weight_scale_2`), as a trunk
        expert's do. Input scales are 1.0 at both GEMMs, as the trunk's are, so `g1` and
        `norm_const` are 1 and the alphas are the `weight_scale_2`s. The bf16 stacks (~4.8 GB
        per rank) are released and handed back to the driver: the engine sizes its KV pool
        from the device's free memory. Draft-only numerics: they move MTP acceptance, never
        the verified tokens.

        Idempotent: the engine may run `post_load_weights` more than once, and the bf16 stacks
        are gone after the first, so a later call returns the first call's operands."""
        built = getattr(self, "_mtp_fp4_built", None)
        if built is not None:
            return built
        mw, j = self._mtp, self.num_layers

        def quant(stack: torch.Tensor, kblk: bool = False):
            e, n, k = stack.shape
            dev = stack.device
            qw = torch.empty((e, n, k // 2), dtype=torch.uint8, device=dev)
            qs = torch.empty(
                (e, n // _weights.SF_ATOM_MN, k // _weights.SF_ATOM_K, _weights.SF_ATOM),
                dtype=torch.uint8,
                device=dev,
            )
            ws2 = torch.empty((e,), dtype=torch.float32, device=dev)
            for x in range(e):
                gs = (448.0 * 6.0 / stack[x].abs().amax().float()).reshape(1)
                q, sf = fp4_quantize(stack[x].contiguous(), gs, 16, False, False)
                q = q.view(n, k // 2)
                # In the layout the trunk's FC1 / FC2 kernels are compiled for (`fc1_kblk` /
                # `fc2_kblk`).
                qw[x] = _fc1_kblock.to_kblock(q) if kblk else q
                qs[x] = _weights._sf_atoms(sf.view(torch.uint8).view(n, k // 16))
                ws2[x] = 1.0 / gs[0]
            return qw, qs.view(torch.float8_e4m3fn), ws2

        q1, s1, a1 = quant(mw["fc1"], self.fc1_kblk)
        q2, s2, a2 = quant(mw["fc2"], self.fc2_kblk)
        one = torch.ones(1, dtype=torch.float32, device=q1.device)
        for key in (f"l{j}_fc1", f"l{j}_fc2"):
            prm = self.w[key]
            prm.data = torch.empty(0, dtype=prm.dtype, device=prm.device)
        mw["fc1"] = mw["fc2"] = None
        torch.cuda.empty_cache()
        self._mtp_fp4_built = SimpleNamespace(
            router=mw["router"],
            router_t=mw["router_t"],
            router_bias=mw["router_bias"],
            fc1_w=q1,
            fc1_s=s1,
            fc2_w=q2,
            fc2_s=s2,
            alpha1=a1,
            norm_const=one,
            alpha2=a2,
            g1=one.clone(),
        )
        return self._mtp_fp4_built

    def post_load_weights(self) -> None:
        """Derive the operands, then build and compile every specialised kernel.

        Everything compiles here because a first compile inside a CUDA-graph capture breaks
        the graph; what needs an engine value compiles in `_derive_first_step`. Named for the
        shell hook that drives it.
        """
        dev = torch.cuda.current_device()
        layers = range(self.num_layers + (1 if self.mtp_enabled else 0))
        # Before any view of the two weights is taken: an MTP3 / MTP5 engine's indexer layers
        # hold their qkva and indexer key / head-weight rows in one buffer (`Gemm.qkva_ix_fold`).
        if self.wide_cells and self._ix_fold_on:
            self._ix_fold_w = self._stack_ix_fold_weights(layers)
        # The same for q_b and the indexer's wq_b (`Gemm.qb_ixqb`).
        if self.wide_cells and self._qbix_on:
            self._qbix_w = self._stack_qbix_weights(layers)
        self._attn = [self._derive_attention(i) for i in layers]
        self._idx = [self._derive_indexer(i) for i in layers]
        self._mlp = [(self.w[f"l{i}_mlp_gu"].t(), self.w[f"l{i}_mlp_dn"].t()) for i in layers]
        self._moe = [
            None if i < self.dense_layers else self._derive_moe(i) for i in range(self.num_layers)
        ]
        # Each layer's closing collective norms with the next layer's input norm; the last
        # layer's with the final norm.
        self._next_norm = [self.w[f"l{i + 1}_norm1"] for i in range(self.num_layers - 1)] + [
            self.w["final_norm"]
        ]
        self._mtp_fp4 = None
        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
            self._mtp_fp4 = _try_build("MTP experts as NVFP4", self._mtp_moe_fp4)

        first_idx = next(i for i in range(self.num_layers) if self._idx[i] is not None)
        # One runner per (projection, compiled row count), built against one layer's row-major
        # weight, and one prefill GEMM per projection for the context row counts above
        # _DECODE_MAX_ROWS; `Gemm.project` sends a row count with neither to cuBLAS.
        proj_weights = {
            "qkva": self._attn[0].qkva,
            "q_b": self._attn[0].qb,
            "o_proj": self._attn[0].o,
            "ix_qb": self._idx[first_idx].qb,
            "ix_fused": self._idx[first_idx].fused,
            "router": self._moe[self.dense_layers].router_t,
            "dense_gate_up": self._mlp[0][0],
            "dense_down": self._mlp[0][1],
            "shared_gate_up": self._mlp[self.dense_layers][0],
            "shared_down": self._mlp[self.dense_layers][1],
        }
        self._proj = {
            name: _projection_op.build_all(name, w_colmajor.t())
            for name, w_colmajor in proj_weights.items()
        }
        if self._ctx_host_diet:
            # A mixed step's generation rows take q_b's decode cells eagerly (`Gemm.project_tail`);
            # the persistent cells among them (up to 16 rows) launch through a TVM-FFI twin of
            # the same kernel. Captured decode graphs keep the cells' own launchers.
            qb_w = proj_weights["q_b"].t()
            self._proj["q_b"] = {
                m: _projection_op.with_eager_twin(r, qb_w) if m <= _DECODE_MAX_ROWS else r
                for m, r in self._proj["q_b"].items()
            }
        # MTP3 / MTP5: the decode cells above 64 rows, taken on `Phase.wide` steps only (a
        # context step of the same row count keeps the paths below); empty for one draft token.
        wide = self.wide_cells
        self._proj_mtp = {
            name: _projection_op.build_all(name, w_colmajor.t(), _projection_op.MTP_ROWS[name])
            for name, w_colmajor in proj_weights.items()
            if wide and _projection_op.MTP_ROWS[name]
        }
        self._qkva_ix_mtp = self._build_qkva_ix_fold(dev) if self._ix_fold_w else {}
        self._qbix_mtp = self._build_qbix(dev) if self._qbix_w else {}
        if self.mtp_enabled:
            proj_weights["mtp_eh"] = self._mtp["eh"]
        self._proj_pre = _projection_op.build_prefill_all(
            {
                name: w_colmajor.t()
                for name, w_colmajor in proj_weights.items()
                if name in _projection_op.PREFILL_ENTRIES
            }
        )
        self._build_proj_mxf8()
        self._qkv_a_norm = compile_qkv_a_rmsnorm(dev, self.eps)
        # The q_a norm that also writes q_a's MXF8 operand for the q_b GEMM reading it.
        self._qkv_a_norm8 = _try_build(
            "q_a norm with the MXF8 q copy", lambda: _qkv_a_norm8_op.compile(dev, self.eps)
        )
        self._dense_swiglu = compile_dense_swiglu()
        # The shared expert's gate/up rows interleaved for the fused SwiGLU kernel, one copy per
        # layer: one copy for every layer runs every layer against one layer's experts.
        self._shared_gu_il = [
            None
            if i < self.dense_layers
            else _shared_gu_swiglu_op.prepare_weight(self._mlp[i][0].t())
            for i in layers
        ]
        self._shared_gu_swiglu = _shared_gu_swiglu_op.build_all(
            self._shared_gu_il[self.dense_layers]
        )
        self._shared_gu_swiglu_mtp = (
            _shared_gu_swiglu_op.build_all(
                self._shared_gu_il[self.dense_layers], _shared_gu_swiglu_op.MTP_ROWS
            )
            if wide
            else {}
        )
        # An MTP3 / MTP5 engine's decode row counts of 1..3 sequences without a cell
        # (`_SHARED_FUSED_UP_SEQS`), each its own compile; taken where `Gemm.cell_up` holds.
        up_rows = sorted(
            {
                (self.max_draft_len + 1) * seqs
                for seqs in range(1, _SHARED_FUSED_UP_SEQS + 1)
                if (self.max_draft_len + 1) * seqs <= _DECODE_MAX_ROWS
            }
            - set(self._shared_gu_swiglu)
        )
        self._shared_gu_swiglu_up = (
            _shared_gu_swiglu_op.build_all(self._shared_gu_il[self.dense_layers], tuple(up_rows))
            if wide and self._dec_cell_up
            else {}
        )
        self._embedding = build_embedding(self.w["embed"])
        # One compiled object per row-count group serves (2, 4, 8) or (16, 32, 48, 64); the
        # prefill kernel every other row count up to 8192, the 1..63 generation rows of a
        # mixed step included.
        self._absorb: dict[str, dict] = {"k_b": {}, "v_b": {}}
        for kind, runners in self._absorb.items():
            for group in _absorbed_bmm_op.GROUPS:
                runner = _absorbed_bmm_op.build(kind, group, dev)
                runners.update(dict.fromkeys(runner.rows, runner))
        self._absorb_pre = {
            kind: _absorbed_bmm_op.build_prefill(kind, dev) for kind in self._absorb
        }
        # The MTP3 / MTP5 decode row counts (128, 192), plain bf16 cells, `Phase.wide` only.
        self._absorb_mtp: dict[str, dict] = {"k_b": {}, "v_b": {}}
        for kind, runners in self._absorb_mtp.items():
            for group in _absorbed_bmm_op.MTP_GROUPS if wide else ():
                runner = _absorbed_bmm_op.build(kind, group, dev)
                runners.update(dict.fromkeys(runner.rows, runner))
        # The context step's K-B with the e4m3 q-base epilogue: on the context rope kernel's
        # path it writes the FMHA's fp8 query bytes itself. None keeps the bf16 K-B there.
        self._kb_fp8_pre = _try_build(
            "fp8 q-base K-B", lambda: _absorbed_bmm_op.build_prefill_fp8_q_base(dev)
        )
        # The decode cells' K-B writing the e4m3 q-base and releasing the rope kernel at
        # once, so the two run concurrently (`Attn.gen_query`); empty keeps the bf16 K-B.
        self._kb8: dict = {}
        for group in _absorbed_bmm_op.GROUPS:
            runner = _try_build(
                f"decode fp8 q-base K-B {group}",
                lambda group=group: _absorbed_bmm_op.build_decode_fp8_q_base(group, dev),
            )
            if runner is None:
                self._kb8 = {}
                break
            self._kb8.update(dict.fromkeys(runner.rows, runner))
        # The same at the MTP3 / MTP5 decode rows (128 / 192, and the rows between on the next
        # one up), `Phase.wide` only: the rope kernel of widths 4 / 6 then runs beside it.
        self._kb8_mtp = {}
        for group in _absorbed_bmm_op.MTP_GROUPS if wide and self._kb8 else ():
            runner = _try_build(
                f"decode fp8 q-base K-B {group}",
                lambda group=group: _absorbed_bmm_op.build_decode_fp8_q_base(group, dev),
            )
            if runner is not None:
                self._kb8_mtp.update(dict.fromkeys(runner.rows, runner))
        self._ix_pre_split = _ix_pre_split_op.compile_all(dev)
        self._ix_pre_split_mtp = (
            _ix_pre_split_op.compile_all(dev, _ix_pre_split_op.MTP_ROWS) if wide else {}
        )
        # The same kernel on a symbolic row count, for the decode row counts in 65..191
        # without a handle above (`Indexer.project`). None keeps them on the prefill split.
        self._ix_pre_split_rag = (
            _try_build("indexer split at runtime M", lambda: _ix_pre_split_op.compile_ragged(dev))
            if wide
            else None
        )
        # Runners re-pinned from the cells built above, if a load ran before (`wide_cell`).
        self.op.gemm._wide_rag = {}
        # The same split at the context row counts, on the prefill GEMM against the decode
        # handles' bf16 weight; every indexer layer shares it. None keeps the fp32 projection.
        try:
            self._ix_pre_split_pre = _ix_pre_split_op.build_prefill(self._idx[first_idx].fused_bf16)
        except Exception as e:  # noqa: BLE001 -- the context rows take the fp32 projection
            logger.warning(
                f"modeling_v2 GLM-5.2: indexer prefill split build failed ({e}); using fp32"
            )
            self._ix_pre_split_pre = None
        self._ix_qside_compiled = build_ix_q_side(dev, is_neox=self.indexer_is_neox)
        # TRTLLM_GLM52_CTX_IX_NOSPLIT (default 1; 0 restores the split): a context step's
        # indexer projection is not split into bf16 keys + packed head weights; the K write and
        # the q side read their columns of the fp32 [T, 160] output in place (pitch-160
        # launchers), the K write rounding the keys to bf16 on load as the split does. Same
        # bytes, one launch fewer per indexer layer.
        self._ix_nosplit = (
            self._ix_pre_split_pre is not None
            and os.environ.get("TRTLLM_GLM52_CTX_IX_NOSPLIT", "1") == "1"
        )
        self._ix_qside_compiled_p160 = (
            build_ix_q_side(dev, is_neox=self.indexer_is_neox, wq_pitch=_IX_PROJ_COLS)
            if self._ix_nosplit
            else None
        )
        self._ix_topk_pre_compiled = build_ix_topk_prefill(
            dev, top_k=self.idx_topk, max_columns=_MOE_MAX_T
        )
        # The MLA rope kernel is keyed by the block-table width, an engine value, and the MTP
        # width: built per pair in `_derive_first_step` (the trunk's) and at the first eager use
        # of another.
        self._mla_rope_compiled: dict[tuple[int, int], _mla_rope_op.CompiledMlaRopeGeneration] = {}
        # The context kernel in both modes and its two scale arrays, built here because a
        # first compile inside a CUDA-graph capture is a broken graph.
        self._fmha_pre_compiled = _fmha_pre_op.compile(device_index=dev)
        self._fmha_pre_dc_compiled = _fmha_pre_op.compile(device_index=dev, dense_causal=True)
        self._fmha_bmm1 = torch.tensor(_FMHA_BMM1_SCALE, dtype=torch.float32, device=dev)
        self._fmha_bmm2 = torch.tensor(_FMHA_BMM2_SCALE, dtype=torch.float32, device=dev)
        self._build_context_mha(dev)
        self._build_dec_mxf8(dev)
        # A CUDA stream created inside a graph capture is not capturable.
        self._side_stream = torch.cuda.Stream(device=dev)
        # A mixed context step's generation-row attention on its own stream beside the context
        # rows' MHA (`PrefillTarget._layer_attention`); on by default,
        # TRTLLM_GLM52_CTX_GEN_SIDE=0 restores the serial order.
        self._gen_stream = (
            torch.cuda.Stream(device=dev)
            if os.environ.get("TRTLLM_GLM52_CTX_GEN_SIDE", "1") == "1"
            else None
        )
        # TRTLLM_GLM52_CTX_GEN_SIDE_QUANT (default 1; 0 restores the main-stream quantize): on
        # that stream, the generation rows' MXF8 quantize for o_proj runs right after their
        # attention instead of on the main stream after the join, off the critical path
        # whenever the generation side finishes first. Same kernel and bytes: the context MHA's
        # epilogue writes the rows below `tc` and their scale bytes, the quantize the rows from
        # `tc` and the padding scale bytes, all byte stores, so the two never write the same
        # byte.
        self._gen_side_quant = (
            self._gen_stream is not None
            and os.environ.get("TRTLLM_GLM52_CTX_GEN_SIDE_QUANT", "1") == "1"
        )
        self._build_moe(dev)
        # The peer buffers are sized for the whole token budget, so every collective shares
        # one workspace. `wide`: MTP3 / MTP5 decode steps (above 64 rows) open their MoE
        # layers on the two-shot build of the AR + norm + NVFP4 (`QUANT_TWOSHOT_ABOVE`); an
        # MTP1 or non-speculative engine builds and runs exactly what it did.
        self._lamport = build_lamport_workspace(
            self.w["embed"].shape[1],
            self.eps,
            _MOE_MAX_T,
            self.tp_rank,
            self.tp_size,
            wide=self.wide_cells,
        )
        # The row ops the fused kernels leave over: layer 0's input norm, the MTP module's
        # norms and output adds, the logits rows' lookup, the forward's copies, routing and
        # SwiGLU at row counts the fused front end has no kernel for, and the greedy picks.
        # Each is independent: one that fails to build keeps its call sites on the stock op.
        self._rows = SimpleNamespace(
            norm=_try_build("hidden RMSNorm", _hidden_norm_op.compile),
            add=_try_build("hidden add", _hidden_add_op.compile),
            # The MTP module's residual add + head norm in one launch (`Rows.add_rmsnorm`): only an
            # engine drafting more than one token returns the module's output normed.
            add_norm=(
                _try_build("hidden add + RMSNorm", _hidden_add_norm_op.compile)
                if self.max_draft_len > 1
                else None
            ),
            gather=_try_build("hidden row gather", _row_gather_op.compile),
            topk=_try_build("sigmoid top-k", _topk_rows_op.compile),
            swiglu=_try_build("SwiGLU rows", _swiglu_rows_op.compile),
            copy=_try_build("copy/convert", _copy_op.compile),
            argmax=_try_build("argmax", _argmax_op.compile),
        )
        # The MTP module's context steps run attention-to-output on the draft head's rows only
        # (`PrefillTarget._mtp_last_plan`): the row gather / scatter they need.
        self._rows_sel = (
            _try_build("row gather / scatter", _rows_sel_op.compile) if self.mtp_enabled else None
        )
        if self.tp_rank == 0:
            built = " ".join(f"{k}={v is not None}" for k, v in vars(self._rows).items())
            logger.info(
                f"modeling_v2 GLM-5.2 row ops built: {built} "
                f"logits_gather={self._lamport.gather is not None}"
            )
        if self.max_draft_len > 1:
            self._warm_draft_rows()

        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        self._wide = WideDecodeTarget(self)

    def _build_moe(self, dev) -> None:
        """The routed experts' buffers and compiled kernels, trunk and MTP.

        Every operand handed to these kernels lives in a buffer allocated here, in two slots
        alternating per layer. The engine allocates its tensors inside the CUDA-graph
        capture; handed to these kernels directly, replays routed as though every id were
        zero. And the kernels launch with PDL, so a layer's kernel can start while the
        previous layer's is still reading: one slot measured every layer but the last at
        cosine 0.64-0.93. The NVFP4 input (`xq` / `xsf`) is one buffer, written by the
        closing collective or the quantize launch and read by FC1 in the same layer.

        A decode kernel that fails to build raises: those steps have no stock path for the
        packed weights. A prefill kernel that fails to build is None, and its call site
        takes the stock op (the decode grouped GEMMs for the prefill pair).
        """
        hidden, inter, experts = self.w["embed"].shape[1], self.moe_inter, self.local_experts
        rows = _MOE_MAX_T
        bf16 = {"dtype": torch.bfloat16, "device": dev}
        f32 = {"dtype": torch.float32, "device": dev}
        u8 = {"dtype": torch.uint8, "device": dev}

        def optional(build, what: str = "MoE prefill kernel"):
            try:
                return build()
            except Exception as e:  # noqa: BLE001 -- that launch takes its stock op at the call site
                logger.warning(f"modeling_v2 GLM-5.2: {what} build failed ({e}); using stock op")
                return None

        def slot(mtp: bool) -> SimpleNamespace:
            counts, barrier = _moe_routing_op.allocate_workspace(dev)
            widths = _MOE_DECODE_WIDTHS if mtp else range(1, _DECODE_MAX_ROWS + 1)
            ns = SimpleNamespace(
                # The routed output is returned to the caller, so it alternates too: one
                # buffer would be overwritten by the next layer before the caller used it.
                out=torch.zeros((rows, hidden), **bf16),
                x=torch.zeros((rows, hidden), **bf16),
                w=torch.zeros((rows, _MOE_TOP_K), **f32),
                sel=torch.empty(rows * _MOE_TOP_K, dtype=torch.int32, device=dev),
                # Context replicas: one more partial tile per replica slot (trunk only).
                maps=_moe_routing_op.allocate_maps(
                    rows, dev, replicas=0 if mtp else self.ctx_replicas
                ),
                counts=counts,
                barrier=barrier,
                router_out=torch.zeros((_DECODE_MAX_ROWS, self.num_experts), **f32),
                router_partials=torch.zeros(
                    (_DECODE_MAX_ROWS, _moe_router_op.SPLIT_K, self.num_experts), **f32
                ),
                # The fused top-k checks its outputs against the exact row count: one set
                # per width; with decode expert replicas the push widths' sets hold the
                # replica top-k's `64 + replicas` tiles.
                topk_out={
                    w: _moe_topk_rep_op.allocate_outputs(w, self.dec_replicas, dev)
                    if not mtp and self.dec_replicas and _moe_topk_rep_op.uses_rep_path(w)
                    else _moe_topk_op.allocate_decode_outputs(w, dev)
                    for w in widths
                },
            )
            if mtp:
                ns.c = _mtp_experts_op.allocate_c(rows, inter, dev)
                return ns
            ns.c = _moe_experts_op.allocate_decode_c(rows, inter, dev)
            ns.sfc = _moe_experts_op.allocate_sfc(rows, inter, dev, self.ctx_replicas)
            ns.router_x = torch.zeros((_DECODE_MAX_ROWS, hidden), **bf16)
            if self.wide_cells:
                # The wide decode router's `[rows, 4, 256]` partials (`wide_cells` only).
                ns.router_wide_out = torch.zeros(
                    _moe_router_op.output_shape(_moe_router_op.WIDE_MAX_ROWS, wide=True), **f32
                )
            ns.logits = torch.zeros((rows, 1, self.num_experts), **f32)
            ns.topk_scratch = _moe_topk_op.allocate_prefill_scratch(dev)
            if self.ctx_replicas:
                ns.topk_rep_scratch = _moe_topk_pre_rep_op.allocate_scratch(self.ctx_replicas, dev)
            return ns

        self._moe_slots = [slot(False) for _ in range(2)]
        # One byte per element, as the kernels' entries allocate: cute counts the shape in
        # elements and reads half as many bytes.
        self._moe_xq = torch.zeros((rows, hidden // 2), **u8)
        self._moe_xsf = torch.zeros((rows, hidden // _MOE_SF_VEC), **u8)
        # The prefill shared expert's SwiGLU output. One is enough: up and down run back to
        # back on the shared branch's stream, which forks again only after this layer's reduce.
        self._moe_mid = torch.zeros((rows, _SHARED_INTER), **bf16)
        s0 = self._moe_slots[0]
        k = SimpleNamespace()
        # One routing function per schedule bucket any step reaches, shared with the MTP
        # layer: a bucket compiled on first use would compile inside a step.
        k.routing = _moe_routing_op.build(_moe_routing_op.ROW_COUNTS, dev)
        k.experts = _moe_experts_op.build_decode(
            c=s0.c,
            sfc=s0.sfc,
            maps=s0.maps,
            xq=self._moe_xq,
            xsf=self._moe_xsf,
            out=s0.out,
            hidden=hidden,
            inter=inter,
            # The decode expert replicas' slots follow this rank's 64 (0 when off).
            experts=experts + self.dec_replicas,
            max_rows=rows,
            fc1_kblk=self.fc1_kblk,
            fc2_kblk=self.fc2_kblk,
            # One FC2 tile counter per MoE call site: the trunk layers and the MTP layer.
            dyn_slots=self.num_layers + 1,
        )
        k.pair = optional(
            lambda: _moe_experts_op.build_prefill_pair(
                hidden, inter, experts, dev, fc1_kblk=self.fc1_kblk, fc2_kblk=self.fc2_kblk
            )
        )
        if k.pair is not None:
            for sl in self._moe_slots:
                sl.pc = _moe_experts_op.allocate_pair_c(rows, inter, dev, self.ctx_replicas)
        # Context replicas: the replica top-k and the prefill pair on the `64 + R` deep
        # stacks, both or neither (the replica maps address slots past 64).
        k.topk_pre_rep = k.pair_rep = None
        self._ctx_hot_ids = {}
        if self.ctx_replicas and k.pair is not None:
            rep = optional(
                lambda: (
                    _moe_topk_pre_rep_op.build_prefill(self.ctx_replicas, dev),
                    _moe_experts_op.build_prefill_pair(
                        hidden,
                        inter,
                        experts + self.ctx_replicas,
                        dev,
                        fc1_kblk=self.fc1_kblk,
                        fc2_kblk=self.fc2_kblk,
                    ),
                )
            )
            k.topk_pre_rep, k.pair_rep = rep if rep is not None else (None, None)
            self._ctx_hot_ids = {
                i: torch.tensor(ids[: self.ctx_replicas], dtype=torch.int32, device=dev)
                for i, ids in self.dec_hot.items()
                if len(ids) >= self.ctx_replicas
            }
        # The maps the replica top-k wrote last (`Moe.topk`); `Moe.experts` runs them on the
        # replica pair.
        self.op.moe._rep_maps = None
        k.router_pre = optional(lambda: _moe_router_op.build_prefill(dev))
        k.topk_pre = optional(lambda: _moe_topk_op.build_prefill(dev))
        k.quant_pre = optional(lambda: _moe_quant_op.build_prefill(dev))
        k.zero_pre = optional(lambda: _moe_zero_op.build_prefill(dev))
        k.merge = optional(lambda: _moe_merge_op.build(dev))
        # The shared expert's prefill pair, both or neither: up writes the SwiGLU output down
        # reads. Built against the first MoE layer's weights; each call passes its own.
        shared = optional(
            lambda: (
                _shared_gu_swiglu_op.build_prefill(self._shared_gu_il[self.dense_layers]),
                _projection_op.build_shared_down_prefill(self._mlp[self.dense_layers][1].t()),
            )
        )
        k.shared_up_pre, k.shared_down_pre = shared if shared is not None else (None, None)
        k.quant = {w: _moe_quant_op.build_decode(w, dev) for w in _MOE_DECODE_WIDTHS}
        k.zero = {w: _moe_zero_op.build_decode(w, dev) for w in _MOE_DECODE_WIDTHS}
        # The fused top-k releases FC1 early only at a power of two: other widths are padded
        # and run the routing kernel next, which does not wait for the ids; released early it
        # reads them mid-write (a 3-request step hung that way).
        # The decode L2 prefetch of the routed FC1 weights: forked at a decode MoE layer's
        # o_proj end, ranked by the counts the same layer's previous decode step's top-k wrote
        # into its row of `hist` (one row per MoE call site, the MTP layer's included).
        self._pf = None
        pf = (
            _try_build("decode MoE L2 prefetch", lambda: _l2pf_op.compile(dev))
            if self._moe_l2_prefetch
            else None
        )
        if pf is not None:
            self._pf = SimpleNamespace(
                compiled=pf,
                hist=torch.zeros((self.num_layers + 1, 64), dtype=torch.int32, device=dev),
                stream=torch.cuda.Stream(device=dev),
                pending=None,
            )
        k.topk = {
            w: _moe_topk_op.build_decode(
                w,
                _moe_router_op.decode_splits(w),
                dev,
                signal_gg1=(w & (w - 1)) == 0,
                expert_hist=self._pf is not None and _moe_topk_op.uses_push_path(w),
            )
            for w in range(1, _DECODE_MAX_ROWS + 1)
        }
        # Decode expert replicas: the push widths select on the replica top-k, with each
        # trunk layer's hot ids (the MTP layer passes all -1: the home mapping).
        k.topk_rep = {}
        if self.dec_replicas:
            k.topk_rep = {
                w: _moe_topk_rep_op.build_decode(
                    w,
                    _moe_router_op.decode_splits(w),
                    self.dec_replicas,
                    dev,
                    signal_gg1=(w & (w - 1)) == 0,
                    expert_hist=self._pf is not None,
                )
                for w in range(1, _DECODE_MAX_ROWS + 1)
                if _moe_topk_rep_op.uses_rep_path(w)
            }
            self._dec_hot_ids = {
                i: torch.tensor(ids, dtype=torch.int32, device=dev)
                for i, ids in self.dec_hot.items()
            }
            self._dec_hot_none = torch.full((self.dec_replicas,), -1, dtype=torch.int32, device=dev)
        # Early-release top-k handles for the unpadded trunk decode widths that are not powers
        # of two (`TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL`): the decode step widths of an MTP3 /
        # MTP5 engine, `requests * (max_draft_len + 1)` up to 64 rows. Trunk only: the MTP
        # layer pads those widths and runs the routing kernel after its top-k, which must not
        # be released early (it does not wait for the ids).
        k.topk_sig = {}
        if self._moe_dec_unpad and self._moe_dec_unpad_signal and self.max_draft_len > 1:
            step = self.max_draft_len + 1
            sig_widths = [w for w in range(step, _DECODE_MAX_ROWS + 1, step) if w & (w - 1)]

            def build_sig(w):
                if w in k.topk_rep:
                    return _moe_topk_rep_op.build_decode(
                        w,
                        _moe_router_op.decode_splits(w),
                        self.dec_replicas,
                        dev,
                        signal_gg1=True,
                        expert_hist=self._pf is not None,
                    )
                return _moe_topk_op.build_decode(
                    w,
                    _moe_router_op.decode_splits(w),
                    dev,
                    signal_gg1=True,
                    expert_hist=self._pf is not None and _moe_topk_op.uses_push_path(w),
                )

            built = _try_build(
                "MoE unpadded decode early-release top-k",
                lambda: {w: build_sig(w) for w in sig_widths},
            )
            k.topk_sig = built or {}
        # The MTP3 / MTP5 decode widths' front end (`Moe.decode_exact`, 65..256 rows): the
        # wide decode router's split-K partials, and the fused top-k summing them, one handle
        # per row range, writing the L2 prefetch's counts. MTP1 and no speculation build
        # neither. The partials need the wide top-k: without it the router is dropped too
        # (the prefill router and top-k serve those steps); without the router the top-k
        # reads the prefill router's one plane.
        k.router_wide = None
        k.topk_wide = {}
        if self.wide_cells and k.topk_pre is not None:
            k.router_wide = optional(
                lambda: _moe_router_op.build_wide(dev), "wide decode MoE router"
            )
            splits = _moe_router_op.SPLIT_K if k.router_wide is not None else 1
            wide = optional(
                lambda: {
                    r: _moe_topk_op.build_wide(
                        r[1],
                        splits,
                        dev,
                        expert_hist=self._pf is not None,
                        cluster=self.wide_topk_schedule[0],
                        trim=self.wide_topk_schedule[1],
                    )
                    for r in _moe_topk_op.WIDE_RANGES
                },
                "wide decode MoE top-k",
            )
            if wide is None:
                k.router_wide = None
            k.topk_wide = wide or {}
        self._build_moe_wide_rep(k, dev)
        # Shared with the MTP layer's front end: the same compiled kernel per row count.
        k.router = _moe_router_op.build_all(dev)
        self._moe_k = k
        if self.mtp_enabled and self._mtp_fp4 is None:
            mw = self._mtp
            self._mtp_slots = [slot(True) for _ in range(2)]
            m0 = self._mtp_slots[0]
            self._mtp_k = SimpleNamespace(
                experts=_mtp_experts_op.build(
                    x=m0.x,
                    c=m0.c,
                    out=m0.out,
                    topk_w=m0.w,
                    maps=m0.maps,
                    fc1_w=mw["fc1"],
                    fc2_w=mw["fc2"],
                    max_rows=rows,
                ),
                # No quantize launch on the bf16 path to carry the output clear, so the top-k
                # clears it; every MTP width is a power of two, so it always releases FC1.
                topk={
                    w: _moe_topk_op.build_decode(
                        w, _moe_router_op.decode_splits(w), dev, signal_gg1=True, fuse_zero=True
                    )
                    for w in _MOE_DECODE_WIDTHS
                },
            )

    def _warm_draft_rows(self) -> None:
        """MTP3 / MTP5, at load: launch once on zeros what an eager draft step runs at a
        per-row-count module. Those compile at load but launch only on a step of that count,
        and the first such launch loads its module inside the step: the load holds the host
        until the GPU drains (19-23 ms on a context step's eager draft step).
        Decode graphs load the counts they capture; an eager draft step takes any count 1..64.
        - The MTP layer's routed experts at every count 1..64: the top-k per count; routing,
          quantize and clear per power of two. Same path, handles and buffers as the step
          (the FC2 tile counters reset at each launch's end).
        - The eh_proj K-split instances up to 64 rows (`Mtp.front`).
        - The shared expert's gate|up at the counts no runtime-M cell serves (33..63 but 48:
          the 48 / 64 cells are compiled for their own M, and the fused MTP cells take 65
          rows and up): `Gemm.project` itself on the step's operand shape, so cuBLAS loads the
          kernels it picks for those shapes here.
        The slot parity, the MTP layer's L2 prefetch counts and the route counters are
        restored after."""
        dev = self.w["embed"].device
        j = self.num_layers
        hits, misses = dict(self.route_hits), dict(self.route_misses)
        x = torch.zeros(
            (_DECODE_MAX_ROWS, self.w["embed"].shape[1]), dtype=torch.bfloat16, device=dev
        )
        if self.mtp_enabled:
            runner = self._mtp["ehproj"]
            for m in _mtp_eh.SUPPORTED_M_VALUES:
                if m <= _DECODE_MAX_ROWS:
                    eh = torch.zeros(
                        (m, _mtp_eh.LOCAL_INPUT_SIZE), dtype=torch.bfloat16, device=dev
                    )
                    runner.forward_rows(eh, m)
            for rows in range(1, _DECODE_MAX_ROWS + 1):
                if rows not in self._shared_gu_swiglu:
                    self.op.gemm.project(DRAFT, "shared_gate_up", x[:rows], self._mlp[j][0])
        if self._mtp_fp4 is not None:
            slot = self.op.moe.slot
            hist = self._pf.hist[j].clone() if self._pf is not None else None
            for rows in range(1, _DECODE_MAX_ROWS + 1):
                self.op.moe.routed(
                    PREFILL, x[:rows], self._mtp_fp4, rows, False, None, False, False, j
                )
            self.op.moe.slot = slot
            if hist is not None:
                self._pf.hist[j].copy_(hist)
        torch.cuda.synchronize()
        self.route_hits.clear()
        self.route_hits.update(hits)
        self.route_misses.clear()
        self.route_misses.update(misses)

    def _select_target(
        self, attn_metadata, rows: int
    ) -> "PrefillTarget | DecodeTarget | WideDecodeTarget":
        """Which target runs this step of `rows` rows.

        Host-side integers only: a decode-only CUDA graph is a per-capture constant
        environment. A mixed batch goes to prefill, so prefill is the general case and the
        decode targets the specializations that may assume there are no context rows. With
        more than one draft token a decode step's rows reach 32 x (max_draft_len + 1), past
        the 64 rows the decode cells serve, as do an MTP1 or non-speculative engine's above 32 /
        64 sequences: those steps are `WideDecodeTarget`'s wherever the wide decode cells are
        built (`wide_cells`).
        """
        if attn_metadata.num_contexts:
            return self._prefill
        if self.wide_cells and rows > _DECODE_MAX_ROWS:
            return self._wide
        return self._decode

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
        # Set on every metadata instance this model sees (the engine's and each CUDA graph
        # runner's); it takes effect from that instance's next `prepare()`.
        attn_metadata.dense_topk_on_device = self._dense_topk_on_device
        self._derive_first_step(attn_metadata)
        rows = int((input_ids if inputs_embeds is None else inputs_embeds).shape[0])
        out = self._select_target(attn_metadata, rows).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        # After the step, so a report includes it. Every context step reports (they run
        # eagerly and are few); decode steps every `_ROUTE_REPORT_EVERY` Python forwards.
        self._forwards += 1
        if self.tp_rank == 0 and (
            attn_metadata.num_contexts or self._forwards % _ROUTE_REPORT_EVERY == 0
        ):
            logger.info(f"modeling_v2 GLM-5.2 routes after {self._forwards} forwards: "
                        f"{self.route_report()}")  # fmt: skip
        return out

    def _derive_first_step(self, md) -> None:
        """What depends on values the engine resolves: compiled on the first forward, which
        is eager, ahead of any capture.

        The indexer K pool views are not cached: TRT-LLM replaces the KV cache manager during
        boot, and a view taken before the swap still reads the same sizes, strides and
        pointer while writing into a dead allocation. The pool's layer mapping does survive
        that swap (same trunk, same page size), but not the speculative one, so the MTP layer
        reads its own live.
        """
        # Derived again for a new KV cache manager: the final one can hold a shorter
        # `max_seq_len`, so a narrower block table, than the one the first forward saw.
        if not self._derive_pending and md.kv_cache_manager is self._derived_for:
            return
        dev = torch.cuda.current_device()
        # A table shorter than max_seq_len is read out of bounds with no check, and
        # `rope_max_positions` is inert on the MLA path.
        if md.max_seq_len > self._rope_positions:
            self._rope_positions = md.max_seq_len
            self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
        pool_map = md.host_kv_cache_pool_mapping
        self._stride_factor = pool_map.shape[0] * md.tokens_per_block
        self._pool_layer_id = [int(row[1]) for row in pool_map.tolist()]
        max_blocks = int(md.block_table.shape[1])
        self._req_global_compiled = build_req_index_to_global(
            dev, topk=self.idx_topk, max_blocks=max_blocks
        )
        self._req_base_compiled = build_req_index_to_base(
            dev, topk=self.idx_topk, max_blocks=max_blocks
        )
        # The decode scorer's column count is frozen at capture, so it is an engine constant:
        # a replay whose sequences outgrew a batch maximum would read unwritten columns.
        self._max_ctx_len = md.max_seq_len
        # A width with no handle takes the trtllm op: max_seq_len <= index_topk, or a failed build.
        # The 1-row steps, MTP1's 2, and MTP3 / MTP5's max_draft_len + 1.
        self._ix_topk_dec_compiled = {}
        for next_n in (1, *self.mtp_widths):
            try:
                self._ix_topk_dec_compiled[next_n] = build_ix_topk_decode(
                    dev, max_columns=md.max_seq_len, top_k=self.idx_topk, next_n=next_n
                )
            except Exception as e:  # noqa: BLE001 -- that width takes the trtllm op
                logger.warning(
                    f"modeling_v2 GLM-5.2: indexer top-k decode build failed for next_n={next_n} "
                    f"({e}); using the trtllm op"
                )
        # GVR only where it serves this engine's width, warmed ahead of any capture.
        self._dec_topk_gvr_live = self._dec_topk_gvr is not None and _dec_topk_gvr_serves(
            int(md.max_seq_len)
        )
        if self._dec_topk_gvr_live:
            self._warm_dec_topk_gvr(md)
        if self.mtp_enabled:
            # MTP3 / MTP5 also width 1: the MTP layer's 1-token draft steps
            # (`Attn.mtp_rope_specialised`). A separate draft pool's block-table width compiles at
            # the first eager draft step, as width max_draft_len + 1 does.
            for p in self.mtp_widths + ((1,) if self.max_draft_len > 1 else ()):
                self._mla_rope_handle(int(md.kv_cache_block_offsets.shape[-1]), dev, p)
        # The context rope/append kernel is keyed by the page size. None keeps the stock op.
        try:
            self._ctx_rope_compiled = _ctx_rope_op.compile(
                tokens_per_block=md.tokens_per_block, device_index=dev
            )
        except Exception as e:  # noqa: BLE001 -- the context rows take the stock rope/append op
            logger.warning(
                f"modeling_v2 GLM-5.2: context rope/append build failed ({e}); using stock op"
            )
            self._ctx_rope_compiled = None
        k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(0)
        assert (
            k_pool.dtype == torch.uint8
            and k_pool.ndim == 4
            and k_pool.shape[2] == 1
            and k_pool.shape[3] >= _INDEXER_PER_TOKEN_MIN
        ), f"the fused indexer k write needs a [blocks, block, 1, >=132] uint8 pool: {k_pool.shape}"
        # Both pool layouts: the trunk's per-layer views are strided and a separate draft pool's
        # may not be, and a layout first met inside a capture cannot compile there.
        self._ix_k_compiled = {
            contiguous: build_ix_k_cache(
                dev,
                is_neox=self.indexer_is_neox,
                cache_contiguous=contiguous,
                f32_input=self._ix_nosplit,
            )
            for contiguous in (False, True)
        }
        if self.wide_cells and self._ix_qk_on:
            self._ix_qk_compiled = (
                _try_build(
                    "indexer q side + K update",
                    lambda: {
                        contiguous: build_ix_qk(
                            dev, is_neox=self.indexer_is_neox, cache_contiguous=contiguous
                        )
                        for contiguous in (False, True)
                    },
                )
                or {}
            )
        # Every split-K variant the decode attention selects for 1..max_batch_size sequences,
        # both index kinds. `max_num_requests` is the engine's max_batch_size, in every
        # metadata it hands over (a CUDA-graph copy keeps it): no step schedules more
        # sequences, and a CUDA-graph batch is padded only up to a captured size no larger.
        # Its grid holds the kernel's default 2 query rows per sequence (MTP1, and the 1-row
        # steps). MTP3 / MTP5 add a second grid for max_draft_len + 1 rows (4 / 6), used only by
        # steps that wide: the split-K pick divides the SMs by sequences x grid width, so a
        # 1-row draft step on the wide grid would take fewer split-K CTAs per sequence (at 32
        # sequences one where the width-2 grid takes two) and leave 3 or 5 of every 4 / 6 query
        # CTAs idle. The variants are keyed by split and reducer only, so most are shared.
        self._fmha_dec_compiled = _fmha_dec_op.compile(
            max_batch_size=int(md.max_num_requests), device_index=dev
        )
        if self.max_draft_len > 1:
            self._fmha_dec_wide = _fmha_dec_op.compile(
                max_batch_size=int(md.max_num_requests),
                device_index=dev,
                max_sq=self.max_draft_len + 1,
            )
            # The trunk's pure decode steps at that width read the indexer's positions
            # directly (one gather of a pool row per group of rows that selected it); the MTP
            # layer and mixed steps keep the per-row handle above.
            if (
                self._union_on
                and self.max_draft_len + 1 in _fmha_union_op.UNION_ROWS_PER_GROUP
                and int(md.tokens_per_block) == _fmha_dec_op.LAYER_ROW_STRIDE
            ):
                self._fmha_union = _fmha_union_op.compile_union(
                    max_batch_size=int(md.max_num_requests),
                    device_index=dev,
                    max_sq=self.max_draft_len + 1,
                )
            elif self._union_on and self.max_draft_len + 1 in _fmha_union_op.UNION_ROWS_PER_GROUP:
                logger.warning(
                    f"modeling_v2 GLM-5.2: union decode attention needs tokens_per_block "
                    f"{_fmha_dec_op.LAYER_ROW_STRIDE} (got {int(md.tokens_per_block)}); "
                    "pure decode steps keep the per-row decode attention"
                )
        self._derive_pending = False
        self._derived_for = md.kv_cache_manager

    def _warm_dec_topk_gvr(self, md) -> None:
        """Compile the GVR decode top-k's launchers for every row count a decode step of this
        engine can carry (1..max_batch_size sequences at each generation width), as the
        engine's `warmup_selfsampling_topk` does for its own `next_n`: a launcher is keyed by the
        exact row count and the logits row stride, and one first met inside a capture cannot
        compile there (that graph then keeps ours, `Indexer.decode`). The scorer's rows are
        the DSL paged-MQA arena's, `max_seq_len` rounded up to 256 columns."""
        from tensorrt_llm._torch.kernels.blackwell.top_k import (
            gvr_topk_decode_self_sampling_host as gvr_host,
        )

        seqs = int(md.max_num_requests)
        for next_n in (1, *self.mtp_widths):
            gvr_host.warmup_varlen(
                self.idx_topk,
                int(md.max_seq_len),
                compress_ratio=1,
                next_n=next_n,
                num_rows_list=tuple(next_n * b for b in range(1, seqs + 1)),
                row_stride=(int(md.max_seq_len) + 255) // 256 * 256,
                deterministic=self._dec_topk_gvr.gvr_deterministic,
            )

    # ---- blocks both targets run ----

    def _count(self, route: str, hit: bool) -> None:
        """One call of `route`: on its CuTe DSL kernel (`hit`) or on the stock op."""
        counts = self.route_hits if hit else self.route_misses
        counts[route] = counts.get(route, 0) + 1

    def route_report(self) -> str:
        """`name=hits` per route, `/misses` appended where a call took the stock op."""
        names = sorted(set(self.route_hits) | set(self.route_misses))
        report = " ".join(
            f"{n}={self.route_hits.get(n, 0)}"
            + (f"/{self.route_misses[n]}" if self.route_misses.get(n) else "")
            for n in names
        )
        if self.mtp_last_declines:
            declined = ",".join(f"{k}:{v}" for k, v in sorted(self.mtp_last_declines.items()))
            report += f" mtp_ctx_last_declined={declined}"
        return report

    def build_logits_head(self, lm_head) -> None:
        """Compile the `lm_head` local GEMM (this rank's vocabulary shard) at every row
        count it has a schedule for. Taken only for the head the gather was built for:
        unpadded, column-parallel, gathering, the TP4 shard shape. Called by the shell
        once the weight is loaded; a head this does not take keeps upstream's path."""
        w = lm_head.weight
        if (
            self._lamport.gather is None
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or int(getattr(lm_head, "tp_size", 1)) != self.tp_size
            or tuple(w.shape) != (_allgather_op.SHARD_N, self.w["embed"].shape[1])
        ):
            logger.warning("modeling_v2 GLM-5.2: lm_head not on the gathered fast path")
            return
        self._lm_head = lm_head
        # MTP3 / MTP5 add their decode row counts with a schedule of their own (MTP5's 192).
        rows = _projection_op.ROWS["lm_head"] + (
            _projection_op.MTP_ROWS["lm_head"] if self.max_draft_len > 1 else ()
        )
        self._lm_head_runners = _projection_op.build_all("lm_head", w, rows)
        if self.tp_rank == 0:
            logger.info(
                f"modeling_v2 GLM-5.2 lm_head: {len(self._lm_head_runners)} of "
                f"{len(rows)} row counts on the CuTe DSL GEMM"
            )

    def logits(self, x, lm_head, logits_processor, attn_metadata):
        """Full-vocabulary fp32 logits for the hidden rows `x`, every one of them (what
        `LogitsProcessor.forward` returns with `return_context_logits=True`): the local
        GEMM on its CuTe DSL kernel, then one Lamport launch that gathers the four shards
        and widens them. Any other head, row count or layout keeps `logits_processor`.

        MTP3 / MTP5 only: a count of 65..255 rows without its own schedule (mixed steps,
        and decode steps of fewer than 32 sequences; MTP5's 192 verify rows have one) runs
        on the next compiled bucket (128, 192 or 256): that instance launched at the exact
        rows (`_projection_op.at_rows`), each row bitwise what the padded launch wrote for
        it; with `_wide_bucket` off (or a bucket compiled for one M), on zero rows past
        `rows`, returning the first `rows`. MTP1 keeps the path above."""
        rows = int(x.shape[0])
        runner = self._lm_head_runners.get(rows) if lm_head is self._lm_head else None
        padded = rows
        if runner is None and lm_head is self._lm_head and self.max_draft_len > 1 and rows < 256:
            padded = min((m for m in self._lm_head_runners if m > rows), default=rows)
            runner = self._lm_head_runners.get(padded) if padded <= 256 else None
            if runner is not None and self._wide_bucket:
                key = ("lm_head", rows)
                if key not in self.op.gemm._wide_rag:
                    self.op.gemm._wide_rag[key] = _projection_op.at_rows(runner, rows)
                if self.op.gemm._wide_rag[key] is not None:
                    runner, padded = self.op.gemm._wide_rag[key], rows
                    self._count("logits_at_rows", True)
        if (
            runner is not None
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 128 == 0
            and padded * _allgather_op.SHARD_N <= self._lamport.max_rows * self._lamport.hidden
        ):
            self._count("logits", True)
            if padded != rows:
                self._count("logits_pad", True)
                xp = torch.zeros([padded, x.shape[1]], dtype=x.dtype, device=x.device)
                xp[:rows].copy_(x)
                x = xp
            local = torch.empty([padded, runner.n], dtype=torch.bfloat16, device=x.device)
            cute_dsl_projection(x, lm_head.weight, local, runner)
            out = torch.empty(
                [padded, self.tp_size * runner.n], dtype=torch.float32, device=x.device
            )
            out = cute_dsl_allgather_widen(local, out, self._lamport)
            return out if padded == rows else out[:rows]
        self._count("logits", False)
        return logits_processor.forward(x, lm_head, attn_metadata, True)

    def gather_rows(self, hidden, rows):
        """`hidden[rows]`: the CuTe DSL row gather, else `embedding(rows, hidden)`."""
        handle = self._rows.gather
        hit = handle is not None and _row_gather_op.takes(hidden, rows)
        self._count("logits_rows", hit)
        if hit:
            return cute_dsl_hidden_row_gather(hidden, rows, handle)
        return nn.functional.embedding(rows, hidden)

    def _mla_rope_handle(
        self, width: int, dev: int, p: int = _MLA_ROPE_MTP_WIDTH
    ) -> "_mla_rope_op.CompiledMlaRopeGeneration":
        """The MLA rope kernels for a block table `width` columns wide and `p` new tokens per
        sequence (one of `mtp_widths`, or 1 for MTP3 / MTP5's 1-token draft steps,
        `Attn.mtp_rope_specialised`). The operand layout is this target's: `q_pe` the rope
        half of the `[T, heads, qk_dim]` q_b output, the latent the tail of the
        `[T, q_lora + lat_dim]` kv_a output. A pair not seen yet compiles here, which is the
        first eager forward that uses it (the trunk's are built in `_derive_first_step`).
        Both q-base variants are built: a step whose rows a decode K-B cell serves (2..64)
        takes the q-base from that GEMM, any wider one (MTP3 / MTP5's 128 / 192 rows: the
        prefill K-B) has the rope kernel emit it."""
        handle = self._mla_rope_compiled.get((width, p))
        if handle is None:
            handle = _mla_rope_op.compile(
                max_blocks_per_seq=width,
                q_pe_head_stride=self.qk_dim,
                q_pe_token_stride=self.heads * self.qk_dim,
                latent_row_stride=self.q_lora + self.lat_dim,
                device_index=dev,
                predicted_tokens_per_seq=p,
                q_base_from_kb=bool(self._kb8),
            )
            self._mla_rope_compiled[(width, p)] = handle
        return handle

    # ---- the trunk's routed experts ----

    def _build_moe_wide_rep(self, k, dev) -> None:
        """The MTP3 / MTP5 wide decode steps' expert replicas: `k.topk_wide_rep`, one handle per
        wide row range (keyed as `k.topk_wide`).

        At 65..256 decode rows the prefill pair streams one expert's weights per active local
        expert, so a rank's routed time follows its count of distinct active experts and the
        post-MoE all-reduce waits for the busiest rank (real routing, 75 layers: 2.66 / 2.09
        experts above the mean at 128 / 192 rows). The wide replica top-k moves each active hot
        expert whole to the EP rank with the fewest active experts (the decode replicas'
        greedy), on the context replicas' slots: the `64 + R` deep prefill pair (`k.pair_rep`)
        and each trunk layer's hot ids (`_ctx_hot_ids`); the MTP layer has none and keeps the
        wide top-k. Built only with the wide top-k (MTP3 / MTP5), the context replicas and
        `wide_dec_replicas`, and only for the ranges up to `_WIDE_REP_MAX_ROWS` (65..128); a
        failed build leaves every step on the wide top-k."""
        k.topk_wide_rep = {}
        if not (
            self.wide_dec_replicas
            and k.topk_wide
            and k.pair_rep is not None
            and self.ctx_replicas
            and self._ctx_hot_ids
        ):
            return
        splits = next(iter(k.topk_wide.values())).logit_splits
        built = _try_build(
            "MoE wide decode replica top-k",
            lambda: {
                r: _moe_topk_wide_rep_op.build_wide_rep(
                    r[1],
                    splits,
                    self.ctx_replicas,
                    dev,
                    expert_hist=h.expert_hist,
                    cluster=h.cluster,
                    trim=h.trim,
                )
                for r, h in k.topk_wide.items()
                if r[1] <= _WIDE_REP_MAX_ROWS
            },
        )
        k.topk_wide_rep = built or {}


class MTPLayer:
    """The checkpoint's MTP module at layer `num_hidden_layers`, replayed once per draft step.

    It differs from a trunk layer in three ways: an `eh_proj` front end mixing the next token's
    embedding into `h`, bf16 experts, and its own head norm. The runtime feeds the returned
    tensor back in as the next draft step's hidden state, so above one draft token it is
    returned head-normalized, as stock's MTP layer returns it (`_mtp_out_normed`); MTP1 never
    feeds back and keeps the norm in `shared_head`.
    Not an `nn.Module`: every weight it reads lives in the core's ParameterDict.
    """

    def __init__(self, core: GlmMoeDsaTep4ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own norm (a distinct parameter from `model.norm`; already applied by
        `PrefillTarget._mtp_close` when `_mtp_out_normed`), then the trunk's `lm_head`."""
        core = self.core
        if core._mtp_out_normed:
            x = hidden_states
        else:
            x = core.op.rows.rmsnorm(hidden_states, core._mtp["head_norm"])
        if return_context_logits:
            return core.logits(x, lm_head, self.logits_processor, attn_metadata)
        return self.logits_processor.forward(x, lm_head, attn_metadata, return_context_logits)

    def forward(
        self, embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
    ):
        """Route the draft step by the phase this call's metadata shows: the worker rewrites the
        metadata in place between draft steps, so one engine step's draft calls can cross from
        prefill to decode."""
        return self.core._select_target(attn_metadata, int(hidden_states.shape[0])).mtp_forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
        )


class DraftModel:
    """The container the spec worker reaches the drafter through.

    GLM-5.2's MTP module carries no `embed_tokens` or `shared_head.head` of its own, so both
    are the trunk's. `embed_tokens` resolves through the core on every read: this is built
    while parameters are still meta tensors."""

    def __init__(self, core: GlmMoeDsaTep4ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2Glm52Nvfp4Sm103Tep4(
    DecoderModelForCausalLM[GlmMoeDsaTep4ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            GlmMoeDsaTep4ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # The fourth argument matters: the engine allocates a separate draft KV pool, and a
            # worker built without it runs the draft step outside the pool swap.
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                should_use_separate_draft_kv_cache(self.spec_config),
            )
            if type(self.spec_worker) is MTPEagleWorker:
                # The target's worker in place of the built-in one, with the same draft KV
                # cache choice: the built-in worker with the greedy picks on the argmax.
                self.spec_worker = Glm52MTPEagleWorker(
                    self.spec_config,
                    model_config,
                    self.spec_worker.use_separate_draft_kv_cache,
                    mapping=model_config.mapping,
                    core=self.model,
                )

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

        `resource_manager` must stay inside `**kwargs`, and `position_ids` must be handed on in
        the engine's `[1, T]` shape: the worker squeezes it itself."""
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                cast(torch.IntTensor, input_ids),
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        logits = self.model.logits(
            self.model.gather_rows(hidden, spec_metadata.gather_ids),
            self.lm_head,
            self.logits_processor,
            attn_metadata,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
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
        # The shell's `lm_head` is the one module of the forward the core does not own: its
        # local GEMM compiles here, against the loaded weight.
        self.model.build_logits_head(self.lm_head)
