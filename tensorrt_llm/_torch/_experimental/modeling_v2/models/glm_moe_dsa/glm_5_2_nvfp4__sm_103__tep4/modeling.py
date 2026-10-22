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
the all-reduce, the routed experts' router, top-k, routing, quantize and grouped GEMMs). A
call site whose row count has no compiled kernel falls back to the stock op. Both attention
phases run the specialised sparse-MLA kernels: the context phase on the sm103 prefill kernel,
the decode phase on the decode kernel.
The context step's projections, absorbed BMMs and rope/KV append run on the prefill
counterparts of the decode kernels; on a step where the engine skipped the indexer for the
context rows, the context attention reads the causal window through the block table.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm.logger import logger

from . import weights as _weights
from .ops.activation.cute_dsl_dense_swiglu import compile as compile_dense_swiglu
from .ops.activation.cute_dsl_dense_swiglu import cute_dsl_dense_swiglu
from .ops.attention import cute_dsl_mla_rope_append_prefill as _ctx_rope_op
from .ops.attention import cute_dsl_mla_rope_generation as _mla_rope_op
from .ops.attention import cute_dsl_sparse_mla_decode as _fmha_dec_op
from .ops.attention import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.cute_dsl_fp8_paged_mqa_logits import cute_dsl_fp8_paged_mqa_logits
from .ops.attention.cute_dsl_indexer_k_cache_update import build as build_ix_k_cache
from .ops.attention.cute_dsl_indexer_k_cache_update import cute_dsl_indexer_k_cache_update
from .ops.attention.cute_dsl_indexer_q_side_fused import build as build_ix_q_side
from .ops.attention.cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused
from .ops.attention.cute_dsl_indexer_topk_decode import build as build_ix_topk_decode
from .ops.attention.cute_dsl_indexer_topk_decode import cute_dsl_indexer_topk_decode
from .ops.attention.cute_dsl_indexer_topk_prefill import build as build_ix_topk_prefill
from .ops.attention.cute_dsl_indexer_topk_prefill import cute_dsl_indexer_topk_prefill
from .ops.attention.cute_dsl_mla_rope_append_prefill import cute_dsl_mla_rope_append_prefill
from .ops.attention.cute_dsl_mla_rope_generation import cute_dsl_mla_rope_generation
from .ops.attention.cute_dsl_req_index_to_base import build as build_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_base import cute_dsl_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_global import build as build_req_index_to_global
from .ops.attention.cute_dsl_req_index_to_global import cute_dsl_req_index_to_global
from .ops.attention.cute_dsl_sparse_mla_decode import (
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
    INDEX_KIND_GLOBAL,
    cute_dsl_sparse_mla_decode,
)
from .ops.attention.cute_dsl_sparse_mla_prefill import cute_dsl_sparse_mla_prefill
from .ops.attention.fp8_mqa_logits import fp8_mqa_logits
from .ops.attention.indexer_k_cache_gather_op import indexer_k_cache_gather_op
from .ops.attention.indexer_topk_decode import indexer_topk_decode
from .ops.attention.mla_rope_append_paged_kv_assign_q import mla_rope_append_paged_kv_assign_q
from .ops.attention.mla_rope_generation import mla_rope_generation
from .ops.comm._lamport import build_workspace as build_lamport_workspace
from .ops.comm.cute_dsl_allreduce import cute_dsl_allreduce
from .ops.comm.cute_dsl_allreduce_add_rmsnorm import cute_dsl_allreduce_add_rmsnorm
from .ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import (
    cute_dsl_allreduce_add_rmsnorm_quant_nvfp4,
)
from .ops.embedding.cute_dsl_embedding import build as build_embedding
from .ops.embedding.cute_dsl_embedding import cute_dsl_embedding
from .ops.gemm import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .ops.gemm import cute_dsl_indexer_pre_proj_split as _ix_pre_split_op
from .ops.gemm import cute_dsl_mtp_eh_proj as _mtp_eh
from .ops.gemm import cute_dsl_projection as _projection_op
from .ops.gemm import cute_dsl_shared_gate_up_swiglu as _shared_gu_swiglu_op
from .ops.gemm.bmm_out import bmm_out
from .ops.gemm.cute_dsl_absorbed_bmm import cute_dsl_absorbed_bmm
from .ops.gemm.cute_dsl_indexer_pre_proj_split import cute_dsl_indexer_pre_proj_split
from .ops.gemm.cute_dsl_mtp_eh_proj import cute_dsl_mtp_eh_proj
from .ops.gemm.cute_dsl_projection import cute_dsl_projection
from .ops.gemm.cute_dsl_shared_gate_up_swiglu import cute_dsl_shared_gate_up_swiglu
from .ops.moe import cute_dsl_moe_merge as _moe_merge_op
from .ops.moe import cute_dsl_moe_nvfp4_experts as _moe_experts_op
from .ops.moe import cute_dsl_moe_router as _moe_router_op
from .ops.moe import cute_dsl_moe_routing as _moe_routing_op
from .ops.moe import cute_dsl_moe_topk_routing as _moe_topk_op
from .ops.moe import cute_dsl_moe_zero as _moe_zero_op
from .ops.moe import cute_dsl_mtp_moe_experts as _mtp_experts_op
from .ops.moe.cute_dsl_moe_merge import cute_dsl_moe_merge
from .ops.moe.cute_dsl_moe_nvfp4_experts import cute_dsl_moe_nvfp4_experts
from .ops.moe.cute_dsl_moe_router import cute_dsl_moe_router
from .ops.moe.cute_dsl_moe_routing import cute_dsl_moe_routing
from .ops.moe.cute_dsl_moe_topk_routing import cute_dsl_moe_topk_routing
from .ops.moe.cute_dsl_moe_zero import cute_dsl_moe_zero
from .ops.moe.cute_dsl_mtp_moe_experts import cute_dsl_mtp_moe_experts
from .ops.moe.noaux_tc_op import noaux_tc_op
from .ops.norm.cute_dsl_mtp_norm_shard import compile as compile_mtp_norm_shard
from .ops.norm.cute_dsl_mtp_norm_shard import cute_dsl_mtp_norm_shard
from .ops.norm.cute_dsl_qkv_a_rmsnorm import compile as compile_qkv_a_rmsnorm
from .ops.norm.cute_dsl_qkv_a_rmsnorm import cute_dsl_qkv_a_rmsnorm
from .ops.quantization import cute_dsl_moe_quantize_nvfp4 as _moe_quant_op
from .ops.quantization.cute_dsl_moe_quantize_nvfp4 import cute_dsl_moe_quantize_nvfp4


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state the context attention reads: the
    rope/KV append's per-sequence offsets and pool addressing, the prefill kernel's pool and
    per-row KV windows."""
    return dict(
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        ctx_cached_token_indptr=md.ctx_cached_token_indptr,
        ctx_kv_indptr=md.ctx_kv_indptr,
        max_ctx_seq_len=md.max_ctx_seq_len,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        max_seq_len=md.max_seq_len,
        cu_seqlen_ks=md.cu_seqlen_ks,
        cu_seqlen_ke=md.cu_seqlen_ke,
        primary_pool=md.kv_cache_manager.get_unique_primary_pool(),
        # The prefill rope kernel's and the dense-causal FMHA's per-request lengths.
        ctx_uncached_token_indptr=md.ctx_uncached_token_indptr,
        num_ctx_cached_tokens=md.num_ctx_cached_tokens,
        seq_lens=md.seq_lens,
        seq_lens_cuda=md.seq_lens_cuda,
    )


_QUANT_MODE_FP8_KV = 128
_KV_RESIDUAL_DIM = 0
_MOE_MAX_T = 8192
# The MTP width the specialised MLA rope kernel and the base-row index kind are built for.
_MLA_ROPE_MTP_WIDTH = 2
# The decode kernels' widest compiled row count. Above it a projection with no compiled runner
# takes the prefill GEMM; at or below it, cuBLAS.
_DECODE_MAX_ROWS = 64
# The embedding kernel's widest row count. A step above it (a `max_num_tokens` past 8192)
# gathers in slices.
_EMBED_MAX_ROWS = 8192
# Token counts at which a kernel's int32 element offsets wrap: the context rope/append kernel
# (16 heads x 576 per token), so a step at or above it takes the stock op; and the indexer
# q-side kernel (32 heads x 128 per token), which has no stock counterpart here.
_ROPE_APPEND_MAX_TOKENS = 2**31 // (16 * 576)
_STEP_MAX_TOKENS = 2**31 // (32 * 128)
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
# fp8_mqa_logits allocates a `[q_rows, kv_rows]` fp32 score matrix per call; prefill chunks are
# tiled to keep it under this many elements. Context steps are not captured.
_PREFILL_LOGITS_BUDGET = 1 << 24


def _tile_bounds(start: int, end: int, kv_rows: int) -> list[tuple[int, int]]:
    rows = max(1, _PREFILL_LOGITS_BUDGET // max(kv_rows, 1))
    return [(a, min(a + rows, end)) for a in range(start, end, rows)]


def _moe_chunk_sizes(total: int) -> list[int]:
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


# The routed experts' row domains: the decode kernels (router, fused top-k, quantize, clear,
# decode grouped GEMMs) serve 1..64 rows -- `max_batch_size` 32 x MTP width 2 -- and the
# prefill kernels 65..8192. A decode step presents a power of two up to 64 after padding.
_MOE_TOP_K = _moe_routing_op.TOP_K
_MOE_DECODE_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
_MOE_PREFILL_MIN = _DECODE_MAX_ROWS + 1
# From here up the prefill pair runs the exact row count. The routing schedule is exact per
# `ceil(rows / 32)` up to 512 rows, so below this a power of two keeps the compiled
# schedules few; above it the four compiled buckets cover every count.
_MOE_PAIR_EXACT_FROM = 513
# NVFP4: one e4m3 scale per 16 elements; the shared expert's per-rank intermediate.
_MOE_SF_VEC = 16
_SHARED_INTER = 512


def _pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


def _direct_ok(t: torch.Tensor, dtype: torch.dtype) -> bool:
    """Read in place rather than staged. A staging copy is a memcpy node, which a PDL edge
    cannot pass through, so it is skipped for a 128-byte aligned contiguous tensor -- the
    collective's output, whose graph-pool address is stable."""
    return t.dtype is dtype and t.is_contiguous() and t.data_ptr() % 128 == 0


class GlmMoeDsaTep4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.tp_rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.quant_mode = _QUANT_MODE_FP8_KV

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
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scale = float(cfg.routed_scaling_factor)
        self.moe_inter = cfg.moe_intermediate_size
        self.local_experts = self.num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank
        self.mtp_enabled = getattr(model_config, "spec_config", None) is not None

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        self._derive_pending = True
        self._derived_for = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._idx_cos_sin: torch.Tensor | None = None
        self._idx_k_checked = False
        self._pool_layer_id: list[int] = []
        self._stride_factor = 0
        self._max_ctx_len = 0
        self._mtp: dict | None = None
        self._ctx_rope_compiled: _ctx_rope_op.CompiledMlaRopeAppendPrefill | None = None
        self._fmha_dec_compiled: _fmha_dec_op.CompiledSparseMlaDecode | None = None
        # The routed experts' operand slots: two of each buffer, alternating per layer (trunk)
        # or per call (MTP). See `_build_moe`.
        self._moe_slot = 0
        self._mtp_slot = 0
        # Context steps on the dense-causal context attention, and the others by reason.
        self.dense_causal_steps = 0
        self.dense_causal_misses: dict[str, int] = {}

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

    def post_load_weights(self) -> None:
        """Derive the operands, then build and compile every specialised kernel.

        Everything compiles here because a first compile inside a CUDA-graph capture breaks
        the graph; what needs an engine value compiles in `_derive_first_step`. Named for the
        shell hook that drives it.
        """
        dev = torch.cuda.current_device()
        layers = range(self.num_layers + (1 if self.mtp_enabled else 0))
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
        if self.mtp_enabled:
            self._mtp = self._derive_mtp()

        first_idx = next(i for i in range(self.num_layers) if self._idx[i] is not None)
        # One runner per (projection, compiled row count), built against one layer's row-major
        # weight, and one prefill GEMM per projection for the context row counts above
        # _DECODE_MAX_ROWS; `_project` sends a row count with neither to cuBLAS.
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
        if self.mtp_enabled:
            proj_weights["mtp_eh"] = self._mtp["eh"]
        self._proj_pre = _projection_op.build_prefill_all(
            {
                name: w_colmajor.t()
                for name, w_colmajor in proj_weights.items()
                if name in _projection_op.PREFILL_ENTRIES
            }
        )
        self._qkv_a_norm = compile_qkv_a_rmsnorm(dev, self.eps)
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
        self._ix_pre_split = _ix_pre_split_op.compile_all(dev)
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
        self._ix_topk_pre_compiled = build_ix_topk_prefill(
            dev, top_k=self.idx_topk, max_columns=_MOE_MAX_T
        )
        # The MLA rope kernel is keyed by the block-table width, an engine value: built per
        # width in `_derive_first_step` (the trunk's) and at the first eager use of another.
        self._mla_rope_compiled: dict[int, _mla_rope_op.CompiledMlaRopeGeneration] = {}
        # The context kernel in both modes and its two scale arrays, built here because a
        # first compile inside a CUDA-graph capture is a broken graph.
        self._fmha_pre_compiled = _fmha_pre_op.compile(device_index=dev)
        self._fmha_pre_dc_compiled = _fmha_pre_op.compile(device_index=dev, dense_causal=True)
        self._fmha_bmm1 = torch.tensor(_FMHA_BMM1_SCALE, dtype=torch.float32, device=dev)
        self._fmha_bmm2 = torch.tensor(_FMHA_BMM2_SCALE, dtype=torch.float32, device=dev)
        # A CUDA stream created inside a graph capture is not capturable.
        self._side_stream = torch.cuda.Stream(device=dev)
        self._build_moe(dev)
        # The peer buffers are sized for the whole token budget, so every collective shares
        # one workspace.
        self._lamport = build_lamport_workspace(
            self.w["embed"].shape[1], self.eps, _MOE_MAX_T, self.tp_rank, self.tp_size
        )

        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

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

        def optional(build):
            try:
                return build()
            except Exception as e:  # noqa: BLE001 -- that launch takes its stock op at the call site
                logger.warning(
                    f"modeling_v2 GLM-5.2: MoE prefill kernel build failed ({e}); using stock op"
                )
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
                maps=_moe_routing_op.allocate_maps(rows, dev),
                counts=counts,
                barrier=barrier,
                router_out=torch.zeros((_DECODE_MAX_ROWS, self.num_experts), **f32),
                router_partials=torch.zeros(
                    (_DECODE_MAX_ROWS, _moe_router_op.SPLIT_K, self.num_experts), **f32
                ),
                # The fused top-k checks its outputs against the exact row count: one set
                # per width.
                topk_out={w: _moe_topk_op.allocate_decode_outputs(w, dev) for w in widths},
            )
            if mtp:
                ns.c = _mtp_experts_op.allocate_c(rows, inter, dev)
                return ns
            ns.c = _moe_experts_op.allocate_decode_c(rows, inter, dev)
            ns.sfc = _moe_experts_op.allocate_sfc(rows, inter, dev)
            ns.router_x = torch.zeros((_DECODE_MAX_ROWS, hidden), **bf16)
            ns.logits = torch.zeros((rows, 1, self.num_experts), **f32)
            ns.topk_scratch = _moe_topk_op.allocate_prefill_scratch(dev)
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
            experts=experts,
            max_rows=rows,
        )
        k.pair = optional(lambda: _moe_experts_op.build_prefill_pair(hidden, inter, experts, dev))
        if k.pair is not None:
            for sl in self._moe_slots:
                sl.pc = _moe_experts_op.allocate_pair_c(rows, inter, dev)
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
        k.topk = {
            w: _moe_topk_op.build_decode(
                w, _moe_router_op.decode_splits(w), dev, signal_gg1=(w & (w - 1)) == 0
            )
            for w in range(1, _DECODE_MAX_ROWS + 1)
        }
        # Shared with the MTP layer's front end: the same compiled kernel per row count.
        k.router = _moe_router_op.build_all(dev)
        self._moe_k = k
        if self.mtp_enabled:
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

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment. A mixed batch goes to prefill, so prefill is the
        general case and decode the specialization that may assume it has no context rows.
        """
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
        self._ix_topk_dec_compiled = {}
        for next_n in (1, _MLA_ROPE_MTP_WIDTH):
            try:
                self._ix_topk_dec_compiled[next_n] = build_ix_topk_decode(
                    dev, max_columns=md.max_seq_len, top_k=self.idx_topk, next_n=next_n
                )
            except Exception as e:  # noqa: BLE001 -- that width takes the trtllm op
                logger.warning(
                    f"modeling_v2 GLM-5.2: indexer top-k decode build failed for next_n={next_n} "
                    f"({e}); using the trtllm op"
                )
        if self.mtp_enabled:
            self._mla_rope_handle(int(md.kv_cache_block_offsets.shape[-1]), dev)
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
                dev, is_neox=self.indexer_is_neox, cache_contiguous=contiguous
            )
            for contiguous in (False, True)
        }
        # Every split-K variant the decode attention selects for 1..max_batch_size sequences,
        # both index kinds. `max_num_requests` is the engine's max_batch_size, in every
        # metadata it hands over (a CUDA-graph copy keeps it): no step schedules more
        # sequences, and a CUDA-graph batch is padded only up to a captured size no larger.
        self._fmha_dec_compiled = _fmha_dec_op.compile(
            max_batch_size=int(md.max_num_requests), device_index=dev
        )
        self._derive_pending = False
        self._derived_for = md.kv_cache_manager

    # ---- blocks both targets run ----

    def _check_rows(self, partial) -> None:
        # The all-reduce kernels take the row count from their input, so a call above the
        # token budget the peer buffers were sized for would write past every peer's buffer.
        assert partial.shape[0] <= self._lamport.max_rows, (
            f"{partial.shape[0]} rows exceed the collective's budget of {self._lamport.max_rows}"
        )

    def _embed(self, input_ids):
        """The embedding rows for `input_ids`, in slices the kernel takes."""
        if input_ids.shape[0] <= _EMBED_MAX_ROWS:
            return cute_dsl_embedding(input_ids, self.w["embed"], self._embedding)
        return torch.cat(
            [
                cute_dsl_embedding(ids, self.w["embed"], self._embedding)
                for ids in input_ids.split(_EMBED_MAX_ROWS)
            ]
        )

    def _reduce(self, partial):
        n = self._lamport.max_rows
        if partial.shape[0] > n:
            # A step above the token budget the peer buffers hold reduces in slices.
            return torch.cat([self._reduce(p) for p in partial.split(n)])
        self._check_rows(partial)
        return cute_dsl_allreduce(partial, torch.empty_like(partial), self._lamport)

    def _project(self, name: str, x, w_colmajor, out_dtype=None, out=None):
        """`x @ w_colmajor` on the projection's CuTe DSL kernel at a compiled row count, on
        the prefill GEMM above the decode row counts, else cuBLAS. The kernels read
        `w_colmajor.t()`, the row-major weight, same memory. `out` is an optional caller-owned
        `[rows, N]` buffer; on the cuBLAS path the result is copied in."""
        rows = int(x.shape[0])
        runner = self._proj.get(name, {}).get(rows)
        pre = self._proj_pre.get(name)
        if (
            runner is None
            and pre is not None
            and _DECODE_MAX_ROWS < rows <= pre.max_rows
            and x.dtype == pre.in_dtype
            and x.is_contiguous()
            and out_dtype in (None, pre.out_dtype)
            and (out is None or out.dtype == pre.out_dtype)
        ):
            runner = pre
        if runner is None:
            res = cublas_mm(x, w_colmajor, None, out_dtype)
            if out is None:
                return res
            out.copy_(res)
            return out
        if out is None:
            out = torch.empty([rows, runner.n], dtype=runner.out_dtype, device=x.device)
        return cute_dsl_projection(x, w_colmajor.t(), out, runner)

    def _absorbed_bmm(self, kind: str, x_t, w_kn, w_nk, out_t) -> None:
        """An absorbed-MLA BMM ("k_b" or "v_b") into `out_t`: the decode kernel at its row
        counts (2..64), the prefill kernel at any other up to 8192, else `bmm_out`, which takes
        the `[H, K, N]` operand instead of `[H, N, K]`."""
        rows = int(x_t.shape[1])
        runner = self._absorb[kind].get(rows)
        if runner is None and rows <= _absorbed_bmm_op.PREFILL_MAX_ROWS:
            runner = self._absorb_pre[kind]
        if runner is None:
            bmm_out(x_t, w_kn, out_t)
        else:
            cute_dsl_absorbed_bmm(x_t, w_nk, out_t, runner)

    def _reduce_norm(self, partial, residual, weight):
        """All-reduce, add into `residual` in place, RMS norm: every trunk collective."""
        n = self._lamport.max_rows
        if partial.shape[0] > n:
            # Row slices of a contiguous `residual` are views, so the in-place add still lands.
            return torch.cat(
                [
                    self._reduce_norm(p, r, weight)
                    for p, r in zip(partial.split(n), residual.split(n))
                ]
            )
        self._check_rows(partial)
        out = torch.empty_like(partial)
        return cute_dsl_allreduce_add_rmsnorm(partial, residual, weight, out, self._lamport)

    def _reduce_norm_quant(self, partial, residual, weight, g1, rows):
        """`_reduce_norm`, plus the routed experts' NVFP4 input when the MoE would quantize
        on its own kernel (a decode width): into the layer's `xq` / `xsf`, exactly the rows
        it would quantize. Returns `(out, prequantized)`."""
        if rows > _DECODE_MAX_ROWS:
            return self._reduce_norm(partial, residual, weight), False
        self._check_rows(partial)
        xq, xsf = self._moe_xq[:rows], self._moe_xsf[:rows]
        out = torch.empty_like(partial)
        out = cute_dsl_allreduce_add_rmsnorm_quant_nvfp4(
            partial, residual, weight, g1, out, xq, xsf, self._lamport
        )
        return out, True

    def _mla_rope_handle(self, width: int, dev: int) -> "_mla_rope_op.CompiledMlaRopeGeneration":
        """The MLA rope kernels for a block table `width` columns wide. The operand layout is
        this target's: `q_pe` the rope half of the `[T, heads, qk_dim]` q_b output, the latent
        the tail of the `[T, q_lora + lat_dim]` kv_a output. A width not seen yet compiles
        here, which is the first eager forward that uses it (the trunk's is built in
        `_derive_first_step`)."""
        handle = self._mla_rope_compiled.get(width)
        if handle is None:
            handle = _mla_rope_op.compile(
                max_blocks_per_seq=width,
                q_pe_head_stride=self.qk_dim,
                q_pe_token_stride=self.heads * self.qk_dim,
                latent_row_stride=self.q_lora + self.lat_dim,
                device_index=dev,
            )
            self._mla_rope_compiled[width] = handle
        return handle

    def _qkv(self, x, a):
        """`(q, q_a, latent)`. The kv norm writes in place, so the latent `[ckv | k_pe]` is the
        fused projection's tail, a strided view."""
        qkva = self._project("qkva", x, a.qkva)
        q_a_raw, ckv_raw, _ = torch.split(qkva, [self.q_lora, self.kv_lora, self.rope_dim], -1)
        q_a, _ = cute_dsl_qkv_a_rmsnorm(
            q_a_raw, a.q_norm, ckv_raw, a.kv_norm, self.eps, self._qkv_a_norm
        )
        latent = torch.split(qkva, [self.q_lora, self.lat_dim], -1)[1]
        return self._project("q_b", q_a, a.qb), q_a, latent

    def _topk_buf(self, md, rows: int, tc: int, skip_ctx: bool, skip_gen: bool):
        """Where this step's selection lands. A phase the engine marked skippable (every
        sequence shorter than index_topk) takes the dense causal index the engine wrote; a
        branch frozen into a graph, safe because short and long steps capture separately."""
        if skip_ctx and skip_gen:
            return md.topk_indices_buffer[:rows]
        # `topk_indices_buffer` exists only while the engine's indexer skip is on.
        buf = torch.empty(
            [rows, self.idx_topk], dtype=torch.int32, device=torch.cuda.current_device()
        )
        if skip_ctx and tc:
            buf[:tc].copy_(md.topk_indices_buffer[:tc])
        if skip_gen and rows > tc:
            buf[tc:].copy_(md.topk_indices_buffer[tc:rows])
        return buf

    def _indexer_project(self, x, q_a, pos, ix, rows: int, need_q: bool):
        """The indexer's front: `(q_fp8, weights, k_pre)`; `q_fp8` and `weights` are None on a
        step where every phase takes the engine's index. The k side always runs: the keys this
        step appends are read by later steps."""
        hi, di = self.idx_heads, self.idx_dim
        dev = x.device
        kb = torch.empty([rows, di], dtype=x.dtype, device=dev)
        split = self._ix_pre_split.get(rows)
        pre = self._ix_pre_split_pre
        if (
            split is None
            and pre is not None
            and _DECODE_MAX_ROWS < rows <= pre.max_rows
            and x.is_contiguous()
        ):
            split = pre
        if split is not None:
            wq = torch.empty([rows, hi], dtype=torch.float32, device=dev)
            cute_dsl_indexer_pre_proj_split(x, ix.fused_bf16, kb, wq, split)
        else:
            # fp32: the only `weights` dtype fp8_mqa_logits takes.
            fused = self._project("ix_fused", x.float(), ix.fused, torch.float32)
            k_raw, wq_raw = torch.split(fused, [di, hi], -1)
            kb.copy_(k_raw)
            # The scorers check `weights`' stride(1) but not stride(0), then read it densely:
            # a strided slice scores each row with another row's head weights.
            wq = wq_raw.contiguous()
        k_pre = (kb, ix.kn_w, ix.kn_b, pos)
        if not need_q:
            return None, None, k_pre
        q = torch.reshape(self._project("ix_qb", q_a, ix.qb), [rows, hi, di])
        q_fp8 = torch.empty([rows, hi, di], dtype=torch.float8_e4m3fn, device=dev)
        weights = torch.empty([rows, hi], dtype=torch.float32, device=dev)
        cute_dsl_indexer_q_side_fused(
            q, pos, self._idx_cos_sin, wq, self.idx_weight_scale, q_fp8, weights,
            is_neox=self.indexer_is_neox, compiled=self._ix_qside_compiled,
        )  # fmt: skip
        return q_fp8, weights, k_pre

    def _indexer_write_k(self, md, k_pool, k_pre, rows: int) -> None:
        """Norm, rope, fp8-pack and append this step's keys in one launch."""
        kb, kn_w, kn_b, pos = k_pre
        cute_dsl_indexer_k_cache_update(
            kb,
            kn_w,
            kn_b,
            pos,
            self._idx_cos_sin.view(-1, 2, self.rope_dim // 2),
            k_pool,
            md.slot_mapping_fp8,
            md.slot_mapping_scale,
            rows,
            is_neox=self.indexer_is_neox,
            compiled=self._ix_k_compiled[k_pool.is_contiguous()],
            # Its validation reduces the slot mappings to a host bool, a sync that invalidates
            # a capture: checked once, on the first eager call.
            validate=not self._idx_k_checked,
        )
        self._idx_k_checked = True

    def _indexer_decode(
        self, md, k_pool, q_fp8, weights, topk_buf, ctx: int, seqs: int, gen_p: int
    ):
        """Score the generation rows against the paged K cache and select `index_topk`.
        `ctx` is the number of context sequences ahead of them in the batch."""
        # Cached plus all `gen_p` of this step's tokens; both consumers take the same vector,
        # which keeps the selector's causal bound inside the columns the scorer wrote.
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + seqs]
        logits = cute_dsl_fp8_paged_mqa_logits(
            torch.reshape(q_fp8, [seqs, gen_p, self.idx_heads, self.idx_dim]),
            k_pool,
            weights,
            kv_lens,
            md.indexer_k_cache_block_offsets[ctx : ctx + seqs],
            md.scheduler_metadata_buffer,
            self._max_ctx_len,
        )
        compiled = self._ix_topk_dec_compiled.get(gen_p)
        if compiled is None:
            indexer_topk_decode(
                logits, kv_lens, topk_buf, gen_p, self.idx_topk,
                radix_aux_indices=md.radix_aux_indices, radix_aux_logits=md.radix_aux_logits,
            )  # fmt: skip
        else:
            cute_dsl_indexer_topk_decode(
                logits, kv_lens, topk_buf, top_k=self.idx_topk, next_n=gen_p, compiled=compiled
            )

    def _attn_gen(
        self,
        a,
        layer: int,
        q,
        latent,
        out,
        slots,
        gen: int,
        gen_p: int,
        ctx: int,
        md,
        slot_kind,
        slot_layer,
        specialised_rope: bool,
    ) -> None:
        """Sparse MLA over the generation rows, absorbed form, into `out`."""
        heads, dev, dt = self.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [gen, heads, self.qk_dim]), [self.nope, self.rope_dim], -1
        )
        fused_q = torch.empty([gen, heads, self.lat_dim], dtype=dt, device=dev)
        fq_nope = torch.split(fused_q, [self.kv_lora, self.rope_dim], -1)[0]
        # Over an fp8 pool the rope kernel reads this half to build the quantized query, so
        # the BMM is issued first on the same stream.
        self._absorbed_bmm(
            "k_b",
            torch.transpose(q_nope, 0, 1),
            a.k_b,
            a.k_b_nk,
            torch.transpose(fq_nope, 0, 1),
        )
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, self.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1 = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2 = torch.empty([1], dtype=torch.float32, device=dev)
        rope = self._rope
        if specialised_rope:
            # The specialised kernel takes the stock op's earlier positional ABI.
            cute_dsl_mla_rope_generation(
                fused_q, q_pe, latent, rope["rotary_cos_sin"], cu_q, cu_kv, counter, bmm1, bmm2,
                quant_q, md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.prompt_lens_cpu_runtime,
                ctx, md.kv_cache_block_offsets, md.host_kv_cache_pool_pointers,
                md.host_kv_cache_pool_mapping, None, None, None, None, [None, None], gen_p, layer,
                heads, 1, self.lat_dim, md.tokens_per_block, md.max_seq_len, 1, self.quant_mode,
                1.0, self.q_lora, self.kv_lora, self.nope, self.rope_dim, self.kv_lora, True,
                compiled=self._mla_rope_handle(int(md.kv_cache_block_offsets.shape[-1]), dev.index),
            )  # fmt: skip
        else:
            mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                # The stock op takes packed rows only.
                latent_cache=latent.contiguous(),
                rotary_cos_sin=rope["rotary_cos_sin"],
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1,
                mla_bmm2_scale=bmm2,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=ctx,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=layer,
                num_heads=heads,
                num_kv_heads=1,
                head_size=self.lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=md.tokens_per_block,
                attention_window_size=md.max_seq_len,
                beam_width=1,
                quant_mode=self.quant_mode,
                q_scaling=1.0,
                q_lora_rank=self.q_lora,
                kv_lora_rank=self.kv_lora,
                qk_nope_head_dim=self.nope,
                qk_rope_head_dim=self.rope_dim,
                v_head_dim=self.kv_lora,
                rope_append=True,
            )
        lat_out = torch.empty([gen, heads * self.kv_lora], dtype=dt, device=dev)
        seqs = gen // gen_p
        cute_dsl_sparse_mla_decode(
            quant_q,
            md.kv_cache_manager.get_unique_primary_pool(),
            slots,
            lat_out,
            cu_q[: seqs + 1],
            md.kv_lens_cuda_runtime[ctx : ctx + seqs],
            bmm1,
            bmm2,
            slot_kind,
            slot_layer,
            compiled=self._fmha_dec_compiled,
        )
        self._absorbed_bmm(
            "v_b",
            torch.transpose(torch.reshape(lat_out, [gen, heads, self.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [gen, heads, self.v_dim]), 0, 1),
        )

    def _shared_mlp(self, x, gu, gu_il, dn, out=None, trunk=False):
        """The shared expert, this rank's slice, a partial sum. `gu_il` is this layer's
        interleaved gate/up weight; `out` is the routed experts' output buffer when FC2 will
        accumulate onto it. A trunk layer's context rows take the prefill shared kernels
        when they built; the MTP layer's keep the two-launch path."""
        rows = int(x.shape[0])
        k = self._moe_k
        if trunk and k.shared_down_pre is not None and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T:
            xs = x if _direct_ok(x, torch.bfloat16) else x.contiguous()
            # The prefill kernels take 128-byte aligned rows; a contiguous view that is not
            # takes the two-launch path.
            if xs.dtype is torch.bfloat16 and xs.data_ptr() % 128 == 0:
                res = out
                if res is None:
                    res = torch.empty((rows, x.shape[1]), dtype=torch.bfloat16, device=x.device)
                mid = self._moe_mid[:rows]
                cute_dsl_shared_gate_up_swiglu(xs, gu_il, mid, k.shared_up_pre)
                return cute_dsl_projection(mid, dn.t(), res, k.shared_down_pre)
        runner = self._shared_gu_swiglu.get(rows)
        if runner is None:
            act = flashinfer_silu_and_mul(self._project("shared_gate_up", x, gu))
        else:
            act = torch.empty([rows, gu.shape[1] // 2], dtype=x.dtype, device=x.device)
            cute_dsl_shared_gate_up_swiglu(x, gu_il, act, runner)
        return self._project("shared_down", act, dn, out=out)

    # ---- the trunk's routed experts ----

    def _moe_pair_ok(self, rows: int) -> bool:
        """Whether `rows` takes the prefill grouped GEMM pair: a context row count above the
        decode kernels' 64."""
        return self._moe_k.pair is not None and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T

    def _moe_padded_rows(self, rows: int) -> int:
        """The row count routing and the grouped GEMMs see for `rows`: a power of two, or the
        exact count on the prefill pair from `_MOE_PAIR_EXACT_FROM` up. `_moe_begin_layer`
        and `_moe_experts` both ask here, so the shared expert's rows and FC2's agree."""
        if rows >= _MOE_PAIR_EXACT_FROM and self._moe_pair_ok(rows):
            return rows
        return _pow2(rows)

    def _moe_begin_layer(self, rows: int):
        """Open one layer: advance the slot for the router, top-k and grouped GEMMs, and
        return the routed output rows for the shared expert to write into, or None. Only at
        a width the GEMMs do not pad: a power of two, or any count on the prefill pair from
        513. Called before the shared branch forks, so the shared expert writes the slot
        FC2 accumulates into."""
        self._moe_slot ^= 1
        if self._moe_padded_rows(rows) != rows or rows > _MOE_MAX_T:
            return None
        return self._moe_slots[self._moe_slot].out[:rows]

    def _moe_router(self, x, m, slot):
        """Router logits in the slot's buffer, or None (the stock projection then runs it).
        Up to 64 rows on the decode router, where above 5 rows these are `[rows, 4, 256]`
        partials only the fused top-k reads; above, on the prefill router (`[rows, 256]`)."""
        rows = int(x.shape[0])
        s = self._moe_slots[slot]
        if rows > _DECODE_MAX_ROWS:
            pre = self._moe_k.router_pre
            if pre is None or rows > _MOE_MAX_T:
                return None
            if not _direct_ok(x, torch.bfloat16):
                x = x.contiguous()
            if x.dtype is not torch.bfloat16 or x.data_ptr() % 16:
                return None
            out = s.logits[:rows]
            cute_dsl_moe_router(x, m.router, out, pre)
            return out.view(rows, -1)
        if _direct_ok(x, torch.bfloat16):
            xs = x
        else:
            xs = s.router_x[:rows]
            xs.copy_(x)
        out = (
            s.router_out[:rows]
            if rows <= _moe_router_op.SIMT_MAX_ROWS
            else s.router_partials[:rows]
        )
        cute_dsl_moe_router(xs, m.router, out, self._moe_k.router[rows])
        return out

    def _moe_topk(self, logits, in_slot: bool, m, slot):
        """Sigmoid top-k with bias correction, writing this layer's tile maps: `(weights,
        ids, maps)` in the slot's buffers, or None (`noaux_tc_op` then selects). Up to 64 rows
        on the decode kernel, above on the prefill one. `in_slot`: the logits are the slot's
        router output, read in place; others are staged into it first. Matches
        `noaux_tc_op` on this ungrouped routing: same sets, weights to 6e-8."""
        rows = int(logits.shape[0])
        s = self._moe_slots[slot]
        k = self._moe_k
        if rows <= _DECODE_MAX_ROWS:
            return cute_dsl_moe_topk_routing(
                logits,
                m.router_bias,
                self.expert_offset,
                self.routed_scale,
                s.topk_out[rows],
                None,
                k.topk[rows],
            )
        if k.topk_pre is None or rows > _MOE_MAX_T:
            return None
        if not in_slot:
            lg = s.logits[:rows].view(rows, -1)
            lg.copy_(logits.reshape(rows, -1))
            logits = lg
        outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_scratch, **s.maps}
        return cute_dsl_moe_topk_routing(
            logits, m.router_bias, self.expert_offset, self.routed_scale, outputs, None, k.topk_pre
        )

    def _moe_experts(
        self, x, topk_ids, topk_w, topk_maps, m, rows: int, prequantized, join, began, acc
    ):
        """One chunk's routed round trip; returns the slot's output rows.

        `topk_maps`: the maps a top-k kernel wrote for these ids, with `topk_w` in its own
        buffer; trusted only at an unpadded row count (maps built for another row count are
        an illegal address in the grouped GEMMs, not a wrong answer). `prequantized`: the
        NVFP4 input already sits in `xq` / `xsf`. `began`: `_moe_begin_layer` advanced the
        slot; otherwise this call uses the current slot and advances it after. `acc`: the
        output already holds the shared expert's partial. `join`: the stream writing that
        partial, waited on between FC1 and FC2 so the shared branch overlaps router, top-k
        and FC1.
        """
        k = self._moe_k
        dev = x.device
        # Pad to a power of two below 513 rows: the routing schedule is keyed by
        # `ceil(rows / 32)` there, and arbitrary widths would compile about twenty. Padding
        # rows route to a non-local expert at weight 0 (`pad_expert_id`), so routing drops
        # them, and are sliced off. The prefill pair takes 513 rows and up exactly.
        use_pair = self._moe_pair_ok(rows)
        padded = self._moe_padded_rows(rows)
        if padded != rows:
            pad = padded - rows
            if not prequantized:
                x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            pad_id = _moe_routing_op.pad_expert_id(self.expert_offset)
            topk_ids = torch.cat(
                [topk_ids, torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev)]
            )
            topk_w = torch.cat(
                [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
            )
        slot = self._moe_slot
        if not began:
            self._moe_slot ^= 1
        s = self._moe_slots[slot]
        if not prequantized:
            x_s = s.x[:padded]
            x_s.copy_(x)
            x = x_s
        # fp32, as both top-k kernels write it and FC2 reads it: a top-k kernel's own output
        # at an unpadded width goes in as it is.
        if topk_maps is None or padded != rows:
            w_s = s.w[:padded]
            w_s.copy_(topk_w)
            topk_w = w_s
        if topk_maps is not None and padded == rows:
            maps = topk_maps
        else:
            # Always staged: the routing kernels launch with PDL and do not wait for their
            # inputs, and the fused top-k signals its dependents before it writes the ids.
            # The copy puts a node no PDL edge can cross between the two.
            sel = s.sel[: padded * _MOE_TOP_K]
            sel.copy_(topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, self.expert_offset, s.maps, s.counts, s.barrier, k.routing
            )
        xq, xsf = self._moe_xq[:padded], self._moe_xsf[:padded]
        if prequantized:
            # Written by the collective; the padding rows are stale by design: routed to a
            # non-local expert, no grouped GEMM tile reads them.
            pass
        elif padded <= _DECODE_MAX_ROWS:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant[padded])
        elif k.quant_pre is not None and padded <= _MOE_MAX_T:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant_pre)
        else:
            xq, xsf = fp4_quantize(x, m.g1, _MOE_SF_VEC, False, False)
        # bf16: FC2 writes through a BFloat16 pointer whatever buffer it is handed.
        out = s.out[:padded]
        if not acc:
            if padded <= _DECODE_MAX_ROWS:
                cute_dsl_moe_zero(out, k.zero[padded])
            elif k.zero_pre is not None and padded <= _MOE_MAX_T:
                cute_dsl_moe_zero(out, k.zero_pre)
            else:
                out.zero_()
        compiled, c = (k.pair, s.pc) if use_pair else (k.experts, s.c)
        cute_dsl_moe_nvfp4_experts(
            xq, xsf, maps, topk_w, m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s,
            m.norm_const, m.alpha1, m.alpha2, out, c, s.sfc, join, compiled,
        )  # fmt: skip
        return out[:rows]

    def _routed(self, x, m, rows: int, prequantized: bool, join, began: bool, acc: bool):
        """This rank's 64-expert window. Router and top-k run on replicated hidden states,
        so every rank selects the same experts and the four windows tile the routing space
        once. No stock fallback for the grouped GEMMs: the weights are packed for these
        kernels alone.

        Each chunk's result is a view of a slot's output buffer, and the slots alternate per
        chunk, so with more than one chunk each result is copied out before a later chunk
        reuses its slot. A single chunk returns the view itself."""
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            slot = self._moe_slot
            # fp32 logits: FC2 reads `token_final_scales` as fp32 and noaux_tc_op's weight
            # dtype follows its logits. bf16 runs clean on garbage.
            logits = self._moe_router(chunk, m, slot)
            in_slot = logits is not None
            if logits is None:
                logits = self._project("router", chunk, m.router_t, torch.float32)
            selected = self._moe_topk(logits, in_slot, m, slot)
            if selected is None:
                topk_w, topk_ids = noaux_tc_op(
                    logits,
                    m.router_bias,
                    self.n_group,
                    self.topk_group,
                    self.topk,
                    self.routed_scale,
                )
                maps = None
            else:
                topk_w, topk_ids, maps = selected
            part = self._moe_experts(
                chunk, topk_ids, topk_w, maps, m, n, prequantized, join, began, acc
            )
            if result is None:
                return part
            result[start : start + n].copy_(part)
            start += n
        return result

    def _mlp_and_close(self, attn_out, residual, i: int, rows: int):
        """o_proj, the layer's MLP, and the closing collective normed for the next layer."""
        a = self._attn[i]
        partial = self._project("o_proj", attn_out, a.o)
        gu, dn = self._mlp[i]
        m = self._moe[i]
        if m is None:
            o = self._reduce_norm(partial, residual, a.norm2)
            act = cute_dsl_dense_swiglu(self._project("dense_gate_up", o, gu), self._dense_swiglu)
            mlp_out = self._project("dense_down", act, dn)
        else:
            o, prequant = self._reduce_norm_quant(partial, residual, a.norm2, m.g1, rows)
            began = rows <= _MOE_MAX_T
            acc = self._moe_begin_layer(rows) if began else None
            main = torch.cuda.current_stream()
            side = self._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared = self._shared_mlp(o, gu, self._shared_gu_il[i], dn, out=acc, trunk=True)
            routed = self._routed(
                o, m, rows, prequant, side if acc is not None else None, began, acc is not None
            )
            if acc is None:
                main.wait_stream(side)
                merge = self._moe_k.merge
                if (
                    merge is not None
                    and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T
                    and routed.shape == shared.shape
                    and routed.is_contiguous()
                ):
                    if not shared.is_contiguous():
                        shared = shared.contiguous()
                    mlp_out = cute_dsl_moe_merge(routed, shared, merge)
                else:
                    mlp_out = torch.add(routed, shared)
            else:
                mlp_out = routed
        return self._reduce_norm(mlp_out, residual, self._next_norm[i])

    def _mtp_front(self, e, h, rows: int):
        """`eh_proj(concat(enorm(e), hnorm(h)))`. At the specialised row counts each rank norms
        one 3072-wide quarter of the concat -- ranks 0-1 halves of `e`, ranks 2-3 halves of
        `h` -- and the projection's K-split partials are summed by the collective."""
        mw = self._mtp
        k = _mtp_eh.LOCAL_INPUT_SIZE
        if rows in _mtp_eh.SUPPORTED_M_VALUES and e.is_contiguous() and h.is_contiguous():
            src, nw = (e, mw["enorm"]) if self.tp_rank < 2 else (h, mw["hnorm"])
            local = torch.empty([rows, k], dtype=e.dtype, device=e.device)
            cute_dsl_mtp_norm_shard(
                src, nw, local, (self.tp_rank % 2) * k, self.eps, mw["norm_shard"]
            )
            return self._reduce(cute_dsl_mtp_eh_proj(local, mw["eh_local"], mw["ehproj"]))
        cat = torch.cat(
            [
                flashinfer_rmsnorm(e, mw["enorm"], self.eps),
                flashinfer_rmsnorm(h, mw["hnorm"], self.eps),
            ],
            -1,
        )
        # The replicated projection: the prefill GEMM above the decode row counts, else cuBLAS.
        return self._project("mtp_eh", cat, mw["eh"])

    def _mtp_experts(self, x, topk_ids, topk_w, rows: int, frontend: bool):
        """One MTP chunk's bf16 round trip; returns the slot's output rows. With `frontend`
        the router and the fused top-k (which also clears the output) run here on the
        input in place; otherwise the caller's ids and weights select, staged with the input
        into the slot's buffers."""
        mw, k = self._mtp, self._mtp_k
        dev = x.device
        padded = _pow2(rows)
        if padded != rows:
            pad = padded - rows
            x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            if not frontend:
                # Padding rows go to a non-local expert at weight 0, see `pad_expert_id`.
                pad_id = _moe_routing_op.pad_expert_id(self.expert_offset)
                topk_ids = torch.cat(
                    [
                        topk_ids,
                        torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev),
                    ]
                )
                topk_w = torch.cat(
                    [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
                )
        slot = self._mtp_slot
        self._mtp_slot ^= 1
        s = self._mtp_slots[slot]
        out = s.out[:padded]
        if frontend:
            # The router and gather GEMM read the activation in place; weights and maps come
            # from the top-k's outputs, and the top-k zeroed the output.
            if padded <= _moe_router_op.SIMT_MAX_ROWS:
                logits = s.router_out[:padded]
            else:
                logits = s.router_partials[:padded]
            cute_dsl_moe_router(x, mw["router"], logits, self._moe_k.router[padded])
            topk_w, _, maps = cute_dsl_moe_topk_routing(
                logits,
                mw["router_bias"],
                self.expert_offset,
                self.routed_scale,
                s.topk_out[padded],
                out,
                k.topk[padded],
            )
            xs = x
        else:
            # The routing kernel does not wait before reading the ids, and `noaux_tc_op`
            # releases it before its stores are visible: these copies are what order them.
            xs = s.x[:padded]
            xs.copy_(x)
            w_s = s.w[:padded]
            w_s.copy_(topk_w)
            topk_w = w_s
            sel = s.sel[: padded * _MOE_TOP_K]
            sel.copy_(topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, self.expert_offset, s.maps, s.counts, s.barrier, self._moe_k.routing
            )
            out.zero_()
        cute_dsl_mtp_moe_experts(xs, maps, topk_w, mw["fc1"], mw["fc2"], out, s.c, k.experts)
        return out[:rows]

    def _mtp_routed(self, x, rows: int):
        """The MTP module's window, bf16. Up to 64 rows the front end runs with the experts;
        above, the stock router projection and noaux_tc_op select. With more than one
        chunk each result is copied out before a later chunk reuses its slot."""
        mw = self._mtp
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            if n <= _DECODE_MAX_ROWS:
                part = self._mtp_experts(chunk, None, None, n, frontend=True)
            else:
                logits = self._project("router", chunk, mw["router_t"], torch.float32)
                topk_w, topk_ids = noaux_tc_op(
                    logits,
                    mw["router_bias"],
                    self.n_group,
                    self.topk_group,
                    self.topk,
                    self.routed_scale,
                )
                part = self._mtp_experts(chunk, topk_ids, topk_w, n, frontend=False)
            if result is None:
                return part
            result[start : start + n].copy_(part)
            start += n
        return result

    def _mtp_close(self, attn_out, residual, rows: int):
        """The MTP layer's o_proj and bf16 MoE. Returns the un-normalized hidden state: the
        module's own head norm is applied by `shared_head`."""
        j = self.num_layers
        a = self._attn[j]
        o = self._reduce_norm(self._project("o_proj", attn_out, a.o), residual, a.norm2)
        gu, dn = self._mlp[j]
        main = torch.cuda.current_stream()
        side = self._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared = self._shared_mlp(o, gu, self._shared_gu_il[j], dn)
        routed = self._mtp_routed(o, rows)
        main.wait_stream(side)
        return torch.add(residual, self._reduce(torch.add(routed, shared)))


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes. Everything checked is fixed at engine construction, so once per target
        instance is sound, and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        return _build_step_args(md)

    def _ctx_rope_step(self, step, tc: int) -> tuple | None:
        """The context rope kernel's per-step operands: the new-row prefix sum and the cached
        lengths, int32, from the engine's int64 prefix sums. None sends the step's context rows
        to the stock rope/append op. Context steps are not captured, so a fresh pair per step
        is fine."""
        core = self.core
        ctx = step["num_contexts"]
        compiled = core._ctx_rope_compiled
        if (
            compiled is None
            or not 0 < ctx <= _ctx_rope_op.MAX_REQUESTS
            or int(step["tokens_per_block"]) != compiled.tokens_per_block
            or tc >= _ROPE_APPEND_MAX_TOKENS
        ):
            return None
        cu = step["ctx_uncached_token_indptr"][: ctx + 1].to(torch.int32)
        cached = torch.diff(step["ctx_cached_token_indptr"][: ctx + 1]).to(torch.int32)
        return cu, cached

    def _dense_causal_step(self, md, step, tc: int, skip_ctx: bool, req_idx, dev):
        """The dense-causal context attention's per-step operands, shared by every layer, or
        None when the context rows take the index path.

        Taken only when the engine skipped the indexer for the context rows: every context
        sequence is within index_topk, so the dense causal index it substituted is the whole
        window and the kernel's per-token mask reproduces it. The windows are rebuilt here
        rather than read from the metadata: the engine's `cu_seqlen_ks` are batch-flat KV
        offsets, while this kernel reads each request through its own block-table row from
        position 0, cached prefix included."""
        core = self.core
        if not tc:
            return None
        reason = (
            "selection"
            if not skip_ctx
            else "block"
            if int(step["tokens_per_block"]) != _fmha_pre_op.DENSE_CAUSAL_BLOCK
            else None
        )
        if reason is not None:
            core.dense_causal_misses[reason] = core.dense_causal_misses.get(reason, 0) + 1
            return None
        ctx = step["num_contexts"]
        host_lens = step["seq_lens"][:ctx].to(torch.int32).cpu()
        assert int(host_lens.sum()) == tc, (host_lens.tolist(), tc)
        cached = None
        if int(step["num_ctx_cached_tokens"]):
            cached = torch.diff(step["ctx_cached_token_indptr"][: ctx + 1]).to(torch.int32)
        ks, ke = _fmha_pre_op.dense_causal_bounds(
            step["seq_lens_cuda"][:ctx].to(torch.int32), cached, tc
        )
        core.dense_causal_steps += 1
        return SimpleNamespace(
            ks=ks,
            ke=ke,
            groups=_fmha_pre_op.dense_causal_groups(host_lens).to(dev),
            req_idx=req_idx[:tc],
            block_table=md.block_table,
            # One fp8 query for the step: each layer writes it and its attention reads it
            # before the next layer writes (one stream), so the bound address holds.
            quant_q=torch.empty(
                [tc, core.heads, core.lat_dim], dtype=torch.float8_e4m3fn, device=dev
            ),
            # Bound on the first layer; later layers re-bind output and layer offset.
            prepared=None,
        )

    def _attn_ctx(self, a, layer: int, q, latent, out, slots, tc: int, step, ctx_rope, dc) -> None:
        """Sparse MLA over the context rows, absorbed form, on the sm103 prefill kernel. The
        index rows are already global pool rows spanning the cached prefix, so no gather; under
        `dc` there are none, and the kernel reads the causal window through the block table.

        The order is fixed: the rope/KV append mutates `fused_q` and the pool, so the quantize
        has to read the rotated query, and nothing after the append may fall back."""
        core = self.core
        heads, dev, dt = core.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [tc, heads, core.qk_dim]), [core.nope, core.rope_dim], -1
        )
        fused_q = torch.empty([tc, heads, core.lat_dim], dtype=dt, device=dev)
        fq_nope, fq_pe = torch.split(fused_q, [core.kv_lora, core.rope_dim], -1)
        core._absorbed_bmm(
            "k_b",
            torch.transpose(q_nope, 0, 1),
            a.k_b,
            a.k_b_nk,
            torch.transpose(fq_nope, 0, 1),
        )
        quant_q = (
            dc.quant_q
            if dc is not None
            else torch.empty([tc, heads, core.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        )
        pool = step["primary_pool"]
        pool_ptrs = step["host_kv_cache_pool_pointers"]
        mapping = step["host_kv_cache_pool_mapping"][layer]
        pool_idx, layer_in_pool = int(mapping[0]), int(mapping[1])
        if (
            ctx_rope is not None
            and int(pool_ptrs[pool_idx, 0]) == pool.data_ptr()
            and int(pool_ptrs[pool_idx, 1]) == 0
        ):
            # One launch: rope(q_pe) into the tail of fused_q straight from the strided view,
            # rope(k_pe) and the latent row appended as e4m3 to this layer's pool rows, and
            # e4m3(fused_q). Context requests are the first `num_contexts` of the batch.
            cute_dsl_mla_rope_append_prefill(
                fused_q,
                q_pe,
                # A packed copy: `contiguous()` keeps a one-row view's 2624 row stride, and
                # the kernel is compiled for the packed 576. Nothing reads the latent after.
                latent.clone(memory_format=torch.contiguous_format),
                quant_q,
                pool,
                layer_in_pool,
                step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]],
                *ctx_rope,
                core._rope["rotary_cos_sin"],
                step["tokens_per_block"],
                core._ctx_rope_compiled,
            )
        else:
            # The append rotates the query's rope tail in place, reading its own
            # pre-rotation contents, and the absorbed BMM wrote only the first kv_lora
            # columns: seed the tail.
            fq_pe.copy_(q_pe)
            mla_rope_append_paged_kv_assign_q(
                torch.reshape(fused_q, [tc, heads * core.lat_dim]),
                latent.contiguous(),
                step["num_contexts"],
                step["ctx_cached_token_indptr"],
                step["ctx_kv_indptr"],
                step["max_ctx_seq_len"],
                core._rope["rotary_cos_sin"],
                heads,
                # The width ahead of each head's rope tail in the query this call hands
                # over: the absorbed head is `kv_lora + rope`, so this is kv_lora, not
                # qk_nope_head_dim.
                core.kv_lora,
                core.rope_dim,
                core.kv_lora,
                step["kv_cache_block_offsets"],
                step["host_kv_cache_pool_pointers"],
                step["host_kv_cache_pool_mapping"],
                None,
                0,
                layer,
                step["tokens_per_block"],
                step["max_seq_len"],
                1,
                core.quant_mode,
            )
            quant_q.copy_(fused_q)
        lat_out = torch.empty([tc, heads * core.kv_lora], dtype=dt, device=dev)
        if dc is not None:
            # The pool rows this layer's selection would name, `to_global` of the dense
            # causal index: the block-table row times the stride, plus this layer's offset.
            layer_offset = layer_in_pool * int(step["tokens_per_block"])
            if dc.prepared is None:
                dc.prepared = _fmha_pre_op.prepare_dense_causal(
                    core._fmha_pre_dc_compiled, quant_q, pool, lat_out, dc.ks, dc.ke,
                    core._fmha_bmm1, core._fmha_bmm2, dc.block_table, dc.req_idx,
                    core._stride_factor, layer_offset, dc.groups,
                )  # fmt: skip
            cute_dsl_sparse_mla_prefill(
                quant_q, pool, None, lat_out, dc.ks, dc.ke, core._fmha_bmm1, core._fmha_bmm2,
                compiled=dc.prepared, block_table=dc.block_table, req_idx=dc.req_idx,
                stride_factor=core._stride_factor, layer_offset=layer_offset,
                dense_causal_groups=dc.groups,
            )  # fmt: skip
        else:
            cute_dsl_sparse_mla_prefill(
                quant_q,
                pool,
                slots,
                lat_out,
                step["cu_seqlen_ks"],
                step["cu_seqlen_ke"],
                core._fmha_bmm1,
                core._fmha_bmm2,
                compiled=core._fmha_pre_compiled,
            )
        core._absorbed_bmm(
            "v_b",
            torch.transpose(torch.reshape(lat_out, [tc, heads, core.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [tc, heads, core.v_dim]), 0, 1),
        )

    def _layer_attention(
        self,
        md,
        step,
        x,
        pos,
        attn_out,
        i: int,
        pool_layer: int,
        slots_of,
        topk_buf,
        rows,
        tc,
        gen,
        gen_p,
        ctx,
        need_q,
        skip_ctx,
        skip_gen,
        ctx_rope=None,
        dc=None,
    ):
        """One layer's attention block, both phases, into `attn_out`. `slots_of(topk_buf)` maps
        the request-local selection to this layer's pool rows, `(context, generation)`; the
        context half is None under the dense-causal context attention (`dc`)."""
        core = self.core
        a = core._attn[i]
        q, q_a, latent = core._qkv(x, a)
        ix = core._idx[i]
        if ix is not None:
            q_fp8, weights, k_pre = core._indexer_project(x, q_a, pos, ix, rows, need_q)
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            core._indexer_write_k(md, k_pool, k_pre, rows)
            if tc and not skip_ctx:
                for chunk in md.indexer_prefill_chunks:
                    n_k = chunk.k_token_end - chunk.k_token_start
                    if n_k <= 0 or chunk.token_end <= chunk.token_start:
                        continue
                    if n_k > core._ix_topk_pre_compiled.max_columns:
                        # A K block wider than any built so far (a long cached prefix, or packed
                        # requests): context steps are never captured, so widen the build here.
                        core._ix_topk_pre_compiled = build_ix_topk_prefill(
                            torch.cuda.current_device(), top_k=core.idx_topk, max_columns=n_k
                        )
                    ck, cs = indexer_k_cache_gather_op(
                        k_pool,
                        md.slot_mapping_fp8_fullkv,
                        md.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        n_k,
                        core.idx_dim,
                    )
                    cs = torch.reshape(cs, [-1])
                    # The window vectors are indexed by absolute context token, not by position
                    # within the chunk.
                    for lo, hi in _tile_bounds(chunk.token_start, chunk.token_end, n_k):
                        ks, ke = chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]
                        logits = fp8_mqa_logits(q_fp8[lo:hi], (ck, cs), weights[lo:hi], ks, ke)
                        cute_dsl_indexer_topk_prefill(
                            logits, ks, ke, topk_buf[lo:hi],
                            top_k=core.idx_topk, compiled=core._ix_topk_pre_compiled,
                        )  # fmt: skip
            if gen and not skip_gen:
                core._indexer_decode(
                    md, k_pool, q_fp8[tc:], weights[tc:], topk_buf[tc:], ctx, gen // gen_p, gen_p
                )
        slots_ctx, slots_gen = slots_of(topk_buf)
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        lat_ctx, lat_gen = torch.split(latent, [tc, gen], 0)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        if tc:
            self._attn_ctx(
                a, pool_layer, q_ctx, lat_ctx, attn_ctx, slots_ctx, tc, step, ctx_rope, dc
            )
        if gen:
            core._attn_gen(
                a, pool_layer, q_gen, lat_gen, attn_gen, slots_gen, gen, gen_p, ctx, md,
                INDEX_KIND_GLOBAL, None, False,
            )  # fmt: skip

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
        md = attn_metadata
        step = self.step_args(md)
        h = core._embed(input_ids) if inputs_embeds is None else inputs_embeds
        rows = h.shape[0]
        # The slot mappings and window vectors are indexed by the prepared token order; a
        # padded input would scatter into stale slots.
        assert rows == md.num_tokens, (rows, md.num_tokens)
        assert rows < _STEP_MAX_TOKENS, (
            f"{rows} tokens in one step overflow the indexer kernel's int32 offsets; "
            f"set max_num_tokens below {_STEP_MAX_TOKENS}"
        )
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = core._topk_buf(md, rows, tc, skip_ctx, skip_gen)
        need_q = not (skip_ctx and skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        # Both built once per step and shared by every layer; None takes the stock rope/append
        # op, or the index path of the context attention.
        ctx_rope = self._ctx_rope_step(step, tc) if tc else None
        dc = self._dense_causal_step(md, step, tc, skip_ctx, req_idx, h.device)
        # Under `dc` the context rows' pool rows are formed inside the attention kernel, so the
        # conversion runs on the generation rows alone, or not at all.
        slot_lo = tc if dc is not None else 0

        x = flashinfer_rmsnorm(h, core._attn[0].norm1, core.eps)
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, layer_id=layer_id):
                if slot_lo == rows:
                    return None, None
                slots = cute_dsl_req_index_to_global(
                    req_idx[slot_lo:], md.block_table, buf[slot_lo:], md.tokens_per_block,
                    core._stride_factor, layer_id, compiled=core._req_global_compiled,
                )  # fmt: skip
                return (None, slots) if slot_lo else torch.split(slots, [tc, gen], 0)

            self._layer_attention(
                md, step, x, pos, attn_out, i, i, slots_of, topk_buf, rows, tc, gen, gen_p, ctx,
                need_q, skip_ctx, skip_gen, ctx_rope, dc,
            )  # fmt: skip
            x = core._mlp_and_close(attn_out, residual, i, rows)
        return x

    def mtp_forward(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        position_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step on the MTP module. Its KV lives in the separate draft pool: the
        pool-local layer index and the pool stride are read live, not from the trunk's."""
        core = self.core
        md = attn_metadata
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        j = core.num_layers
        emb = core._embed(input_ids)
        x = core._mtp_front(emb, hidden_states, rows)
        residual = x
        xn = flashinfer_rmsnorm(x, core._attn[j].norm1, core.eps)
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = core._topk_buf(md, rows, tc, skip_ctx, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]

        def slots_of(buf):
            slots = convert_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, core.idx_topk, stride, pool_layer
            )
            return torch.split(slots, [tc, gen], 0)

        # The MTP layer's context rows take the stock rope/append op and the index path.
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        self._layer_attention(
            md, step, xn, torch.reshape(position_ids, [-1]), attn_out, j, local, slots_of,
            topk_buf, rows, tc, gen, gen_p, ctx, not (skip_ctx and skip_gen), skip_ctx, skip_gen,
        )  # fmt: skip
        return core._mtp_close(attn_out, residual, rows)


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Two paths only this target takes, both at the MTP width of 2: the specialised MLA rope
    kernel, and pool rows computed once per selection rather than once per layer (the decode
    kernel adds `layer * 32` itself).
    """

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes. Everything checked is fixed at engine construction, so once per target
        instance is sound, and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """States the two context counts instead of reading them back: this target runs only
        when `num_contexts` is zero."""
        step = _build_step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

    def _layer_attention(
        self,
        md,
        x,
        pos,
        attn_out,
        i: int,
        pool_layer: int,
        slots_of,
        topk_buf,
        rows,
        gen_p,
        skip_gen,
        specialised_rope,
    ):
        core = self.core
        a = core._attn[i]
        q, q_a, latent = core._qkv(x, a)
        ix = core._idx[i]
        if ix is not None:
            q_fp8, weights, k_pre = core._indexer_project(x, q_a, pos, ix, rows, not skip_gen)
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            core._indexer_write_k(md, k_pool, k_pre, rows)
            if not skip_gen:
                core._indexer_decode(md, k_pool, q_fp8, weights, topk_buf, 0, rows // gen_p, gen_p)
        slots, slot_kind, slot_layer = slots_of(topk_buf, ix is not None)
        core._attn_gen(
            a, pool_layer, q, latent, attn_out, slots, rows, gen_p, 0, md, slot_kind, slot_layer,
            specialised_rope,
        )  # fmt: skip

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
        md = attn_metadata
        h = core._embed(input_ids) if inputs_embeds is None else inputs_embeds
        rows = h.shape[0]
        assert rows == md.num_tokens, (rows, md.num_tokens)
        gen_p = rows // md.num_seqs
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = core._topk_buf(md, rows, 0, True, skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        mtp_width = gen_p == _MLA_ROPE_MTP_WIDTH
        base = None

        x = flashinfer_rmsnorm(h, core._attn[0].norm1, core.eps)
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, fresh, layer_id=layer_id):
                nonlocal base
                if not mtp_width:
                    rows_ = cute_dsl_req_index_to_global(
                        req_idx, md.block_table, buf, md.tokens_per_block, core._stride_factor,
                        layer_id, compiled=core._req_global_compiled,
                    )  # fmt: skip
                    return rows_, INDEX_KIND_GLOBAL, None
                # Layer-independent rows, recomputed only when a layer selected afresh.
                if fresh or base is None:
                    base = cute_dsl_req_index_to_base(
                        req_idx, md.block_table, buf, md.tokens_per_block, core._stride_factor,
                        compiled=core._req_base_compiled,
                    )  # fmt: skip
                return base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id

            self._layer_attention(
                md, x, pos, attn_out, i, i, slots_of, topk_buf, rows, gen_p, skip_gen, mtp_width
            )
            x = core._mlp_and_close(attn_out, residual, i, rows)
        return x

    def mtp_forward(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        position_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step, decode rows only. Pool rows go through the stock conversion against
        the draft pool's live mapping."""
        core = self.core
        md = attn_metadata
        rows = hidden_states.shape[0]
        gen_p = rows // md.num_seqs
        j = core.num_layers
        emb = core._embed(input_ids)
        x = core._mtp_front(emb, hidden_states, rows)
        residual = x
        xn = flashinfer_rmsnorm(x, core._attn[j].norm1, core.eps)
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = core._topk_buf(md, rows, 0, True, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]

        def slots_of(buf, fresh):
            rows_ = convert_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, core.idx_topk, stride, pool_layer
            )
            return rows_, INDEX_KIND_GLOBAL, None

        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        self._layer_attention(
            md, xn, torch.reshape(position_ids, [-1]), attn_out, j, local, slots_of, topk_buf,
            rows, gen_p, skip_gen, gen_p == _MLA_ROPE_MTP_WIDTH,
        )  # fmt: skip
        return core._mtp_close(attn_out, residual, rows)


class MTPLayer:
    """The checkpoint's MTP module at layer `num_hidden_layers`, replayed once per draft step.

    It differs from a trunk layer in three ways: an `eh_proj` front end mixing the next token's
    embedding into `h`, bf16 experts, and an output returned un-normalized (`shared_head`
    applies the module's own norm and the runtime feeds the un-normalized tensor back in).
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
        """The module's own norm (a distinct parameter from `model.norm`), then the trunk's
        `lm_head`."""
        core = self.core
        return self.logits_processor.forward(
            flashinfer_rmsnorm(hidden_states, core._mtp["head_norm"], core.eps),
            lm_head,
            attn_metadata,
            return_context_logits,
        )

    def forward(
        self, embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
    ):
        """Route the draft step by the phase this call's metadata shows: the worker rewrites the
        metadata in place between draft steps, so one engine step's draft calls can cross from
        prefill to decode."""
        return self.core._select_target(attn_metadata).mtp_forward(
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
        logits = self.logits_processor.forward(
            nn.functional.embedding(spec_metadata.gather_ids, hidden),
            self.lm_head,
            attn_metadata,
            True,
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
