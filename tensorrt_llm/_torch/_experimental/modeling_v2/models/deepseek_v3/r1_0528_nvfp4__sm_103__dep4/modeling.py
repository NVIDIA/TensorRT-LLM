# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-r1-0528-nvfp4 / sm_103 / dep4 — self-contained modeling code.

DeepSeek-R1-0528, modelopt NVFP4 export: 61 layers, hidden 7168, 128 query heads, layers 0-2
dense and 3-60 MoE (256 routed experts at top-8 plus a shared expert). Only the MLP path is
NVFP4; attention, router, embedding and lm_head are bf16, and the KV cache is fp8-e4m3. Layer
61 is the checkpoint's bf16 MTP module (`MTPLayer`), live only under a `speculative_config`.
`dep4` is attention DP with the 256 experts split four ways, so only the MoE needs a collective.
"""

import math
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import advance_step_generation
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (  # noqa: E501
    LoadPagedKvCacheForMla,
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    MlaRopeAppendPagedKvAssignQ,
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    MlaRopeGeneration,
    mla_rope_generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    ThopAttention,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import (
    Allgather,
    allgather,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import (
    Reducescatter,
    reducescatter,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import BmmOut, bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm, cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import Nvfp4Gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    Fp4BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import (
    NoauxTcOp,
    noaux_tc_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    FlashinferFusedAddRmsnorm,
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    FlashinferRmsnorm,
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    register_auto_model,
)
from tensorrt_llm._torch.speculative import get_spec_worker

from . import weights as _weights

_CACHED_CTX_FIELDS = (
    "enable_context_mla_with_cached_kv",
    "ctx_cached_token_indptr",
    "ctx_kv_indptr",
    "max_ctx_seq_len",
    "max_ctx_kv_len",
)


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state both MLA attention calls of one
    forward share."""
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
        workspace_=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        attention_window_size=md.max_seq_len,
        rope_max_positions=md.max_seq_len,
        rope_original_max_positions=md.max_seq_len,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


_KV_RESIDUAL_DIM = 0

# Both kv scale tensors stay None: None is read as a scale of exactly 1.0, which is what this
# checkpoint's k_scale/v_scale hold and the only value the context flavors are correct at.
# The MLA path reads the rope table and nothing else — rope_scale_type, rope_scale, the two
# m-scales and the two position windows are silently ignored, so the YaRN blend must live in
# the table.
_CALL_INERT = dict(
    output_sf=None,
    out_scale=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    update_kv_cache=True,
    beam_width=1,
    mask_type=1,
    use_paged_context_fmha=False,
    is_mla_enable=True,
    rope_append=True,
    position_embedding_type=8,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    cache_indirection=None,
    block_ids_per_seq=None,
    max_context_q_len_override=None,
    is_cross=False,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets_for_cpp=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_decoding_target_max_draft_tokens=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    num_sparse_topk=None,
    flash_mla_tile_scheduler_metadata=None,
    flash_mla_num_splits=None,
    # kv_norm_weight must stay None: a weight here folds kv_a_layernorm into the KV kernel,
    # which this target already applies itself — a silent double normalization.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

_QUANT_MODE_FP8_KV = 128

_SF_VEC = 16
# The MoE runner reads the activation scales as a linear buffer and the dense GEMM reads the
# 128x4-swizzled one; the two have the same byte count whenever num_tokens is a multiple of 128
# — every decode CUDA-graph batch — and the wrong one is then accepted silently.
_SF_LINEAR = False
_SF_SWIZZLED = True
_ROUTING_METHOD_INERT = 1
_MOE_MAX_T = 8192
_ACT_TYPE_SWIGLU = 0
# A different enum from the trtllm-gen runner's above: 5 is Swiglu here, 0 is not a gated type.
_FUSED_MOE_ACT_SWIGLU = 5
# True = `concat(enorm(e), hnorm(h))`, which contradicts the DeepSeek-V3 report's
# `M_k[RMSNorm(h); RMSNorm(e)]`; a flip is silent, only the draft acceptance rate shows it.
_MTP_EMBED_BLOCK_FIRST = True


def _moe_chunk_sizes(total: int) -> list[int]:
    """Split `total` gathered rows into expert-call chunks of at most `_MOE_MAX_T` rows."""
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


def _mtp_dp_rows(all_rank_num_tokens, rank: int, dp_size: int, rows: int) -> int:
    """The uniform row count the MTP layer pads its token block to before its MoE round trip.

    Callers must pass the worker's own basis, never `attn_metadata.all_rank_num_tokens`, which
    from draft step 1 onwards is still the trunk's list: at equal byte counts the collectives
    diverge silently on every rank, with no hang."""
    counts = [int(n) for n in all_rank_num_tokens]
    return max(counts)


def _yarn_mscale(factor: float, mscale: float) -> float:
    """DeepSeek's YaRN magnitude scaling `m(x)`."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


class DeepseekV3ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it, and False for the
        # whole life of a served engine: everything the check looks at is fixed
        # at engine construction, so it runs once.
        self._contract_pending = step_contract_enabled()

        mapping = model_config.mapping
        self.rank = mapping.rank
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.dp_group = list(range(mapping.world_size))
        self.dp_size = mapping.world_size

        dt = model_config.torch_dtype
        self.quant_mode = _QUANT_MODE_FP8_KV

        num_layers = cfg.num_hidden_layers
        hidden = cfg.hidden_size
        vocab = cfg.vocab_size
        self.mtp_layers = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None

        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim

        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = rope_cfg.get("rope_theta", getattr(cfg, "rope_theta", None))
        self.theta = float(theta)
        self.rope_factor = float(rope_cfg["factor"])
        self.rope_orig_max = int(rope_cfg["original_max_position_embeddings"])
        self.beta_fast = float(rope_cfg["beta_fast"])
        self.beta_slow = float(rope_cfg["beta_slow"])
        mscale = float(rope_cfg["mscale"])
        mscale_all_dim = float(rope_cfg["mscale_all_dim"])
        # YaRN's magnitude correction splits in two: the amplitude multiplies cos/sin, the
        # temperature folds into q_scaling as 1/m^2. Either in the other's place is silently wrong.
        self.rope_amplitude = _yarn_mscale(self.rope_factor, mscale) / _yarn_mscale(
            self.rope_factor, mscale_all_dim
        )
        temperature = _yarn_mscale(self.rope_factor, mscale_all_dim)
        self.q_scaling = 1.0 / (temperature * temperature)
        self._call = dict(_CALL_INERT, quant_mode=self.quant_mode, q_scaling=self.q_scaling)

        dense_layers = cfg.first_k_dense_replace
        num_experts = cfg.n_routed_experts
        moe_inter = cfg.moe_intermediate_size
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        dense_inter = cfg.intermediate_size
        self.local_experts = num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank

        def P(*shape, dtype=dt):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        u8, f32 = torch.uint8, torch.float32
        w = nn.ParameterDict()
        for i in range(num_layers):
            w[f"l{i}_norm1"] = P(hidden)
            w[f"l{i}_qa"] = P(q_lora, hidden)
            w[f"l{i}_q_norm"] = P(q_lora)
            w[f"l{i}_qb"] = P(heads * qk_dim, q_lora)
            w[f"l{i}_kva"] = P(lat_dim, hidden)
            w[f"l{i}_kv_norm"] = P(kv_lora)
            w[f"l{i}_kvb"] = P(heads * (nope + v_dim), kv_lora)
            w[f"l{i}_o"] = P(hidden, heads * v_dim)
            w[f"l{i}_k_scale"] = P(1, dtype=f32)
            w[f"l{i}_v_scale"] = P(1, dtype=f32)
            w[f"l{i}_norm2"] = P(hidden)
            inter = dense_inter if i < dense_layers else shared_inter
            w[f"l{i}_mlp_gu_w"] = P(2 * inter, hidden // 2, dtype=u8)
            w[f"l{i}_mlp_gu_s"] = P(2 * inter * (hidden // _SF_VEC), dtype=u8)
            w[f"l{i}_mlp_dn_w"] = P(hidden, inter // 2, dtype=u8)
            w[f"l{i}_mlp_dn_s"] = P(hidden * (inter // _SF_VEC), dtype=u8)
            for name in (
                "isc1",
                "isc1_up",
                "ws2_1",
                "ws2_1_up",
                "isc2",
                "ws2_2",
            ):
                w[f"l{i}_mlp_{name}"] = P(1, dtype=f32)
            if i < dense_layers:
                continue
            e, mi = self.local_experts, moe_inter
            w[f"l{i}_router"] = P(num_experts, hidden)
            w[f"l{i}_router_bias"] = P(num_experts, dtype=f32)
            w[f"l{i}_fc1_w"] = P(e, 2 * mi, hidden // 2, dtype=u8)
            w[f"l{i}_fc1_s"] = P(e, 2 * mi, hidden // _SF_VEC, dtype=u8)
            w[f"l{i}_fc2_w"] = P(e, hidden, mi // 2, dtype=u8)
            w[f"l{i}_fc2_s"] = P(e, hidden, mi // _SF_VEC, dtype=u8)
            for name in (
                "isc1",
                "isc1_up",
                "ws2_1",
                "ws2_1_up",
                "isc2",
                "ws2_2",
            ):
                w[f"l{i}_e_{name}"] = P(num_experts, dtype=f32)
        w["final_norm"] = P(hidden)
        w["embed"] = P(vocab, hidden)
        if self.mtp_enabled:
            e, mi = self.local_experts, moe_inter
            w["mtp_enorm"] = P(hidden)
            w["mtp_hnorm"] = P(hidden)
            w["mtp_eh"] = P(hidden, 2 * hidden)
            w["mtp_norm1"] = P(hidden)
            w["mtp_qa"] = P(q_lora, hidden)
            w["mtp_q_norm"] = P(q_lora)
            w["mtp_qb"] = P(heads * qk_dim, q_lora)
            w["mtp_kva"] = P(lat_dim, hidden)
            w["mtp_kv_norm"] = P(kv_lora)
            w["mtp_kvb"] = P(heads * (nope + v_dim), kv_lora)
            w["mtp_o"] = P(hidden, heads * v_dim)
            w["mtp_k_scale"] = P(1, dtype=f32)
            w["mtp_v_scale"] = P(1, dtype=f32)
            w["mtp_norm2"] = P(hidden)
            w["mtp_router"] = P(num_experts, hidden)
            w["mtp_router_bias"] = P(num_experts, dtype=f32)
            w["mtp_fc1"] = P(e, 2 * mi, hidden)
            w["mtp_fc2"] = P(e, hidden, mi)
            w["mtp_sh_gu"] = P(2 * shared_inter, hidden)
            w["mtp_sh_dn"] = P(hidden, shared_inter)
            w["mtp_head_norm"] = P(hidden)
        self.w = w

        self._mtp: dict | None = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._side_stream: torch.cuda.Stream | None = None
        self._cached_ctx = False

    def _rope_tables(self, device, positions: int) -> dict:
        """The duplicated-layout GPT-J rope table the MLA ops read, plus its inverse-frequency
        vector.

        `positions` is a row count, not a model property: every row depends only on its own
        index, so rebuilding at a larger size changes no existing row."""
        rope_dim = self.model_config.pretrained_config.qk_rope_head_dim
        half = rope_dim // 2
        d = torch.arange(half, dtype=torch.float64)
        freq = self.theta ** (2.0 * d / rope_dim)
        two_pi = 2.0 * math.pi
        log_theta = math.log(self.theta)
        low = max(
            0.0,
            math.floor(
                rope_dim
                * math.log(self.rope_orig_max / (self.beta_fast * two_pi))
                / (2.0 * log_theta)
            ),
        )
        high = min(
            rope_dim - 1.0,
            math.ceil(
                rope_dim
                * math.log(self.rope_orig_max / (self.beta_slow * two_pi))
                / (2.0 * log_theta)
            ),
        )
        ramp = ((d - low) / max(high - low, 0.001)).clamp(0.0, 1.0)
        inv = ramp / (self.rope_factor * freq) + (1.0 - ramp) / freq
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos() * self.rope_amplitude, ang.sin() * self.rope_amplitude
        table = torch.empty(positions, rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        return {
            "rotary_cos_sin": table.reshape(1, positions * rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": rope_dim,
            "rope_base": self.theta,
        }

    def derive_after_load(self) -> None:
        """Bind every op this target calls, once, against the real weights."""
        cfg = self.model_config.pretrained_config
        num_layers = cfg.num_hidden_layers
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        eps = cfg.rms_norm_eps
        dense_layers = cfg.first_k_dense_replace
        num_experts = cfg.n_routed_experts
        topk = cfg.num_experts_per_tok
        moe_inter = cfg.moe_intermediate_size

        w = self.w
        device = w["final_norm"].device
        self._rope_positions = cfg.max_position_embeddings
        self._rope = self._rope_tables(device, self._rope_positions)
        window = slice(self.expert_offset, self.expert_offset + self.local_experts)

        self._rms0 = FlashinferRmsnorm()
        self._rms0.bind_const(weight=w["l0_norm1"], eps=eps)

        self._qa = CublasMm()
        self._q_lora_norm = FlashinferRmsnorm()
        self._q_lora_norm.bind_const(eps=eps)
        self._qb = CublasMm()
        self._kva = CublasMm()
        self._kv_lora_norm = FlashinferRmsnorm()
        self._kv_lora_norm.bind_const(eps=eps)
        self._kvb = CublasMm()
        self._o = CublasMm()
        self._norm2 = FlashinferFusedAddRmsnorm()
        self._norm2.bind_const(eps=eps)
        self._next_norm = FlashinferFusedAddRmsnorm()
        self._next_norm.bind_const(eps=eps)
        self._absorb = BmmOut()
        self._expand = BmmOut()

        # In MLA, TrtllmAttention serves context_only or generation_only, never a mixed batch,
        # which is why the two phases are separate instances and separate calls in `forward`:
        # context materializes explicit K/V, generation works in latent space via absorption.
        self._attn_ctx = ThopAttention()
        self._attn_ctx.bind_const(
            num_heads=heads,
            num_kv_heads=heads,
            head_size=qk_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=v_dim,
            is_fused_qkv=False,
            attention_input_type=1,
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
            predicted_tokens_per_seq=1,
            **_CALL_INERT,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
        )
        self._attn_gen = ThopAttention()
        self._attn_gen.bind_const(
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            is_fused_qkv=True,
            attention_input_type=2,
            k=None,
            v=None,
            **_CALL_INERT,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
        )

        self._mla_append = MlaRopeAppendPagedKvAssignQ()
        self._mla_append.bind_const(
            head_num=heads,
            nope_size=nope,
            rope_size=rope_dim,
            lora_size=kv_lora,
            kv_scale_orig_quant=None,
            residual_dim=_KV_RESIDUAL_DIM,
            beam_width=1,
            quant_mode=self.quant_mode,
        )
        self._mla_load = LoadPagedKvCacheForMla()
        self._mla_load.bind_const(
            kv_scale_quant_orig=None,
            kv_lora_rank=kv_lora,
            qk_rope_head_dim=rope_dim,
            beam_width=1,
            quant_mode=self.quant_mode,
        )
        self._mla_gen = MlaRopeGeneration()
        self._mla_gen.bind_const(
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            kv_cache_scale_orig_quant=None,
            out_scale=None,
            block_ids_per_seq=None,
            helix_tensor_params=[None, None],
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            residual_dim=_KV_RESIDUAL_DIM,
            beam_width=1,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=kv_lora,
            rope_append=True,
        )

        self._allgather = Allgather()
        self._allgather.bind_const(sizes=None, group=self.dp_group)
        self._reducescatter = Reducescatter()
        self._reducescatter.bind_const(sizes=None, group=self.dp_group)

        self._noaux = NoauxTcOp()
        self._noaux.bind_const(
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            topk=topk,
            routed_scaling_factor=float(cfg.routed_scaling_factor),
        )
        self._router = CublasMm()

        self._mlp_gu_quant = Fp4Quantize()
        self._mlp_gu_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_SWIZZLED
        )
        self._mlp_gu_gemm = Nvfp4Gemm()
        self._mlp_dn_quant = Fp4Quantize()
        self._mlp_dn_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_SWIZZLED
        )
        self._mlp_dn_gemm = Nvfp4Gemm()

        self._moe_quant = Fp4Quantize()
        self._moe_quant.bind_const(
            sf_vec_size=_SF_VEC, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_LINEAR
        )
        self._moe_runner = Fp4BlockScaleMoeRunner()
        self._moe_runner.bind_const(
            routing_logits=None,
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_bias=None,
            num_experts=num_experts,
            top_k=topk,
            n_group=None,
            topk_group=None,
            intermediate_size=moe_inter,
            local_expert_offset=self.expert_offset,
            local_num_experts=self.local_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_INERT,
            do_finalize=True,
            act_type=_ACT_TYPE_SWIGLU,
        )

        hn = heads * nope
        for i in range(num_layers):
            kvb = w[f"l{i}_kvb"]
            self._qa.bind_layered(i, mat_b=w[f"l{i}_qa"].t())
            self._q_lora_norm.bind_layered(i, weight=w[f"l{i}_q_norm"])
            self._qb.bind_layered(i, mat_b=w[f"l{i}_qb"].t())
            self._kva.bind_layered(i, mat_b=w[f"l{i}_kva"].t())
            self._kv_lora_norm.bind_layered(i, weight=w[f"l{i}_kv_norm"])
            self._kvb.bind_layered(i, mat_b=kvb.t())
            self._o.bind_layered(i, mat_b=w[f"l{i}_o"].t())
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < num_layers else w["final_norm"]
            self._next_norm.bind_layered(i, weight=next_w)
            self._absorb.bind_layered(i, b=kvb[:hn].reshape(heads, nope, kv_lora))
            self._expand.bind_layered(
                i, b=torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2)
            )
            self._attn_ctx.bind_layered(i, local_layer_idx=i)
            self._attn_gen.bind_layered(i, local_layer_idx=i)
            self._mla_append.bind_layered(i, layer_idx=i)
            self._mla_load.bind_layered(i, layer_idx=i)
            self._mla_gen.bind_layered(i, layer_idx=i)

            # The checkpoint stores reciprocals (`input_scale` = 1/g_act, `weight_scale_2` =
            # 1/g_w), so the quantizer's global scale is 1/input_scale and the GEMM alpha is
            # their product — no further reciprocal anywhere.
            isc1 = w[f"l{i}_mlp_isc1"]
            isc2 = w[f"l{i}_mlp_isc2"]
            self._mlp_gu_quant.bind_layered(i, global_scale=(1.0 / isc1).contiguous())
            self._mlp_gu_gemm.bind_layered(
                i,
                weight=w[f"l{i}_mlp_gu_w"],
                weight_scale=w[f"l{i}_mlp_gu_s"],
                alpha=(isc1 * w[f"l{i}_mlp_ws2_1"]).contiguous(),
            )
            self._mlp_dn_quant.bind_layered(i, global_scale=(1.0 / isc2).contiguous())
            self._mlp_dn_gemm.bind_layered(
                i,
                weight=w[f"l{i}_mlp_dn_w"],
                weight_scale=w[f"l{i}_mlp_dn_s"],
                alpha=(isc2 * w[f"l{i}_mlp_ws2_2"]).contiguous(),
            )

            if i < dense_layers:
                continue
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t())
            self._noaux.bind_layered(i, bias=w[f"l{i}_router_bias"])
            # One quantization of the gathered hidden states feeds every expert on every rank,
            # so the routed FC1 activation scale has to be a single value and the same value on
            # all four ranks, or the windows no longer sum to the whole layer; the shared
            # expert's `input_scale` is the max over all routed ones, so it is the safe one.
            e_isc2 = w[f"l{i}_e_isc2"][window]
            gate1 = (w[f"l{i}_mlp_isc1"][0] * w[f"l{i}_e_ws2_1"][window]).contiguous()
            self._moe_quant.bind_layered(i, global_scale=(1.0 / w[f"l{i}_mlp_isc1"]).contiguous())
            self._moe_runner.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                output1_scale_scalar=(gate1 / e_isc2).contiguous(),
                output1_scale_gate_scalar=gate1,
                output2_scale_scalar=(e_isc2 * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            )

        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
        self._side_stream = torch.cuda.Stream(device=device)

    def _derive_mtp(self) -> dict:
        """The MTP layer's operand set, derived exactly as a trunk layer's is."""
        cfg = self.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        w = self.w
        hn = heads * nope
        kvb = w["mtp_kvb"]
        return {
            "enorm": w["mtp_enorm"],
            "hnorm": w["mtp_hnorm"],
            "eh": w["mtp_eh"].t(),
            "norm1": w["mtp_norm1"],
            "qa": w["mtp_qa"].t(),
            "q_norm": w["mtp_q_norm"],
            "qb": w["mtp_qb"].t(),
            "kva": w["mtp_kva"].t(),
            "kv_norm": w["mtp_kv_norm"],
            "k_b": kvb[:hn].reshape(heads, nope, kv_lora),
            "v_b_t": torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2),
            "kvb": kvb.t(),
            "o": w["mtp_o"].t(),
            "norm2": w["mtp_norm2"],
            "router": w["mtp_router"].t(),
            "router_bias": w["mtp_router_bias"],
            "fc1": w["mtp_fc1"],
            "fc2": w["mtp_fc2"],
            "sh_gu": w["mtp_sh_gu"].t(),
            "sh_dn": w["mtp_sh_dn"].t(),
            "head_norm": w["mtp_head_norm"],
        }

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it.

        Calling the projection is the check: it reads every metadata field this model
        consumes, so a rename or removal of that private trtllm surface surfaces here
        rather than mid-forward. A hand-kept list of the same names would drift
        silently the first time `_build_step_args` gains a field.
        """
        if not self._contract_pending:
            return
        try:
            _build_step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._after_contract_check(md)
        self._contract_pending = False

    def _after_contract_check(self, md) -> None:
        """Grow the rope table to cover every position the engine admits, and fix the context
        flavor this engine construction runs."""
        # A table shorter than the engine's max_seq_len is read out of bounds with no check, and
        # `rope_max_positions` is one of the arguments the MLA path ignores. `speculative_config`
        # inflates max_seq_len past `max_position_embeddings`, so take the engine's own number.
        if md.max_seq_len > self._rope_positions:
            self._rope_positions = md.max_seq_len
            self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
        self._cached_ctx = all(hasattr(md, n) for n in _CACHED_CTX_FIELDS) and bool(
            md.enable_context_mla_with_cached_kv
        )

    def _dp_rows(self, md, num_tokens: int) -> int:
        """The uniform row count every rank pads its token block to before the MoE round trip."""
        counts = [int(n) for n in md.all_rank_num_tokens]
        return max(counts)

    def _dense_mlp(self, x, layer: int, dt):
        """One NVFP4 SwiGLU MLP over this rank's own tokens, used for both the dense layers' MLP
        and every MoE layer's shared expert."""
        xq, xsf = self._mlp_gu_quant(x, layer=layer)
        gu = self._mlp_gu_gemm(act_fp4=xq, act_sf=xsf, output_dtype=dt, layer=layer)
        act = flashinfer_silu_and_mul(gu)
        aq, asf = self._mlp_dn_quant(act, layer=layer)
        return self._mlp_dn_gemm(act_fp4=aq, act_sf=asf, output_dtype=dt, layer=layer)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        advance_step_generation()

        cfg = self.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        num_layers = cfg.num_hidden_layers
        dense_layers = cfg.first_k_dense_replace

        md = attn_metadata
        self._check_step_contract(md)
        rope = self._rope

        step = _build_step_args(md)
        self._attn_ctx.bind_const(**step, **rope)
        self._attn_gen.bind_const(**step, **rope)
        self._mla_append.bind_const(
            cos_sin_cache=rope["rotary_cos_sin"],
            tokens_per_block=md.tokens_per_block,
            attention_window_size=md.max_seq_len,
        )
        self._mla_load.bind_const(
            tokens_per_block=md.tokens_per_block, attention_window_size=md.max_seq_len
        )
        self._mla_gen.bind_const(
            rotary_cos_sin=rope["rotary_cos_sin"],
            tokens_per_block=md.tokens_per_block,
            attention_window_size=md.max_seq_len,
        )

        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, self.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        gen = num_tokens - tc
        dt = h.dtype
        dev = h.device
        # Query rows per generation sequence (the whole draft chain under MTP): the only thing
        # that tells the attention ops the query block is taller, and what aligns the
        # bottom-right within-block causal mask.
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        ctx_kv_tokens = int(md.host_total_kv_lens[0]) if tc else 0
        dp_rows = self._dp_rows(md, num_tokens)
        pad_rows = dp_rows - num_tokens

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        x = self._rms0(h)
        residual = h
        for i in range(num_layers):
            q = self._qb(self._q_lora_norm(self._qa(x, layer=i), layer=i), layer=i)
            kva = self._kva(x, layer=i)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = self._kv_lora_norm(ckv_raw, layer=i)
            latent = torch.cat([ckv, k_pe], -1)
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
            latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

            if tc:
                if self._cached_ctx:
                    self._mla_append(
                        q=q_ctx,
                        latent_cache=latent_ctx,
                        num_contexts=num_ctx,
                        cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                        cu_seq_lens=md.ctx_kv_indptr,
                        max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                        kv_cache_block_offsets=block_offsets,
                        host_kv_cache_pool_pointers=pool_ptrs,
                        host_kv_cache_pool_mapping=pool_map,
                        layer=i,
                    )
                    ckv_full, k_pe_full = self._mla_load(
                        out_dtype=dt,
                        num_contexts=num_ctx,
                        num_ctx_kv_tokens=ctx_kv_tokens,
                        max_ctx_kv_len=int(md.max_ctx_kv_len),
                        cu_ctx_kv_lens=md.ctx_kv_indptr,
                        kv_cache_block_offsets=block_offsets,
                        host_kv_cache_pool_pointers=pool_ptrs,
                        host_kv_cache_pool_mapping=pool_map,
                        layer=i,
                    )
                    latent_arg = None
                else:
                    ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                    k_pe_full = None
                    latent_arg = latent_ctx
                tkv = ctx_kv_tokens
                # The context FMHA hard-codes V's row stride as the full packed [k_nope | v]
                # width and reads the column block, so V must stay this split view.
                kv = self._kvb(ckv_full, layer=i)
                k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
                k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
                k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
                k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
                if k_pe_full is not None:
                    k_pe_dst.copy_(
                        torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                    )
                self._attn_ctx(
                    q=q_ctx,
                    k=torch.reshape(k, [tkv, heads * qk_dim]),
                    v=v_view,
                    output=attn_ctx,
                    latent_cache=latent_arg,
                    layer=i,
                )

            if gen:
                q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
                q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
                fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
                fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
                # Over an fp8 pool `_mla_gen` reads this half to build the quantized query, so
                # this BMM must be issued first on the same stream; overlapping them is a silent
                # race (they only write disjoint halves on a bf16 pool).
                self._absorb(
                    a=torch.transpose(q_nope, 0, 1), out=torch.transpose(fq_nope, 0, 1), layer=i
                )
                cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
                cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
                counter = torch.empty([1], dtype=torch.uint32, device=dev)
                quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
                bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
                bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
                self._mla_gen(
                    fused_q=fused_q,
                    q_pe=q_pe,
                    latent_cache=latent_gen,
                    cu_q_seqlens=cu_q,
                    cu_kv_seqlens=cu_kv,
                    fmha_scheduler_counter=counter,
                    mla_bmm1_scale=bmm1_scale,
                    mla_bmm2_scale=bmm2_scale,
                    quant_q_buffer=quant_q,
                    sequence_length=md.kv_lens_cuda_runtime,
                    host_past_key_value_lengths=md.kv_lens_runtime,
                    host_context_lengths=md.prompt_lens_cpu_runtime,
                    num_contexts=num_ctx,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    predicted_tokens_per_seq=gen_p,
                    layer=i,
                )
                lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
                self._attn_gen(
                    q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                    output=lat_out,
                    latent_cache=latent_gen,
                    q_pe=q_pe,
                    cu_q_seqlens=cu_q,
                    cu_kv_seqlens=cu_kv,
                    fmha_scheduler_counter=counter,
                    mla_bmm1_scale=bmm1_scale,
                    mla_bmm2_scale=bmm2_scale,
                    quant_q_buffer=quant_q,
                    predicted_tokens_per_seq=gen_p,
                    layer=i,
                )
                self._expand(
                    a=torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1),
                    out=torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1),
                    layer=i,
                )

            o = self._o(attn_out, layer=i)
            self._norm2(o, residual, layer=i)
            if i < dense_layers:
                mlp_out = self._dense_mlp(o, i, dt)
            else:
                side = self._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._dense_mlp(o, i, dt)
                # Gathering before the router is what keeps the four expert windows tiling the
                # routing space exactly once: every rank routes the identical token set, so a
                # token's top-8 ids agree across ranks and each id falls in exactly one window.
                o_pad = (
                    o
                    if not pad_rows
                    else nn.functional.pad(o, [0, 0, 0, pad_rows], mode="constant", value=0.0)
                )
                o_all = self._allgather(o_pad)
                parts = []
                for chunk in torch.split(o_all, _moe_chunk_sizes(o_all.shape[0]), 0):
                    logits = self._router(chunk, layer=i)
                    topk_w, topk_ids = self._noaux(logits, layer=i)
                    xq, xsf = self._moe_quant(chunk, layer=i)
                    parts.append(
                        self._moe_runner(
                            hidden_states=xq,
                            hidden_states_scale=xsf.view(torch.float8_e4m3fn),
                            topk_weights=topk_w,
                            topk_ids=topk_ids,
                            layer=i,
                        )[0]
                    )
                routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
                routed_pad = self._reducescatter(routed_all)
                routed = (
                    routed_pad
                    if not pad_rows
                    else torch.split(routed_pad, [num_tokens, pad_rows], 0)[0]
                )
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            self._next_norm(mlp_out, residual, layer=i)
            x = mlp_out
        return x


class MTPLayer:
    """The checkpoint's multi-token-prediction module, at layer index `num_hidden_layers`,
    replayed once per draft step.

    It differs from a trunk layer in three ways the code alone does not show: an `eh_proj` front
    end that mixes the next token's embedding into `h`, bf16 experts (the checkpoint excludes
    `model.layers.61*` from NVFP4, so they go to `fused_moe`, not the trtllm-gen runner), and an
    output returned **un-normalized** — `shared_head` applies the module's own norm, and the
    runtime feeds the un-normalized tensor back in as the next step's `h`."""

    def __init__(self, core: DeepseekV3ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own output head: its own RMS norm — a distinct parameter from the
        trunk's `model.norm` — then the trunk's `lm_head`."""
        mw = self.core._mtp
        return self.logits_processor.forward(
            flashinfer_rmsnorm(
                hidden_states,
                mw["head_norm"],
                self.core.model_config.pretrained_config.rms_norm_eps,
            ),
            lm_head,
            attn_metadata,
            return_context_logits,
        )

    def _shared_mlp(self, x, mw):
        """The module's shared expert: one bf16 SwiGLU MLP over this rank's own tokens."""
        return cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, mw["sh_gu"])), mw["sh_dn"])

    def _routed_experts(self, x, mw, all_rank_num_tokens, rows, dt):
        """The module's expert-parallel round trip, the trunk's shape at a different dtype.

        `fused_moe` wants fp32 `token_final_scales`, which is why the router GEMM emits fp32
        here and the trunk's emits bf16, and it requires a token's `topk` ids to be distinct
        above 256 tokens — a repeat reads out of bounds in `finalizeMoeRoutingKernel`, faulting
        or returning silent garbage — which a top-k over expert indices gives structurally."""
        core = self.core
        cfg = core.model_config.pretrained_config
        dp_rows = _mtp_dp_rows(all_rank_num_tokens, core.rank, core.dp_size, rows)
        pad_rows = dp_rows - rows
        x_pad = (
            x
            if not pad_rows
            else nn.functional.pad(x, [0, 0, 0, pad_rows], mode="constant", value=0.0)
        )
        x_all = allgather(x_pad, None, core.dp_group)
        parts = []
        for chunk in torch.split(x_all, _moe_chunk_sizes(x_all.shape[0]), 0):
            logits = cublas_mm(chunk, mw["router"], None, torch.float32)
            topk_w, topk_ids = noaux_tc_op(
                logits,
                mw["router_bias"],
                cfg.n_group,
                cfg.topk_group,
                cfg.num_experts_per_tok,
                float(cfg.routed_scaling_factor),
            )
            parts.append(
                fused_moe(
                    chunk,
                    topk_ids,
                    topk_w,
                    mw["fc1"],
                    None,
                    mw["fc2"],
                    None,
                    dt,
                    [],
                    ep_size=core.ep_size,
                    ep_rank=core.ep_rank,
                    activation_type=_FUSED_MOE_ACT_SWIGLU,
                )[0]
            )
        routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        routed_pad = reducescatter(routed_all, None, core.dp_group)
        return routed_pad if not pad_rows else torch.split(routed_pad, [rows, pad_rows], 0)[0]

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        cfg = core.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        rope_dim = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        q_lora = cfg.q_lora_rank
        qk_dim = nope + rope_dim
        lat_dim = kv_lora + rope_dim
        eps = cfg.rms_norm_eps
        num_layers = cfg.num_hidden_layers

        mw, rope = core._mtp, core._rope
        md = attn_metadata
        step = _build_step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        gen = rows - tc
        # Read the phase from the metadata on every call: the worker rewrites `attn_metadata` in
        # place between draft step 0 and step 1+, so a value computed on the first of the N
        # calls in one forward is wrong on the rest, and frozen wrong under capture.
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        ctx_kv_tokens = int(md.host_total_kv_lens[0]) if tc else 0
        # Under a one-model MTP mode the engine raises the KV pool's layer count by
        # `num_nextn_predict_layers`, so this module addresses layer `num_hidden_layers` in the
        # same pool as the trunk's 61 layers. Nothing in the config stub or manifest says so.
        layer_idx = num_layers

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, mw["enorm"], eps)
        hn = flashinfer_rmsnorm(hidden_states, mw["hnorm"], eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), mw["eh"])

        residual = x
        xn = flashinfer_rmsnorm(x, mw["norm1"], eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        q = cublas_mm(
            flashinfer_rmsnorm(cublas_mm(xn, mw["qa"]), mw["q_norm"], eps),
            mw["qb"],
        )
        kva = cublas_mm(xn, mw["kva"])
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, mw["kv_norm"], eps)
        latent = torch.cat([ckv, k_pe], -1)
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

        if tc:
            if core._cached_ctx:
                mla_rope_append_paged_kv_assign_q(
                    q_ctx,
                    latent_ctx,
                    num_ctx,
                    md.ctx_cached_token_indptr,
                    md.ctx_kv_indptr,
                    int(md.max_ctx_seq_len),
                    rope["rotary_cos_sin"],
                    heads,
                    nope,
                    rope_dim,
                    kv_lora,
                    block_offsets,
                    pool_ptrs,
                    pool_map,
                    None,
                    _KV_RESIDUAL_DIM,
                    layer_idx,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    core.quant_mode,
                )
                ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                    dt,
                    num_ctx,
                    ctx_kv_tokens,
                    int(md.max_ctx_kv_len),
                    md.ctx_kv_indptr,
                    block_offsets,
                    pool_ptrs,
                    pool_map,
                    None,
                    layer_idx,
                    kv_lora,
                    rope_dim,
                    tokens_per_block,
                    md.max_seq_len,
                    1,
                    core.quant_mode,
                )
                latent_arg = None
            else:
                ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                k_pe_full = None
                latent_arg = latent_ctx
            tkv = ctx_kv_tokens
            kv = cublas_mm(ckv_full, mw["kvb"])
            k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
            k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
            k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
            k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
            if k_pe_full is not None:
                k_pe_dst.copy_(
                    torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                )
            thop_attention(
                q=q_ctx,
                k=torch.reshape(k, [tkv, heads * qk_dim]),
                v=v_view,
                output=attn_ctx,
                latent_cache=latent_arg,
                q_pe=None,
                local_layer_idx=layer_idx,
                is_fused_qkv=False,
                attention_input_type=1,
                num_heads=heads,
                num_kv_heads=heads,
                head_size=qk_dim,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=v_dim,
                cu_q_seqlens=None,
                cu_kv_seqlens=None,
                fmha_scheduler_counter=None,
                predicted_tokens_per_seq=1,
                **step,
                **rope,
                **core._call,
            )

        if gen:
            q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
            fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
            bmm_out(torch.transpose(q_nope, 0, 1), mw["k_b"], torch.transpose(fq_nope, 0, 1))
            cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            counter = torch.empty([1], dtype=torch.uint32, device=dev)
            quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
            bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
            bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
            mla_rope_generation(
                fused_q,
                q_pe,
                latent_gen,
                rope["rotary_cos_sin"],
                cu_q,
                cu_kv,
                counter,
                bmm1_scale,
                bmm2_scale,
                quant_q,
                md.kv_lens_cuda_runtime,
                md.kv_lens_runtime,
                md.prompt_lens_cpu_runtime,
                num_ctx,
                block_offsets,
                pool_ptrs,
                pool_map,
                None,
                None,
                None,
                None,
                None,
                [None, None],
                gen_p,
                layer_idx,
                heads,
                1,
                lat_dim,
                _KV_RESIDUAL_DIM,
                tokens_per_block,
                md.max_seq_len,
                1,
                core.quant_mode,
                core.q_scaling,
                q_lora,
                kv_lora,
                nope,
                rope_dim,
                v_dim,
                True,
            )
            lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
            thop_attention(
                q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                k=None,
                v=None,
                output=lat_out,
                latent_cache=latent_gen,
                q_pe=q_pe,
                local_layer_idx=layer_idx,
                is_fused_qkv=True,
                attention_input_type=2,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                predicted_tokens_per_seq=gen_p,
                **step,
                **rope,
                **core._call,
            )
            bmm_out(
                torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1),
                mw["v_b_t"],
                torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1),
            )

        o = cublas_mm(attn_out, mw["o"])
        flashinfer_fused_add_rmsnorm(o, residual, mw["norm2"], eps)
        shared = self._shared_mlp(o, mw)
        routed = self._routed_experts(o, mw, all_rank_num_tokens, rows, dt)
        return torch.add(residual, torch.add(routed, shared))


class DraftModel:
    """The container the spec worker reaches this target's drafter through.

    `embed_tokens` must resolve through the core on every read: this container is built while
    every parameter is still a meta tensor, and the engine materializes the registry by
    replacing the tensor objects, so a reference snapshotted here stays on meta."""

    def __init__(self, core: DeepseekV3ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


@register_auto_model("ModelingV2DeepseekR10528Nvfp4Sm103Dep4")
class ModelingV2DeepseekR10528Nvfp4Sm103Dep4(
    DecoderModelForCausalLM[DeepseekV3ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            DeepseekV3ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            self.spec_worker = get_spec_worker(self.spec_config, model_config, model_config.mapping)

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

        `resource_manager` must stay inside `**kwargs` — naming it here drops it from what the
        base forward receives — and `position_ids` must be handed on in the engine's `[1, T]`
        shape, since the worker squeezes it itself and flattening it here is silently wrong."""
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
        _weights.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.derive_after_load()
