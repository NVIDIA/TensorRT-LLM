# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: gpt-oss-120b / sm_103 / tp1 — self-contained modeling code.

Flat single-entry forward assembled from catalog entries only, running the
MXFP4 sparse MoE W4A8 on a single GPU (tp1). Weights are target-owned and
declared in the sibling weights.py. Each target is self-contained: it binds
every op it calls against the real weights in its own `__init__` and shares
no code with the other target.
"""

import math

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._core import ModelingV2Core
from tensorrt_llm._torch._experimental.modeling_v2._target import Phase, Target, phase_of
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import advance_step_generation
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    FusedQkNormRope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    ThopAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    Mxe4m3Mxe2m1BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    FlashinferFusedAddRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    FlashinferRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    Mxfp8Quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, register_auto_model

from . import weights as _weights

_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    # In-kernel RoPE is disabled here; the rope_* values below are inert placeholders, not this model's rope config.
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

_FC1_K_ALIGN = _weights.FC1_K_ALIGN


class GptOssModelingV2(ModelingV2Core):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config

        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        self.theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 0.0)))

        layer_types = list(cfg.layer_types)
        self.sliding = [t == "sliding_attention" for t in layer_types]
        self.window = cfg.sliding_window

        d = _weights.MODEL_WEIGHTS.dims(self)
        self.inter_pad = d.inter_pad
        self.fc1_k_pad = d.fc1_k_pad
        self.fc2_rows_pad = d.fc2_rows_pad
        self.w = _weights.MODEL_WEIGHTS.declare(d)

        self._prefill: Target | None = None
        self._decode: Target | None = None

    def build_layer_views(self) -> None:
        """Construct the two targets, now that the weights are real.

        Must only be called after meta init is over, never from `__init__`, where the
        shell's containers are still meta.
        """
        cfg = self.model_config.pretrained_config
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        rotary_dim = cfg.head_dim
        yarn_factor = float(rope["factor"])
        beta_fast = float(rope["beta_fast"])
        beta_slow = float(rope["beta_slow"])
        orig_max = float(rope["original_max_position_embeddings"])
        truncate = bool(rope.get("truncate", True))

        def correction_dim(rotations: float) -> float:
            return (
                rotary_dim
                * math.log(orig_max / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(self.theta))
            )

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        yarn_low = max(low, 0.0)
        yarn_high = min(high, rotary_dim - 1.0)
        yarn_attn_factor = 0.1 * math.log(yarn_factor) + 1.0 if yarn_factor > 1.0 else 1.0

        self._prefill = PrefillTarget(
            self,
            yarn_factor=yarn_factor,
            yarn_low=yarn_low,
            yarn_high=yarn_high,
            yarn_attn_factor=yarn_attn_factor,
        )
        self._decode = DecodeTarget(
            self,
            yarn_factor=yarn_factor,
            yarn_low=yarn_low,
            yarn_high=yarn_high,
            yarn_attn_factor=yarn_attn_factor,
        )

    def _select_target(self, attn_metadata) -> Target:
        """Which target runs this step."""
        return self._decode if phase_of(attn_metadata) is Phase.DECODE else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Route the step, and nothing else."""
        advance_step_generation()
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget(Target):
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(
        self,
        core: GptOssModelingV2,
        *,
        yarn_factor: float,
        yarn_low: float,
        yarn_high: float,
        yarn_attn_factor: float,
    ) -> None:
        """Bind every op this target calls, once, against the real weights."""
        super().__init__(core)
        cfg = core.model_config.pretrained_config
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype
        n = cfg.num_hidden_layers

        self._norm0 = FlashinferRmsnorm()
        self._norm0.bind_const(weight=w["l0_norm1"], eps=cfg.rms_norm_eps)

        self._qkv = CublasMm()
        for i in range(n):
            self._qkv.bind_layered(i, mat_b=w[f"l{i}_qkv"].t(), bias=w[f"l{i}_qkv_bias"])

        # fused_qk_norm_rope requires valid q/k norm weight tensors even when is_qk_norm=False, where
        # the values are unused.
        no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._qk_rope = FusedQkNormRope()
        self._qk_rope.bind_const(
            num_heads_q=cfg.num_attention_heads,
            num_heads_k=cfg.num_key_value_heads,
            num_heads_v=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            q_weight=no_qk_norm,
            k_weight=no_qk_norm,
            base=core.theta,
            is_neox=True,
            factor=yarn_factor,
            low=yarn_low,
            high=yarn_high,
            attention_factor=yarn_attn_factor,
            is_qk_norm=False,
        )

        self._attn = ThopAttention()
        for i in range(n):
            self._attn.bind_layered(i, attention_sinks=w[f"l{i}_sinks"], local_layer_idx=i)
        self._attn.bind_const(
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            **_CALL_CONSTANTS,
        )

        self._o_proj = CublasMm()
        for i in range(n):
            self._o_proj.bind_layered(i, mat_b=w[f"l{i}_o"].t(), bias=w[f"l{i}_o_bias"])

        self._norm2 = FlashinferFusedAddRmsnorm()
        for i in range(n):
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
        self._norm2.bind_const(eps=cfg.rms_norm_eps)

        self._router = CublasMm()
        for i in range(n):
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t(), bias=w[f"l{i}_router_bias"])

        self._quant = Mxfp8Quantize()
        # The swizzled 128x4 order has the same byte count whenever num_tokens % 128 == 0, which every
        # decode CUDA-graph batch is, so a wrong value here is accepted silently as a wrong answer.
        self._quant.bind_const(swizzled_layout=False, alignment=_FC1_K_ALIGN)

        self._moe = Mxe4m3Mxe2m1BlockScaleMoeRunner()
        for i in range(n):
            self._moe.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"],
                gemm1_bias=w[f"l{i}_fc1_b"],
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"],
                gemm2_bias=w[f"l{i}_fc2_b"],
            )
        self._moe.bind_glu(cfg.num_local_experts, cfg.swiglu_limit, device)
        self._moe.bind_const(
            # The MoE op silently ignores routing_bias on this routing method; the bias rides the
            # router GEMM epilogue instead.
            routing_bias=None,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=core.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=cfg.intermediate_size,
            local_expert_offset=0,
            local_num_experts=cfg.num_local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            act_type=0,
        )

        self._norm_next = FlashinferFusedAddRmsnorm()
        for i in range(n):
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < n else w["final_norm"]
            self._norm_next.bind_layered(i, weight=next_w)
        self._norm_next.bind_const(eps=cfg.rms_norm_eps)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state, once per forward."""
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
            spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
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
        cfg = core.model_config.pretrained_config

        self._check_step_contract(attn_metadata)
        self._attn.bind_const(**self.step_args(attn_metadata))

        full_window = attn_metadata.max_seq_len
        for i, sliding in enumerate(core.sliding):
            self._attn.bind_layered(
                i, attention_window_size=core.window if sliding else full_window
            )

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        attn_out = torch.empty(
            [num_tokens, cfg.num_attention_heads * cfg.head_dim], dtype=dt, device=h.device
        )

        x = self._norm0(h)
        residual = h
        for i in range(cfg.num_hidden_layers):
            qkv = self._qkv(x, layer=i)
            qkv = self._qk_rope(qkv, position_ids=pos)
            attn_out = self._attn(q=qkv, output=attn_out, layer=i)
            o = self._o_proj(attn_out, layer=i)
            o, residual = self._norm2(o, residual, layer=i)
            router_logits = self._router(o, layer=i)
            hidden_fp8, hidden_sf = self._quant(o)
            moe = self._moe(
                routing_logits=router_logits,
                hidden_states=hidden_fp8,
                hidden_states_scale=hidden_sf,
                layer=i,
            )
            moe, residual = self._norm_next(moe, residual, layer=i)
            x = moe
        return x


class DecodeTarget(Target):
    """The specialization: routed to only when there are no context rows."""

    def __init__(
        self,
        core: GptOssModelingV2,
        *,
        yarn_factor: float,
        yarn_low: float,
        yarn_high: float,
        yarn_attn_factor: float,
    ) -> None:
        """Bind every op this target calls, once, against the real weights."""
        super().__init__(core)
        cfg = core.model_config.pretrained_config
        w = core.w
        device = w["final_norm"].device
        dtype = w["final_norm"].dtype
        n = cfg.num_hidden_layers

        self._norm0 = FlashinferRmsnorm()
        self._norm0.bind_const(weight=w["l0_norm1"], eps=cfg.rms_norm_eps)

        self._qkv = CublasMm()
        for i in range(n):
            self._qkv.bind_layered(i, mat_b=w[f"l{i}_qkv"].t(), bias=w[f"l{i}_qkv_bias"])

        # fused_qk_norm_rope requires valid q/k norm weight tensors even when is_qk_norm=False, where
        # the values are unused.
        no_qk_norm = torch.zeros(cfg.head_dim, dtype=dtype, device=device)
        self._qk_rope = FusedQkNormRope()
        self._qk_rope.bind_const(
            num_heads_q=cfg.num_attention_heads,
            num_heads_k=cfg.num_key_value_heads,
            num_heads_v=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            q_weight=no_qk_norm,
            k_weight=no_qk_norm,
            base=core.theta,
            is_neox=True,
            factor=yarn_factor,
            low=yarn_low,
            high=yarn_high,
            attention_factor=yarn_attn_factor,
            is_qk_norm=False,
        )

        self._attn = ThopAttention()
        for i in range(n):
            self._attn.bind_layered(i, attention_sinks=w[f"l{i}_sinks"], local_layer_idx=i)
        self._attn.bind_const(
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
            **_CALL_CONSTANTS,
        )

        self._o_proj = CublasMm()
        for i in range(n):
            self._o_proj.bind_layered(i, mat_b=w[f"l{i}_o"].t(), bias=w[f"l{i}_o_bias"])

        self._norm2 = FlashinferFusedAddRmsnorm()
        for i in range(n):
            self._norm2.bind_layered(i, weight=w[f"l{i}_norm2"])
        self._norm2.bind_const(eps=cfg.rms_norm_eps)

        self._router = CublasMm()
        for i in range(n):
            self._router.bind_layered(i, mat_b=w[f"l{i}_router"].t(), bias=w[f"l{i}_router_bias"])

        self._quant = Mxfp8Quantize()
        # The swizzled 128x4 order has the same byte count whenever num_tokens % 128 == 0, which every
        # decode CUDA-graph batch is, so a wrong value here is accepted silently as a wrong answer.
        self._quant.bind_const(swizzled_layout=False, alignment=_FC1_K_ALIGN)

        self._moe = Mxe4m3Mxe2m1BlockScaleMoeRunner()
        for i in range(n):
            self._moe.bind_layered(
                i,
                gemm1_weights=w[f"l{i}_fc1_w"],
                gemm1_weights_scale=w[f"l{i}_fc1_s"],
                gemm1_bias=w[f"l{i}_fc1_b"],
                gemm2_weights=w[f"l{i}_fc2_w"],
                gemm2_weights_scale=w[f"l{i}_fc2_s"],
                gemm2_bias=w[f"l{i}_fc2_b"],
            )
        self._moe.bind_glu(cfg.num_local_experts, cfg.swiglu_limit, device)
        self._moe.bind_const(
            # The MoE op silently ignores routing_bias on this routing method; the bias rides the
            # router GEMM epilogue instead.
            routing_bias=None,
            num_experts=cfg.num_local_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=core.inter_pad,
            valid_hidden_size=cfg.hidden_size,
            valid_intermediate_size=cfg.intermediate_size,
            local_expert_offset=0,
            local_num_experts=cfg.num_local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            act_type=0,
        )

        self._norm_next = FlashinferFusedAddRmsnorm()
        for i in range(n):
            next_w = w[f"l{i + 1}_norm1"] if i + 1 < n else w["final_norm"]
            self._norm_next.bind_layered(i, weight=next_w)
        self._norm_next.bind_const(eps=cfg.rms_norm_eps)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state, once per forward."""
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
            spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
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
        cfg = core.model_config.pretrained_config

        self._check_step_contract(attn_metadata)
        self._attn.bind_const(**self.step_args(attn_metadata))

        full_window = attn_metadata.max_seq_len
        for i, sliding in enumerate(core.sliding):
            self._attn.bind_layered(
                i, attention_window_size=core.window if sliding else full_window
            )

        pos = torch.reshape(position_ids, [-1])

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        dt = h.dtype
        attn_out = torch.empty(
            [num_tokens, cfg.num_attention_heads * cfg.head_dim], dtype=dt, device=h.device
        )

        x = self._norm0(h)
        residual = h
        for i in range(cfg.num_hidden_layers):
            qkv = self._qkv(x, layer=i)
            qkv = self._qk_rope(qkv, position_ids=pos)
            attn_out = self._attn(q=qkv, output=attn_out, layer=i)
            o = self._o_proj(attn_out, layer=i)
            o, residual = self._norm2(o, residual, layer=i)
            router_logits = self._router(o, layer=i)
            hidden_fp8, hidden_sf = self._quant(o)
            moe = self._moe(
                routing_logits=router_logits,
                hidden_states=hidden_fp8,
                hidden_states_scale=hidden_sf,
                layer=i,
            )
            moe, residual = self._norm_next(moe, residual, layer=i)
            x = moe
        return x


@register_auto_model("ModelingV2GptOss120bSm103Tp1")
class ModelingV2GptOss120bSm103Tp1(DecoderModelForCausalLM[GptOssModelingV2, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        # This checkpoint's config.json declares no dtype, so lm_head would be sized fp32 while every
        # other tensor is bf16, failing two layers away.
        if cfg.torch_dtype is None:
            cfg.torch_dtype = model_config.torch_dtype
        super().__init__(
            GptOssModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.build_layer_views()
