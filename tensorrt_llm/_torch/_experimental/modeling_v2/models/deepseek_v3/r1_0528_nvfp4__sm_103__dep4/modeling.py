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
from typing import NamedTuple, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._target import Phase, Target, phase_of
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import get_spec_worker

from . import weights as _weights
from .ops.attention.load_paged_kv_cache_for_mla import load_paged_kv_cache_for_mla  # noqa: E501
from .ops.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q,
)
from .ops.attention.mla_rope_generation import mla_rope_generation
from .ops.gemm.bmm_out import bmm_out
from .ops.moe.fp4_block_scale_moe_runner import fp4_block_scale_moe_runner  # noqa: E501
from .ops.moe.fused_moe import fused_moe
from .ops.moe.noaux_tc_op import noaux_tc_op
from .ops.quantization.fp4_quantize import fp4_quantize

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


class DeepseekV3Dep4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        # The first-step derivation runs once and always; the contract probe that
        # shared this flag now lives on each target, behind its own.
        self._derive_pending = True

        mapping = model_config.mapping
        self.rank = mapping.rank
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.dp_group = list(range(mapping.world_size))
        self.dp_size = mapping.world_size

        self.quant_mode = _QUANT_MODE_FP8_KV

        self.mtp_layers = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None

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

        num_experts = cfg.n_routed_experts
        self.local_experts = num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

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

    def post_load_weights(self) -> None:
        """Build the two targets, now that the weights are real.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has none.

        The rope tables, the MTP operands and the side stream stay on the core rather
        than a target: `MTPLayer` reads the first two, and a CUDA stream created inside
        a graph capture is not capturable, so it is made once, here.
        """
        cfg = self.model_config.pretrained_config
        device = self.w["final_norm"].device
        self._rope_positions = cfg.max_position_embeddings
        self._rope = self._rope_tables(device, self._rope_positions)
        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
        self._side_stream = torch.cuda.Stream(device=device)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata) -> Target:
        """Which target runs this step."""
        return self._decode if phase_of(attn_metadata) is Phase.DECODE else self._prefill

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

    def _derive_mtp(self) -> dict:
        """The MTP layer's operand set, derived exactly as a trunk layer's is."""
        nl = self.model_config.pretrained_config.num_hidden_layers
        cfg = self.model_config.pretrained_config
        heads = cfg.num_attention_heads
        nope = cfg.qk_nope_head_dim
        v_dim = cfg.v_head_dim
        kv_lora = cfg.kv_lora_rank
        w = self.w
        hn = heads * nope
        kvb = w[f"l{nl}_kvb"]
        return {
            "enorm": w[f"l{nl}_enorm"],
            "hnorm": w[f"l{nl}_hnorm"],
            "eh": w[f"l{nl}_eh"].t(),
            "norm1": w[f"l{nl}_norm1"],
            "qa": w[f"l{nl}_qa"].t(),
            "q_norm": w[f"l{nl}_q_norm"],
            "qb": w[f"l{nl}_qb"].t(),
            "kva": w[f"l{nl}_kva"].t(),
            "kv_norm": w[f"l{nl}_kv_norm"],
            "k_b": kvb[:hn].reshape(heads, nope, kv_lora),
            "v_b_t": torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2),
            "kvb": kvb.t(),
            "o": w[f"l{nl}_o"].t(),
            "norm2": w[f"l{nl}_norm2"],
            "router": w[f"l{nl}_router"].t(),
            "router_bias": w[f"l{nl}_router_bias"],
            "fc1": w[f"l{nl}_fc1"],
            "fc2": w[f"l{nl}_fc2"],
            "sh_gu": w[f"l{nl}_sh_gu"].t(),
            "sh_dn": w[f"l{nl}_sh_dn"].t(),
            "head_norm": w[f"l{nl}_head_norm"],
        }

    def _derive_first_step(self, md) -> None:
        """Grow the rope table to cover every position the engine admits, and fix the context
        flavor this engine construction runs.

        Unconditional, unlike the targets' contract probe: both of these decide what the
        model computes, not whether it is well-formed. They used to hang off that probe
        and so ran only under TRTLLM_MODELING_V2_VALIDATE, which the accuracy tests set
        and a served engine does not -- leaving production with a rope table sized to
        `max_position_embeddings` and `_cached_ctx` stuck at False.
        """
        if not self._derive_pending:
            return
        # A table shorter than the engine's max_seq_len is read out of bounds with no check, and
        # `rope_max_positions` is one of the arguments the MLA path ignores. `speculative_config`
        # inflates max_seq_len past `max_position_embeddings`, so take the engine's own number.
        if md.max_seq_len > self._rope_positions:
            self._rope_positions = md.max_seq_len
            self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
        self._cached_ctx = all(hasattr(md, n) for n in _CACHED_CTX_FIELDS) and bool(
            md.enable_context_mla_with_cached_kv
        )
        self._derive_pending = False

    def _dp_rows(self, md, num_tokens: int) -> int:
        """The uniform row count every rank pads its token block to before the MoE round trip."""
        counts = [int(n) for n in md.all_rank_num_tokens]
        return max(counts)


class _Layer(NamedTuple):
    """One trunk layer's weights, as the views the ops read.

    GEMM operands are `.t()` views; `k_b` / `v_b_t` are the two halves of `kv_b` in the
    shapes the absorbed-q and expand BMMs take; `norm_next` is the *next* layer's input norm
    (the final norm on the last layer). The MoE fields are None on the dense layers.
    """

    qa: torch.Tensor
    q_norm: torch.Tensor
    qb: torch.Tensor
    kva: torch.Tensor
    kv_norm: torch.Tensor
    kvb: torch.Tensor
    k_b: torch.Tensor
    v_b_t: torch.Tensor
    o: torch.Tensor
    norm2: torch.Tensor
    norm_next: torch.Tensor
    mlp_gu_gscale: torch.Tensor
    mlp_gu_w: torch.Tensor
    mlp_gu_s: torch.Tensor
    mlp_gu_alpha: torch.Tensor
    mlp_dn_gscale: torch.Tensor
    mlp_dn_w: torch.Tensor
    mlp_dn_s: torch.Tensor
    mlp_dn_alpha: torch.Tensor
    router: torch.Tensor | None
    router_bias: torch.Tensor | None
    moe_gscale: torch.Tensor | None
    fc1_w: torch.Tensor | None
    fc1_s: torch.Tensor | None
    fc2_w: torch.Tensor | None
    fc2_s: torch.Tensor | None
    out1_scale: torch.Tensor | None
    out1_gate_scale: torch.Tensor | None
    out2_scale: torch.Tensor | None


def _layer_views(core: "DeepseekV3Dep4ModelingV2") -> list[_Layer]:
    """Every trunk layer's `_Layer`, read off the realized weights."""
    cfg = core.model_config.pretrained_config
    w = core.w
    num_layers = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    nope = cfg.qk_nope_head_dim
    v_dim = cfg.v_head_dim
    kv_lora = cfg.kv_lora_rank
    dense_layers = cfg.first_k_dense_replace
    hn = heads * nope
    window = slice(core.expert_offset, core.expert_offset + core.local_experts)

    layers = []
    for i in range(num_layers):
        kvb = w[f"l{i}_kvb"]
        # The checkpoint stores reciprocals (`input_scale` = 1/g_act, `weight_scale_2` =
        # 1/g_w), so the quantizer's global scale is 1/input_scale and the GEMM alpha is
        # their product — no further reciprocal anywhere.
        isc1 = w[f"l{i}_mlp_isc1"]
        isc2 = w[f"l{i}_mlp_isc2"]
        moe: dict = dict(
            router=None,
            router_bias=None,
            moe_gscale=None,
            fc1_w=None,
            fc1_s=None,
            fc2_w=None,
            fc2_s=None,
            out1_scale=None,
            out1_gate_scale=None,
            out2_scale=None,
        )
        if i >= dense_layers:
            # One quantization of the gathered hidden states feeds every expert on every rank,
            # so the routed FC1 activation scale has to be a single value and the same value on
            # all four ranks, or the windows no longer sum to the whole layer; the shared
            # expert's `input_scale` is the max over all routed ones, so it is the safe one.
            e_isc2 = w[f"l{i}_e_isc2"][window]
            gate1 = (isc1[0] * w[f"l{i}_e_ws2_1"][window]).contiguous()
            moe = dict(
                router=w[f"l{i}_router"].t(),
                router_bias=w[f"l{i}_router_bias"],
                moe_gscale=(1.0 / isc1).contiguous(),
                fc1_w=w[f"l{i}_fc1_w"],
                fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                fc2_w=w[f"l{i}_fc2_w"],
                fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
                out1_scale=(gate1 / e_isc2).contiguous(),
                out1_gate_scale=gate1,
                out2_scale=(e_isc2 * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            )
        layers.append(
            _Layer(
                qa=w[f"l{i}_qa"].t(),
                q_norm=w[f"l{i}_q_norm"],
                qb=w[f"l{i}_qb"].t(),
                kva=w[f"l{i}_kva"].t(),
                kv_norm=w[f"l{i}_kv_norm"],
                kvb=kvb.t(),
                k_b=kvb[:hn].reshape(heads, nope, kv_lora),
                v_b_t=torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2),
                o=w[f"l{i}_o"].t(),
                norm2=w[f"l{i}_norm2"],
                norm_next=w[f"l{i + 1}_norm1"] if i + 1 < num_layers else w["final_norm"],
                mlp_gu_gscale=(1.0 / isc1).contiguous(),
                mlp_gu_w=w[f"l{i}_mlp_gu_w"],
                mlp_gu_s=w[f"l{i}_mlp_gu_s"],
                mlp_gu_alpha=(isc1 * w[f"l{i}_mlp_ws2_1"]).contiguous(),
                mlp_dn_gscale=(1.0 / isc2).contiguous(),
                mlp_dn_w=w[f"l{i}_mlp_dn_w"],
                mlp_dn_s=w[f"l{i}_mlp_dn_s"],
                mlp_dn_alpha=(isc2 * w[f"l{i}_mlp_ws2_2"]).contiguous(),
                **moe,
            )
        )
    return layers


class PrefillTarget(Target):
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: "DeepseekV3Dep4ModelingV2") -> None:
        """Take the per-layer weight views this target's forward reads, once, off the real
        weights. The MTP operands are read off `core._mtp` at the call instead."""
        super().__init__(core)
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The batch state both MLA calls of one step share."""
        return _build_step_args(md)

    def _dense_mlp(self, x, L: _Layer, dt):
        """One NVFP4 SwiGLU MLP over this rank's own tokens, used for both the dense layers' MLP
        and every MoE layer's shared expert."""
        xq, xsf = fp4_quantize(x, L.mlp_gu_gscale, _SF_VEC, False, _SF_SWIZZLED)
        gu = nvfp4_gemm(xq, L.mlp_gu_w, xsf, L.mlp_gu_s, L.mlp_gu_alpha, dt)
        act = flashinfer_silu_and_mul(gu)
        aq, asf = fp4_quantize(act, L.mlp_dn_gscale, _SF_VEC, False, _SF_SWIZZLED)
        return nvfp4_gemm(aq, L.mlp_dn_w, asf, L.mlp_dn_s, L.mlp_dn_alpha, dt)

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
        num_experts = cfg.n_routed_experts
        topk = cfg.num_experts_per_tok
        moe_inter = cfg.moe_intermediate_size
        n_group = cfg.n_group
        topk_group = cfg.topk_group
        routed_scale = float(cfg.routed_scaling_factor)
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        md = attn_metadata
        rope = core._rope
        call = core._call
        step = self.step_args(md)
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        # The paged-cache address book, shared by the two MLA preprocessing ops and both
        # attention calls. Under attention DP each rank owns a pool holding only its own
        # requests' latent rows.
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
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
        ctx_kv_tokens = int(md.host_total_kv_lens[0])
        dp_rows = core._dp_rows(md, num_tokens)
        pad_rows = dp_rows - num_tokens
        cached_ctx = core._cached_ctx

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        # One set of attention scratch for all layers: every tensor here is fully
        # rewritten by layer i's producer before layer i's consumer reads it, and
        # the whole attention path runs on the main stream, so layer i+1 cannot
        # overwrite what layer i still reads. 9 allocations per step, not 9 x 61.
        # `k`'s rope half is written only in the cached-ctx flavor; the fresh
        # flavor never reads it, in either allocation scheme.
        tkv = ctx_kv_tokens
        k = torch.empty([tkv, heads, qk_dim], dtype=dt, device=dev)
        k_nope_dst, k_pe_dst = torch.split(k, [nope, rope_dim], -1)
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            q = cublas_mm(flashinfer_rmsnorm(cublas_mm(x, L.qa), L.q_norm, eps), L.qb)
            kva = cublas_mm(x, L.kva)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = flashinfer_rmsnorm(ckv_raw, L.kv_norm, eps)
            latent = torch.cat([ckv, k_pe], -1)
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
            latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

            if cached_ctx:
                # The ctx_* metadata fields exist only in this flavor.
                q_ctx = mla_rope_append_paged_kv_assign_q(
                    q=q_ctx,
                    latent_cache=latent_ctx,
                    num_contexts=num_ctx,
                    cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                    cu_seq_lens=md.ctx_kv_indptr,
                    max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                    cos_sin_cache=rope["rotary_cos_sin"],
                    head_num=heads,
                    nope_size=nope,
                    rope_size=rope_dim,
                    lora_size=kv_lora,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    kv_scale_orig_quant=None,
                    residual_dim=_KV_RESIDUAL_DIM,
                    layer_idx=i,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                )
                ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                    out_dtype=dt,
                    num_contexts=num_ctx,
                    num_ctx_kv_tokens=ctx_kv_tokens,
                    max_ctx_kv_len=int(md.max_ctx_kv_len),
                    cu_ctx_kv_lens=md.ctx_kv_indptr,
                    kv_cache_block_offsets=block_offsets,
                    host_kv_cache_pool_pointers=pool_ptrs,
                    host_kv_cache_pool_mapping=pool_map,
                    kv_scale_quant_orig=None,
                    layer_idx=i,
                    kv_lora_rank=kv_lora,
                    qk_rope_head_dim=rope_dim,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                )
                latent_arg = None
            else:
                ckv_full, _ = torch.split(ckv, [tc, gen], 0)
                k_pe_full = None
                latent_arg = latent_ctx
            # The context FMHA hard-codes V's row stride as the full packed [k_nope | v]
            # width and reads the column block, so V must stay this split view.
            kv = cublas_mm(ckv_full, L.kvb)
            k_nope, v_view = torch.split(kv, [heads * nope, heads * v_dim], -1)
            k_nope_dst.copy_(torch.reshape(k_nope, [tkv, heads, nope]))
            if k_pe_full is not None:
                k_pe_dst.copy_(
                    torch.reshape(k_pe_full, [tkv, 1, rope_dim]).expand([tkv, heads, rope_dim])
                )
            attn_ctx = thop_attention(
                q=q_ctx,
                k=torch.reshape(k, [tkv, heads * qk_dim]),
                v=v_view,
                output=attn_ctx,
                latent_cache=latent_arg,
                q_pe=None,
                local_layer_idx=i,
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
                # The context call's row count is `num_ctx_tokens` whatever the generation
                # phase carries, and the entry certifies the MLA context flavors at 1 only.
                predicted_tokens_per_seq=1,
                **step,
                **rope,
                **call,
            )

            if gen:
                q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
                q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
                # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
                # query, so this BMM must be issued first on the same stream; overlapping them
                # is a silent race (they only write disjoint halves on a bf16 pool).
                fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=L.k_b, out=fq_nope_t)
                fused_q = mla_rope_generation(
                    fused_q=fused_q,
                    q_pe=q_pe,
                    latent_cache=latent_gen,
                    rotary_cos_sin=rope["rotary_cos_sin"],
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
                    kv_scale_orig_quant=None,
                    kv_scale_quant_orig=None,
                    kv_cache_scale_orig_quant=None,
                    out_scale=None,
                    block_ids_per_seq=None,
                    helix_tensor_params=[None, None],
                    predicted_tokens_per_seq=gen_p,
                    layer_idx=i,
                    num_heads=heads,
                    num_kv_heads=1,
                    head_size=lat_dim,
                    residual_dim=_KV_RESIDUAL_DIM,
                    tokens_per_block=tokens_per_block,
                    attention_window_size=window,
                    beam_width=1,
                    quant_mode=quant_mode,
                    q_scaling=q_scaling,
                    q_lora_rank=q_lora,
                    kv_lora_rank=kv_lora,
                    qk_nope_head_dim=nope,
                    qk_rope_head_dim=rope_dim,
                    v_head_dim=kv_lora,
                    rope_append=True,
                )
                lat_out = thop_attention(
                    q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                    k=None,
                    v=None,
                    output=lat_out,
                    latent_cache=latent_gen,
                    q_pe=q_pe,
                    local_layer_idx=i,
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
                    **call,
                )
                attn_gen_t = bmm_out(a=lat_out_t, b=L.v_b_t, out=attn_gen_t)

            o = cublas_mm(attn_out, L.o)
            o, residual = flashinfer_fused_add_rmsnorm(o, residual, L.norm2, eps)
            if L.router is None:
                mlp_out = self._dense_mlp(o, L, dt)
            else:
                side = core._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._dense_mlp(o, L, dt)
                # Gathering before the router is what keeps the four expert windows tiling the
                # routing space exactly once: every rank routes the identical token set, so a
                # token's top-8 ids agree across ranks and each id falls in exactly one window.
                o_pad = (
                    o
                    if not pad_rows
                    else nn.functional.pad(o, [0, 0, 0, pad_rows], mode="constant", value=0.0)
                )
                o_all = allgather(o_pad, None, core.dp_group)
                parts = []
                for chunk in torch.split(o_all, _moe_chunk_sizes(o_all.shape[0]), 0):
                    logits = cublas_mm(chunk, L.router)
                    topk_w, topk_ids = noaux_tc_op(
                        logits, L.router_bias, n_group, topk_group, topk, routed_scale
                    )
                    xq, xsf = fp4_quantize(chunk, L.moe_gscale, _SF_VEC, False, _SF_LINEAR)
                    parts.append(
                        fp4_block_scale_moe_runner(
                            routing_logits=None,
                            routing_bias=None,
                            hidden_states=xq,
                            hidden_states_scale=xsf.view(torch.float8_e4m3fn),
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
                            num_experts=num_experts,
                            top_k=topk,
                            n_group=None,
                            topk_group=None,
                            intermediate_size=moe_inter,
                            local_expert_offset=core.expert_offset,
                            local_num_experts=core.local_experts,
                            routed_scaling_factor=None,
                            routing_method_type=_ROUTING_METHOD_INERT,
                            do_finalize=True,
                            act_type=_ACT_TYPE_SWIGLU,
                            topk_weights=topk_w,
                            topk_ids=topk_ids,
                        )[0]
                    )
                routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
                routed_pad = reducescatter(routed_all, None, core.dp_group)
                routed = (
                    routed_pad
                    if not pad_rows
                    else torch.split(routed_pad, [num_tokens, pad_rows], 0)[0]
                )
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            mlp_out, residual = flashinfer_fused_add_rmsnorm(mlp_out, residual, L.norm_next, eps)
            x = mlp_out
        return x

    def _mtp_shared_mlp(self, x, mw: dict):
        """The module's shared expert: one bf16 SwiGLU MLP over this rank's own tokens."""
        return cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, mw["sh_gu"])), mw["sh_dn"])

    def _mtp_routed_experts(self, x, mw: dict, all_rank_num_tokens, rows, dt):
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
            topk_w, topk_ids = noaux_tc_op(
                cublas_mm(chunk, mw["router"], None, torch.float32),
                mw["router_bias"],
                cfg.n_group,
                cfg.topk_group,
                cfg.num_experts_per_tok,
                float(cfg.routed_scaling_factor),
            )
            parts.append(
                fused_moe(
                    input=chunk,
                    token_selected_experts=topk_ids,
                    token_final_scales=topk_w,
                    fc1_expert_weights=mw["fc1"],
                    fc1_expert_biases=None,
                    fc2_expert_weights=mw["fc2"],
                    fc2_expert_biases=None,
                    output_dtype=dt,
                    quant_scales=[],
                    ep_size=core.ep_size,
                    ep_rank=core.ep_rank,
                    activation_type=_FUSED_MOE_ACT_SWIGLU,
                )[0]
            )
        routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        routed_pad = reducescatter(routed_all, None, core.dp_group)
        return routed_pad if not pad_rows else torch.split(routed_pad, [rows, pad_rows], 0)[0]

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """The draft step: a forward over the MTP module's operands, `core._mtp`.

        The same mirror of the trunk the MTP module is: its eleven attention-side weights are
        layer 61's, its experts are bf16. `shared_head` and its norm stay on `MTPLayer` -- the
        worker calls that entry point directly and it reads no phase.
        """
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
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling
        # Under a one-model MTP mode the engine raises the KV pool's layer count by
        # `num_nextn_predict_layers`, so this module addresses layer `num_hidden_layers` in the
        # same pool as the trunk's 61 layers. Nothing in the config stub or manifest says so.
        layer_idx = cfg.num_hidden_layers

        mw = core._mtp
        # Read every call, not once: the worker rewrites `attn_metadata` in place between
        # draft step 0 and step 1+, and `core._rope` may have grown on the engine's first step.
        rope = core._rope
        call = core._call
        md = attn_metadata
        step = _build_step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        num_ctx = md.num_contexts
        tc = md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - num_ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len
        ctx_kv_tokens = int(md.host_total_kv_lens[0])

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, mw["enorm"], eps)
        hn = flashinfer_rmsnorm(hidden_states, mw["hnorm"], eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), mw["eh"])

        residual = x
        xn = flashinfer_rmsnorm(x, mw["norm1"], eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_ctx, attn_gen = torch.split(attn_out, [tc, gen], 0)
        q = cublas_mm(flashinfer_rmsnorm(cublas_mm(xn, mw["qa"]), mw["q_norm"], eps), mw["qb"])
        kva = cublas_mm(xn, mw["kva"])
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, mw["kv_norm"], eps)
        latent = torch.cat([ckv, k_pe], -1)
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        latent_ctx, latent_gen = torch.split(latent, [tc, gen], 0)

        if core._cached_ctx:
            q_ctx = mla_rope_append_paged_kv_assign_q(
                q=q_ctx,
                latent_cache=latent_ctx,
                num_contexts=num_ctx,
                cu_ctx_cached_kv_lens=md.ctx_cached_token_indptr,
                cu_seq_lens=md.ctx_kv_indptr,
                max_input_uncached_seq_len=int(md.max_ctx_seq_len),
                cos_sin_cache=rope["rotary_cos_sin"],
                head_num=heads,
                nope_size=nope,
                rope_size=rope_dim,
                lora_size=kv_lora,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_orig_quant=None,
                residual_dim=_KV_RESIDUAL_DIM,
                layer_idx=layer_idx,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
            )
            ckv_full, k_pe_full = load_paged_kv_cache_for_mla(
                out_dtype=dt,
                num_contexts=num_ctx,
                num_ctx_kv_tokens=ctx_kv_tokens,
                max_ctx_kv_len=int(md.max_ctx_kv_len),
                cu_ctx_kv_lens=md.ctx_kv_indptr,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_quant_orig=None,
                layer_idx=layer_idx,
                kv_lora_rank=kv_lora,
                qk_rope_head_dim=rope_dim,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
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
        attn_ctx = thop_attention(
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
            **call,
        )

        if gen:
            q3 = torch.reshape(q_gen, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
            fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
            fq_nope_t = torch.transpose(fq_nope, 0, 1)
            fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=mw["k_b"], out=fq_nope_t)
            cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
            counter = torch.empty([1], dtype=torch.uint32, device=dev)
            quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
            bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
            bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
            fused_q = mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent_gen,
                rotary_cos_sin=rope["rotary_cos_sin"],
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
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=layer_idx,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
                q_scaling=q_scaling,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=v_dim,
                rope_append=True,
            )
            lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
            lat_out = thop_attention(
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
                **call,
            )
            lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
            attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
            attn_gen_t = bmm_out(a=lat_out_t, b=mw["v_b_t"], out=attn_gen_t)

        o = cublas_mm(attn_out, mw["o"])
        o, residual = flashinfer_fused_add_rmsnorm(o, residual, mw["norm2"], eps)
        shared = self._mtp_shared_mlp(o, mw)
        routed = self._mtp_routed_experts(o, mw, all_rank_num_tokens, rows, dt)
        return torch.add(residual, torch.add(routed, shared))


class DecodeTarget(Target):
    """The specialization: routed to only when there are no context rows.

    Calls four fewer ops than the prefill target -- the context attention flavor, the
    kv_b projection and the two paged-KV helpers all serve a path this target's routing
    guarantees never runs.
    """

    def __init__(self, core: "DeepseekV3Dep4ModelingV2") -> None:
        """Take the per-layer weight views this target's forward reads, once, off the real
        weights. The MTP operands are read off `core._mtp` at the call instead."""
        super().__init__(core)
        self._norm0 = core.w["l0_norm1"]
        self._layers = _layer_views(core)

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The batch state both MLA calls of one step share.

        States the two context counts instead of reading them back: this target runs
        only when `num_contexts` is zero, so reading them would ask the metadata to
        confirm what the routing already decided.
        """
        step = _build_step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

    def _dense_mlp(self, x, L: _Layer, dt):
        """One NVFP4 SwiGLU MLP over this rank's own tokens, used for both the dense layers' MLP
        and every MoE layer's shared expert."""
        xq, xsf = fp4_quantize(x, L.mlp_gu_gscale, _SF_VEC, False, _SF_SWIZZLED)
        gu = nvfp4_gemm(xq, L.mlp_gu_w, xsf, L.mlp_gu_s, L.mlp_gu_alpha, dt)
        act = flashinfer_silu_and_mul(gu)
        aq, asf = fp4_quantize(act, L.mlp_dn_gscale, _SF_VEC, False, _SF_SWIZZLED)
        return nvfp4_gemm(aq, L.mlp_dn_w, asf, L.mlp_dn_s, L.mlp_dn_alpha, dt)

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
        num_experts = cfg.n_routed_experts
        topk = cfg.num_experts_per_tok
        moe_inter = cfg.moe_intermediate_size
        n_group = cfg.n_group
        topk_group = cfg.topk_group
        routed_scale = float(cfg.routed_scaling_factor)
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling

        md = attn_metadata
        rope = core._rope
        call = core._call
        step = self.step_args(md)
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        gen = num_tokens
        dt = h.dtype
        dev = h.device
        # Query rows per generation sequence (the whole draft chain under MTP): the only thing
        # that tells the attention ops the query block is taller, and what aligns the
        # bottom-right within-block causal mask.
        gen_seqs = md.num_seqs
        gen_p = gen // gen_seqs if gen_seqs else 1
        dp_rows = core._dp_rows(md, num_tokens)
        pad_rows = dp_rows - num_tokens

        attn_out = torch.empty([num_tokens, heads * v_dim], dtype=dt, device=dev)
        attn_gen = attn_out
        # One set of attention scratch for all layers: every tensor here is fully
        # rewritten by layer i's mla_rope_generation before layer i's attention reads
        # it, and the whole attention path runs on the main stream, so layer i+1 cannot
        # overwrite what layer i still reads. 8 allocations per step, not 8 x 61.
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, L in enumerate(self._layers):
            q = cublas_mm(flashinfer_rmsnorm(cublas_mm(x, L.qa), L.q_norm, eps), L.qb)
            kva = cublas_mm(x, L.kva)
            ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
            ckv = flashinfer_rmsnorm(ckv_raw, L.kv_norm, eps)
            latent = torch.cat([ckv, k_pe], -1)

            q3 = torch.reshape(q, [gen, heads, qk_dim])
            q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
            # Over an fp8 pool `mla_rope_generation` reads this half to build the quantized
            # query, so this BMM must be issued first on the same stream; overlapping them
            # is a silent race (they only write disjoint halves on a bf16 pool).
            fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=L.k_b, out=fq_nope_t)
            fused_q = mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                latent_cache=latent,
                rotary_cos_sin=rope["rotary_cos_sin"],
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1_scale,
                mla_bmm2_scale=bmm2_scale,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=0,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_ptrs,
                host_kv_cache_pool_mapping=pool_map,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=i,
                num_heads=heads,
                num_kv_heads=1,
                head_size=lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=tokens_per_block,
                attention_window_size=window,
                beam_width=1,
                quant_mode=quant_mode,
                q_scaling=q_scaling,
                q_lora_rank=q_lora,
                kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope,
                qk_rope_head_dim=rope_dim,
                v_head_dim=kv_lora,
                rope_append=True,
            )
            lat_out = thop_attention(
                q=torch.reshape(fused_q, [gen, heads * lat_dim]),
                k=None,
                v=None,
                output=lat_out,
                latent_cache=latent,
                q_pe=q_pe,
                local_layer_idx=i,
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
                **call,
            )
            attn_gen_t = bmm_out(a=lat_out_t, b=L.v_b_t, out=attn_gen_t)

            o = cublas_mm(attn_out, L.o)
            o, residual = flashinfer_fused_add_rmsnorm(o, residual, L.norm2, eps)
            if L.router is None:
                mlp_out = self._dense_mlp(o, L, dt)
            else:
                side = core._side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._dense_mlp(o, L, dt)
                # Gathering before the router is what keeps the four expert windows tiling the
                # routing space exactly once: every rank routes the identical token set, so a
                # token's top-8 ids agree across ranks and each id falls in exactly one window.
                o_pad = (
                    o
                    if not pad_rows
                    else nn.functional.pad(o, [0, 0, 0, pad_rows], mode="constant", value=0.0)
                )
                o_all = allgather(o_pad, None, core.dp_group)
                parts = []
                for chunk in torch.split(o_all, _moe_chunk_sizes(o_all.shape[0]), 0):
                    logits = cublas_mm(chunk, L.router)
                    topk_w, topk_ids = noaux_tc_op(
                        logits, L.router_bias, n_group, topk_group, topk, routed_scale
                    )
                    xq, xsf = fp4_quantize(chunk, L.moe_gscale, _SF_VEC, False, _SF_LINEAR)
                    parts.append(
                        fp4_block_scale_moe_runner(
                            routing_logits=None,
                            routing_bias=None,
                            hidden_states=xq,
                            hidden_states_scale=xsf.view(torch.float8_e4m3fn),
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
                            num_experts=num_experts,
                            top_k=topk,
                            n_group=None,
                            topk_group=None,
                            intermediate_size=moe_inter,
                            local_expert_offset=core.expert_offset,
                            local_num_experts=core.local_experts,
                            routed_scaling_factor=None,
                            routing_method_type=_ROUTING_METHOD_INERT,
                            do_finalize=True,
                            act_type=_ACT_TYPE_SWIGLU,
                            topk_weights=topk_w,
                            topk_ids=topk_ids,
                        )[0]
                    )
                routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
                routed_pad = reducescatter(routed_all, None, core.dp_group)
                routed = (
                    routed_pad
                    if not pad_rows
                    else torch.split(routed_pad, [num_tokens, pad_rows], 0)[0]
                )
                main.wait_stream(side)
                mlp_out = torch.add(routed, shared)
            mlp_out, residual = flashinfer_fused_add_rmsnorm(mlp_out, residual, L.norm_next, eps)
            x = mlp_out
        return x

    def _mtp_shared_mlp(self, x, mw: dict):
        """The module's shared expert: one bf16 SwiGLU MLP over this rank's own tokens."""
        return cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, mw["sh_gu"])), mw["sh_dn"])

    def _mtp_routed_experts(self, x, mw: dict, all_rank_num_tokens, rows, dt):
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
            topk_w, topk_ids = noaux_tc_op(
                cublas_mm(chunk, mw["router"], None, torch.float32),
                mw["router_bias"],
                cfg.n_group,
                cfg.topk_group,
                cfg.num_experts_per_tok,
                float(cfg.routed_scaling_factor),
            )
            parts.append(
                fused_moe(
                    input=chunk,
                    token_selected_experts=topk_ids,
                    token_final_scales=topk_w,
                    fc1_expert_weights=mw["fc1"],
                    fc1_expert_biases=None,
                    fc2_expert_weights=mw["fc2"],
                    fc2_expert_biases=None,
                    output_dtype=dt,
                    quant_scales=[],
                    ep_size=core.ep_size,
                    ep_rank=core.ep_rank,
                    activation_type=_FUSED_MOE_ACT_SWIGLU,
                )[0]
            )
        routed_all = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        routed_pad = reducescatter(routed_all, None, core.dp_group)
        return routed_pad if not pad_rows else torch.split(routed_pad, [rows, pad_rows], 0)[0]

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """The draft step over generation rows only: a forward over the MTP module's
        operands, `core._mtp`, with the context path dropped."""
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
        quant_mode = core.quant_mode
        q_scaling = core.q_scaling
        # Under a one-model MTP mode the engine raises the KV pool's layer count by
        # `num_nextn_predict_layers`, so this module addresses layer `num_hidden_layers` in the
        # same pool as the trunk's 61 layers. Nothing in the config stub or manifest says so.
        layer_idx = cfg.num_hidden_layers

        mw = core._mtp
        # Read every call, not once: the worker rewrites `attn_metadata` in place between
        # draft step 0 and step 1+, and `core._rope` may have grown on the engine's first step.
        rope = core._rope
        call = core._call
        md = attn_metadata
        step = _build_step_args(md)
        rows = hidden_states.shape[0]
        dt = hidden_states.dtype
        dev = hidden_states.device
        gen = rows
        gen_seqs = md.num_seqs
        gen_p = gen // gen_seqs if gen_seqs else 1
        tokens_per_block = md.tokens_per_block
        block_offsets = md.kv_cache_block_offsets
        pool_ptrs = md.host_kv_cache_pool_pointers
        pool_map = md.host_kv_cache_pool_mapping
        window = md.max_seq_len

        e = nn.functional.embedding(input_ids, embed_tokens)
        en = flashinfer_rmsnorm(e, mw["enorm"], eps)
        hn = flashinfer_rmsnorm(hidden_states, mw["hnorm"], eps)
        halves = [en, hn] if _MTP_EMBED_BLOCK_FIRST else [hn, en]
        x = cublas_mm(torch.cat(halves, -1), mw["eh"])

        residual = x
        xn = flashinfer_rmsnorm(x, mw["norm1"], eps)
        attn_out = torch.empty([rows, heads * v_dim], dtype=dt, device=dev)
        attn_gen = attn_out
        q = cublas_mm(flashinfer_rmsnorm(cublas_mm(xn, mw["qa"]), mw["q_norm"], eps), mw["qb"])
        kva = cublas_mm(xn, mw["kva"])
        ckv_raw, k_pe = torch.split(kva, [kv_lora, rope_dim], -1)
        ckv = flashinfer_rmsnorm(ckv_raw, mw["kv_norm"], eps)
        latent = torch.cat([ckv, k_pe], -1)

        q3 = torch.reshape(q, [gen, heads, qk_dim])
        q_nope, q_pe = torch.split(q3, [nope, rope_dim], -1)
        fused_q = torch.empty([gen, heads, lat_dim], dtype=dt, device=dev)
        fq_nope, _ = torch.split(fused_q, [kv_lora, rope_dim], -1)
        fq_nope_t = torch.transpose(fq_nope, 0, 1)
        fq_nope_t = bmm_out(a=torch.transpose(q_nope, 0, 1), b=mw["k_b"], out=fq_nope_t)
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        quant_q = torch.empty([gen, heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        bmm1_scale = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2_scale = torch.empty([1], dtype=torch.float32, device=dev)
        fused_q = mla_rope_generation(
            fused_q=fused_q,
            q_pe=q_pe,
            latent_cache=latent,
            rotary_cos_sin=rope["rotary_cos_sin"],
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1_scale,
            mla_bmm2_scale=bmm2_scale,
            quant_q_buffer=quant_q,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            num_contexts=0,
            kv_cache_block_offsets=block_offsets,
            host_kv_cache_pool_pointers=pool_ptrs,
            host_kv_cache_pool_mapping=pool_map,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            kv_cache_scale_orig_quant=None,
            out_scale=None,
            block_ids_per_seq=None,
            helix_tensor_params=[None, None],
            predicted_tokens_per_seq=gen_p,
            layer_idx=layer_idx,
            num_heads=heads,
            num_kv_heads=1,
            head_size=lat_dim,
            residual_dim=_KV_RESIDUAL_DIM,
            tokens_per_block=tokens_per_block,
            attention_window_size=window,
            beam_width=1,
            quant_mode=quant_mode,
            q_scaling=q_scaling,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope_dim,
            v_head_dim=v_dim,
            rope_append=True,
        )
        lat_out = torch.empty([gen, heads * kv_lora], dtype=dt, device=dev)
        lat_out = thop_attention(
            q=torch.reshape(fused_q, [gen, heads * lat_dim]),
            k=None,
            v=None,
            output=lat_out,
            latent_cache=latent,
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
            **call,
        )
        lat_out_t = torch.transpose(torch.reshape(lat_out, [gen, heads, kv_lora]), 0, 1)
        attn_gen_t = torch.transpose(torch.reshape(attn_gen, [gen, heads, v_dim]), 0, 1)
        attn_gen_t = bmm_out(a=lat_out_t, b=mw["v_b_t"], out=attn_gen_t)

        o = cublas_mm(attn_out, mw["o"])
        o, residual = flashinfer_fused_add_rmsnorm(o, residual, mw["norm2"], eps)
        shared = self._mtp_shared_mlp(o, mw)
        routed = self._mtp_routed_experts(o, mw, all_rank_num_tokens, rows, dt)
        return torch.add(residual, torch.add(routed, shared))


class MTPLayer:
    """The checkpoint's multi-token-prediction module, at layer index `num_hidden_layers`,
    replayed once per draft step.

    It differs from a trunk layer in three ways the code alone does not show: an `eh_proj` front
    end that mixes the next token's embedding into `h`, bf16 experts (the checkpoint excludes
    `model.layers.61*` from NVFP4, so they go to `fused_moe`, not the trtllm-gen runner), and an
    output returned **un-normalized** — `shared_head` applies the module's own norm, and the
    runtime feeds the un-normalized tensor back in as the next step's `h`."""

    def __init__(self, core: DeepseekV3Dep4ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own output head: its own RMS norm — a distinct parameter from the
        trunk's `model.norm` — then the trunk's `lm_head`."""
        core = self.core
        return self.logits_processor.forward(
            flashinfer_rmsnorm(
                hidden_states,
                core._mtp["head_norm"],
                core.model_config.pretrained_config.rms_norm_eps,
            ),
            lm_head,
            attn_metadata,
            return_context_logits,
        )

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Route the draft step by the phase this call's metadata shows.

        Per call, not per engine forward: the worker rewrites `attn_metadata` in place
        between draft step 0 and step 1+, so one engine step's N draft calls can cross
        from prefill to decode.
        """
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
        )


class DraftModel:
    """The container the spec worker reaches this target's drafter through.

    `embed_tokens` must resolve through the core on every read: this container is built while
    every parameter is still a meta tensor, and the engine materializes the registry by
    replacing the tensor objects, so a reference snapshotted here stays on meta."""

    def __init__(self, core: DeepseekV3Dep4ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2DeepseekR10528Nvfp4Sm103Dep4(
    DecoderModelForCausalLM[DeepseekV3Dep4ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            DeepseekV3Dep4ModelingV2(model_config),
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
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
