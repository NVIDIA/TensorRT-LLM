# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every size the target computes with, read once from the text config and the mapping.

The checkpoint is the composite ``Glm5NextForConditionalGeneration`` release; the text
model's fields sit on ``text_config``. Shares are this rank's under tensor parallelism 4:
attention heads (KDA and MLA query heads) and FFN intermediates are split over the four
ranks, the latent and indexer state and every norm are replicated.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PretrainedConfig

from tensorrt_llm.mapping import Mapping

from .schedule import LayerSchedule

#: The checkpoint's key prefix of the text model.
TEXT_PREFIX = "model.language_model."


@dataclass(frozen=True)
class Dims:
    tp_size: int
    tp_rank: int
    hidden: int
    vocab: int
    num_layers: int  # decoder layers; the MTP layer is index num_layers
    schedule: LayerSchedule
    rms_eps: float
    # KDA (linear attention)
    kda_heads: int  # total
    kda_head_dim: int
    kda_conv_size: int
    kda_gate_lower_bound: float
    # Sparse MLA
    mla_heads: int  # total query heads
    q_lora: int
    kv_lora: int
    qk_nope: int
    v_head: int
    index_heads: int
    index_head_dim: int
    index_kpool: int
    index_topk: int
    index_always_select_tail: bool
    # FFN
    intermediate: int  # dense MLP, total
    moe_intermediate: int  # routed and shared expert, total
    num_experts: int
    top_k: int
    routed_scaling: float
    swiglu_limit: float
    # mHC
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    mtp: bool  # the MTP layer is built (one-model MTP is configured)
    max_num_tokens: int  # the engine's per-step token budget (autotuning range of the MoE runner)

    # -- this rank's shares ------------------------------------------------------------

    @property
    def kda_heads_local(self) -> int:
        return self.kda_heads // self.tp_size

    @property
    def kda_channels_local(self) -> int:
        return self.kda_heads_local * self.kda_head_dim

    @property
    def mla_heads_local(self) -> int:
        return self.mla_heads // self.tp_size

    @property
    def intermediate_local(self) -> int:
        return self.intermediate // self.tp_size

    @property
    def moe_intermediate_local(self) -> int:
        return self.moe_intermediate // self.tp_size

    @property
    def vocab_local(self) -> int:
        return -(-self.vocab // self.tp_size)

    @property
    def select_k(self) -> int:
        """Pools the indexer selects per row."""
        return self.index_topk // self.index_kpool

    @property
    def mtp_layer(self) -> int:
        return self.num_layers

    def layer_prefix(self, layer: int) -> str:
        return f"{TEXT_PREFIX}layers.{layer}."

    @classmethod
    def from_config(
        cls, text: PretrainedConfig, mapping: Mapping, mtp: bool, max_num_tokens: int
    ) -> "Dims":
        """The sizes of a config ``routing.fp8_model_refused`` admits (the MoE routing, the
        geometry the kernels compile in and the KDA gate bound are checked there)."""
        linear = dict(text.linear_attn_config)
        return cls(
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            hidden=int(text.hidden_size),
            vocab=int(text.vocab_size),
            num_layers=int(text.num_hidden_layers),
            schedule=LayerSchedule.from_text_config(text),
            rms_eps=float(text.rms_norm_eps),
            kda_heads=int(linear["num_heads"]),
            kda_head_dim=int(linear["head_dim"]),
            kda_conv_size=int(linear["short_conv_kernel_size"]),
            kda_gate_lower_bound=float(linear["gate_lower_bound"]),
            mla_heads=int(text.num_attention_heads),
            q_lora=int(text.q_lora_rank),
            kv_lora=int(text.kv_lora_rank),
            qk_nope=int(text.qk_nope_head_dim),
            v_head=int(text.v_head_dim),
            index_heads=int(text.index_n_heads),
            index_head_dim=int(text.index_head_dim),
            index_kpool=int(text.index_kpool),
            index_topk=int(text.index_topk),
            index_always_select_tail=bool(text.index_kpool_always_select_tail),
            intermediate=int(text.intermediate_size),
            moe_intermediate=int(text.moe_intermediate_size),
            num_experts=int(text.n_routed_experts),
            top_k=int(text.num_experts_per_tok),
            routed_scaling=float(text.routed_scaling_factor),
            swiglu_limit=float(text.swiglu_limit),
            hc_mult=int(text.hc_mult),
            hc_sinkhorn_iters=int(text.hc_sinkhorn_iters),
            hc_eps=float(text.hc_eps),
            mtp=mtp,
            max_num_tokens=int(max_num_tokens),
        )
