# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MiniMax-M3 text language-model.

Layers 0-2 are dense attention + dense MLP. Layers 3-59 are sparse
attention (MiniMax index-K block selector + sparse GQA,
``disable_index_value=True`` for every layer in the M3 checkpoint) +
MoE (top-4 of 128 routed experts plus one shared expert, sigmoid
routing with bias, ``routed_scaling_factor=2.0``, ``swigluoai``
activation). Q/K are per-head Gemma-RMSNormed before partial RoPE
(``rotary_dim=64`` of ``head_dim=128``, ``rope_theta=5e6``).
"""

from __future__ import annotations

import copy
import dataclasses
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from typing import Mapping as TMapping

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig

from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import AllReduce, AllReduceParams, MiniMaxAllReduceRMS
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM3MoeRoutingMethod, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode, copy_weight, load_weight_shard
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..utils import (
    ActivationType,
    AuxStreamType,
    EventType,
    get_model_extra_attrs,
    is_torch_compiling,
)
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, ModelConfig, register_auto_model

# Dense layers use SDPA with non-contiguous Q/K/V and a bool attn_mask.
# Limit backends to memory-efficient and math; cuDNN SDPA fails for this layout,
# and flash SDPA does not accept attn_mask.
_DENSE_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

# ---------------------------------------------------------------------------
# Config normalization helpers
# ---------------------------------------------------------------------------


def is_minimax_m3_vl_config(pretrained_config: PretrainedConfig) -> bool:
    """Return True if ``pretrained_config`` is a MiniMax-M3 VL config."""
    model_type = getattr(pretrained_config, "model_type", None)
    if model_type == "minimax_m3_vl":
        return True
    if model_type == "minimax_m3":
        return False
    architectures = getattr(pretrained_config, "architectures", None) or []
    return any("ForConditionalGeneration" in a for a in architectures)


def _wrap_dict_as_config(d: Dict[str, Any]) -> PretrainedConfig:
    """Wrap a plain ``dict`` text_config as a ``PretrainedConfig`` instance.

    ``AutoConfig.from_pretrained`` on the M3 checkpoint returns the VL
    config, but the nested ``text_config`` arrives as a raw ``dict``
    because the M3 ``configuration_minimax_m3_vl.py`` does not register a
    typed ``text_config_class``. The rest of this module expects attribute
    access (``cfg.num_hidden_layers``), so we materialise a
    ``PretrainedConfig`` and copy every key onto it.
    """
    out = PretrainedConfig()
    for k, v in d.items():
        setattr(out, k, v)
    return out


def get_text_config(pretrained_config: PretrainedConfig) -> PretrainedConfig:
    """Extract the language-model text config from a possibly-VL config.

    Returns the text config unchanged when ``pretrained_config`` is
    already a text config; raises if a VL config is missing
    ``text_config``. When the VL wrapper stores ``text_config`` as a
    plain ``dict`` (the M3 checkpoint case), it is materialised into a
    ``PretrainedConfig`` so downstream attribute access works.
    """
    if not is_minimax_m3_vl_config(pretrained_config):
        return pretrained_config
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None:
        raise ValueError(
            "MiniMax-M3 VL config has no `text_config`; expected the text "
            "language model under config.text_config"
        )
    if isinstance(text_config, dict):
        text_config = _wrap_dict_as_config(text_config)
    # Propagate fields that some loaders expect on the text config.
    if (
        getattr(text_config, "torch_dtype", None) is None
        and getattr(pretrained_config, "torch_dtype", None) is not None
    ):
        text_config.torch_dtype = pretrained_config.torch_dtype
    if (
        getattr(text_config, "tie_word_embeddings", None) is None
        and getattr(pretrained_config, "tie_word_embeddings", None) is not None
    ):
        text_config.tie_word_embeddings = pretrained_config.tie_word_embeddings
    return text_config


def get_text_model_config(
    model_config: "ModelConfig[PretrainedConfig]",
) -> "ModelConfig[PretrainedConfig]":
    """Return a ``ModelConfig`` whose pretrained_config is the M3 text config."""
    cfg = copy.deepcopy(model_config)
    text_cfg = get_text_config(cfg.pretrained_config)
    cfg = dataclasses.replace(cfg, pretrained_config=text_cfg)
    return cfg


def get_sparse_layer_ids(text_config: PretrainedConfig) -> Tuple[List[int], List[int]]:
    """Return ``(dense_layer_ids, sparse_layer_ids)`` for the M3 text model."""
    sparse_cfg = getattr(text_config, "sparse_attention_config", None)
    n_layers = int(text_config.num_hidden_layers)
    if sparse_cfg is None:
        return list(range(n_layers)), []
    if not sparse_cfg.get("use_sparse_attention", True):
        return list(range(n_layers)), []
    freq = sparse_cfg.get("sparse_attention_freq")
    if freq is None:
        return list(range(n_layers)), []
    if len(freq) != n_layers:
        raise ValueError(
            f"sparse_attention_freq length {len(freq)} does not match num_hidden_layers {n_layers}"
        )
    dense = [i for i, f in enumerate(freq) if int(f) == 0]
    sparse = [i for i, f in enumerate(freq) if int(f) != 0]
    return dense, sparse


def get_sparse_disable_index_value_layer_ids(text_config: PretrainedConfig) -> List[int]:
    """Return layer ids where the sparse index-value branch is disabled."""
    sparse_cfg = getattr(text_config, "sparse_attention_config", None)
    if sparse_cfg is None:
        return []
    flags = sparse_cfg.get("sparse_disable_index_value")
    if flags is None:
        return []
    return [i for i, f in enumerate(flags) if int(f) != 0]


def get_moe_layer_ids(text_config: PretrainedConfig) -> Tuple[List[int], List[int]]:
    """Return ``(dense_mlp_layer_ids, moe_layer_ids)`` for the M3 text model."""
    freq = getattr(text_config, "moe_layer_freq", None)
    n_layers = int(text_config.num_hidden_layers)
    if freq is None:
        return [], list(range(n_layers))
    if len(freq) != n_layers:
        raise ValueError(
            f"moe_layer_freq length {len(freq)} does not match num_hidden_layers {n_layers}"
        )
    dense = [i for i, f in enumerate(freq) if int(f) == 0]
    moe = [i for i, f in enumerate(freq) if int(f) != 0]
    return dense, moe


# Outside ``language_model.*``: deferred multimodal branches.
_VL_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _minimax_m3_swiglu_oai(gate_up: torch.Tensor, *, alpha: float, limit: float) -> torch.Tensor:
    """SwiGLU with asymmetric gate clamp + alpha scaling (SGLang's
    ``swiglu_no_interleaved_with_alpha_and_limit``)."""
    gate, up = gate_up.chunk(2, dim=-1)
    # Gate clamp is upper-side only (checkpoint contract — clamping
    # below -limit drifts vs SGLang for large-negative gates).
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


def _build_swiglu_oai_dense_mlp(
    model_config: "ModelConfig[PretrainedConfig]",
    intermediate_size: int,
    *,
    is_shared_expert: bool = False,
) -> GatedMLP:
    """Build a dense MLP using the MiniMax-M3 ``swigluoai`` activation.

    Layers 0-2 use this with ``dense_intermediate_size`` (the
    ``is_shared_expert=False`` path); the MoE shared expert uses it
    with ``shared_intermediate_size`` and ``is_shared_expert=True``.

    Under Attention DP (``mapping.enable_attention_dp=True``) each rank
    processes rank-local tokens independently. Sharding ``gate_up_proj``
    (COLUMN) and ``down_proj`` (ROW) across the global TP group and then
    all-reducing ``down_proj`` would mix outputs across independent
    rank-local token sets, producing wrong results. We collapse TP to
    1 via ``overridden_tp_size=1`` so the MLP is replicated and operates
    purely rank-locally (no sharding, no allreduce on ``down_proj``).

    For the MoE shared-expert path (``is_shared_expert=True``)
    ``reduce_output`` is forced ``False`` regardless of ADP: the routed
    branch is also constructed with ``reduce_results=False`` and the
    composition runs a single external AllReduce on
    ``routed + shared``, matching the DeepSeekV3 / GLM convention.
    """
    config = model_config.pretrained_config
    swiglu_alpha = float(getattr(config, "swiglu_alpha", 1.702))
    swiglu_limit = float(getattr(config, "swiglu_limit", 7.0))
    enable_adp = model_config.mapping.enable_attention_dp
    # Shared experts inside the MoE block defer the cross-rank reduction
    # to the MoE's single external AllReduce. The dense MLP path (no
    # MoE composition) carries its own reduction unless ADP collapses
    # TP to 1.
    reduce_output = False if is_shared_expert else (not enable_adp)
    return GatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        activation=partial(_minimax_m3_swiglu_oai, alpha=swiglu_alpha, limit=swiglu_limit),
        dtype=config.torch_dtype,
        config=model_config,
        overridden_tp_size=1 if enable_adp else None,
        reduce_output=reduce_output,
        is_shared_expert=is_shared_expert,
    )


def _resolve_minimax_m3_expert_size_per_partition(
    num_experts: int,
    mapping: Mapping,
    moe_load_balancer_config,
) -> int:
    """Compute the local expert/slot count the MoE module will resolve.

    Sizes the per-expert SwiGLU parameter tensors we hand to
    ``create_moe`` to match what the backend sees. Some backends assert
    ``swiglu_alpha.shape == (expert_size_per_partition,)`` at construct
    time; a mismatch surfaces as an opaque CUDA-side shape failure.

    Priority:
      1. EPLB config → ``num_slots // moe_ep_size``.
      2. Plain EP    → ``num_experts // moe_ep_size`` (with ``max(1, ...)``
                        for tiny test configs).
    """
    ep_size = mapping.moe_ep_size
    if moe_load_balancer_config is not None and moe_load_balancer_config.num_slots:
        # Mirror ``MoeLoadBalancerConfig.num_local_slots`` without
        # requiring ``.setup(ep_rank, ep_size)`` to have been called
        # (the ``num_local_slots`` property raises otherwise).
        return moe_load_balancer_config.num_slots // ep_size

    return max(1, num_experts // ep_size)


class MiniMaxM3Gate(nn.Module):
    """MiniMax-M3 router gate: float32 sigmoid scoring + per-expert bias correction.

    Owns the router projection ``weight`` and the per-expert
    ``e_score_correction_bias``, matching the DeepSeekV3 / Laguna /
    Qwen3 gate convention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        # Linear-style weight layout: ``(out_features, in_features)``
        # so the HF checkpoint's ``gate.weight`` (shape
        # ``[num_experts, hidden_size]``) loads as-is.
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.float32),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts,), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router runs in fp32 to match SGLang.
        return torch.nn.functional.linear(hidden_states.to(torch.float32), self.weight)

    def load_weights(self, weights: List[Dict]):
        """Load the router weight and the e_score_correction_bias.

        The HF MiniMax-M3 checkpoint stores the bias at
        ``block_sparse_moe.e_score_correction_bias`` (sibling of
        ``block_sparse_moe.gate.weight``), so
        :class:`MiniMaxM3ForCausalLM` rewrites it to
        ``block_sparse_moe.gate.e_score_correction_bias`` before the
        generic loader dispatches here. Both keys are tolerated as
        missing so callers can partial-load the gate (e.g. tests that
        only set the weight).
        """
        assert len(weights) == 1
        w = weights[0]
        if "weight" in w:
            self.weight.copy_(w["weight"][:].to(self.weight.dtype))
        if "e_score_correction_bias" in w:
            self.e_score_correction_bias.copy_(
                w["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
            )

    @property
    def routing_method(self) -> MiniMaxM3MoeRoutingMethod:
        # Pass a callable so routing always sees the latest parameter
        # value — important for DWDP redistribution and post-load
        # adjustments that mutate ``e_score_correction_bias`` in place.
        return MiniMaxM3MoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_experts,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )


class MiniMaxM3MoE(nn.Module):
    """M3 routed MoE block + shared expert.

    The routed experts use the ``swigluoai`` activation:
    ``gate_clamped * sigmoid(alpha * gate_clamped) * (up_clamped + 1.0)``
    with asymmetric ``gate.clamp(max=limit)`` and symmetric
    ``up.clamp(min=-limit, max=limit)``. This matches SGLang's
    ``swiglu_no_interleaved_with_alpha_and_limit`` shape. We plumb
    ``alpha``, ``beta=1.0`` (the ``up + 1`` offset), and ``limit`` per
    local expert into ``create_moe`` together with
    ``ActivationType.SwigluBias``, which dispatches the CUTLASS-family
    ``SwigluBiasAdaptor`` operator inside the fused MoE kernel.

    The routed branch and the shared expert both produce local partial
    outputs (``reduce_results=False`` / ``reduce_output=False``); their
    sum runs through a single external AllReduce only when not under
    Attention DP and ``tp_size > 1``. This matches the
    DeepSeekV3 / GLM convention and avoids the duplicate communication
    that the previous independent-reduction wiring incurred.
    """

    @staticmethod
    def _get_experts_quant_config(model_config: "ModelConfig", layer_idx: int) -> QuantConfig:
        """Return the per-layer quant config for the routed experts.

        For MIXED_PRECISION checkpoints (MXFP8 base + NVFP4 experts),
        ``ModelConfig._set_minimax_m3_moe_quant_config`` pre-populates
        ``quant_config_dict`` with coarse entries keyed by
        ``model.layers.N.block_sparse_moe.experts``.  Falls back to the
        global ``quant_config`` when no per-layer entry exists (e.g. BF16
        or user-supplied global NVFP4 config).
        """
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{layer_idx}.block_sparse_moe.experts",
            model_config.quant_config,
        )

    def __init__(
        self,
        model_config: "ModelConfig[PretrainedConfig]",
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.use_dp = self.enable_attention_dp
        self.mapping = model_config.mapping

        self.swiglu_alpha_value = float(getattr(config, "swiglu_alpha", 1.702))
        self.swiglu_beta_value = 1.0  # SGLang's ``(up + 1)`` offset in swiglu_no_interleaved.
        self.swiglu_limit_value = float(getattr(config, "swiglu_limit", 7.0))

        # Size the per-expert SwiGLU parameter tensors from the same
        # ``expert_size_per_partition`` the MoE module will resolve, not
        # from a hand-rolled ``num_slots // ep_size`` guess. EPLB and
        # DWDP can shift the local slot count (see
        # :func:`_resolve_minimax_m3_expert_size_per_partition` for the
        # priority order), and some backends assert
        # ``swiglu_alpha.shape == (expert_size_per_partition,)``.
        moe_load_balancer_config = model_config.moe_load_balancer
        self.expert_size_per_partition = _resolve_minimax_m3_expert_size_per_partition(
            num_experts=self.num_experts,
            mapping=model_config.mapping,
            moe_load_balancer_config=moe_load_balancer_config,
        )
        self.swiglu_alpha = torch.tensor(
            [self.swiglu_alpha_value] * self.expert_size_per_partition,
            dtype=torch.float32,
        ).cuda()
        self.swiglu_beta = torch.tensor(
            [self.swiglu_beta_value] * self.expert_size_per_partition,
            dtype=torch.float32,
        ).cuda()
        self.swiglu_limit = torch.tensor(
            [self.swiglu_limit_value] * self.expert_size_per_partition,
            dtype=torch.float32,
        ).cuda()

        # Router gate owns the float32 projection weight, the per-expert
        # ``e_score_correction_bias``, and the ``routing_method`` it
        # exposes to ``create_moe`` — same convention as DeepSeekV3 /
        # Laguna / Qwen3.
        self.gate = MiniMaxM3Gate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # Routed branch produces local partial outputs; the external
        # AllReduce below combines routed + shared in a single round
        # (matches DeepSeekV3). Under Attention DP the fused MoE already
        # skips its in-op all-reduce, so ``reduce_results=False`` is
        # also the correct flag there.
        experts_quant_config = MiniMaxM3MoE._get_experts_quant_config(model_config, layer_idx)
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            aux_stream_dict=aux_stream_dict,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
            override_quant_config=experts_quant_config,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            activation_type=ActivationType.SwigluBias,
        )
        # Defensive: if a future MoE-resolution path (new load-balancer
        # mode, new DWDP variant) shifts the local expert count in a
        # way our resolver doesn't yet model, fail here with a
        # diagnostic message instead of inside a CUDA-side shape
        # assertion deep in a backend kernel dispatch.
        resolved = getattr(self.experts, "expert_size_per_partition", None)
        assert resolved is None or resolved == self.expert_size_per_partition, (
            f"MiniMax-M3 SwiGLU sizing mismatch: pre-create_moe estimate "
            f"{self.expert_size_per_partition} != MoE-resolved {resolved}. "
            f"Update _resolve_minimax_m3_expert_size_per_partition to "
            f"match the MoE module's resolved layout."
        )

        # Shared expert: dense MLP fused into MoE output. Constructed
        # with ``is_shared_expert=True`` so ``reduce_output=False``
        # (the external AllReduce below performs the combined
        # reduction) and so the LoRA module types match the shared-
        # expert convention.
        n_shared = int(getattr(config, "n_shared_experts", 0) or 0)
        if n_shared > 0:
            shared_intermediate = (
                int(getattr(config, "shared_intermediate_size", None) or config.intermediate_size)
                * n_shared
            )
            self.shared_experts = _build_swiglu_oai_dense_mlp(
                model_config=model_config,
                intermediate_size=shared_intermediate,
                is_shared_expert=True,
            )
        else:
            self.shared_experts = None

        # External AllReduce on the combined routed + shared output.
        # Skipped under Attention DP (each rank's tokens are independent
        # so cross-rank reduction would mix them) and under tp_size==1
        # (nothing to reduce).
        self.allreduce = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

        # Aux stream + events for the routed-vs-shared parallel
        # execution path. The stream is only actually used when CUDA
        # graph multi-stream is enabled; otherwise
        # ``maybe_execute_in_parallel`` runs fn0/fn1 sequentially.
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        final_all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        def _compute_routed_output():
            router_logits = self.gate(hidden_states)
            return self.experts(
                hidden_states,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
            )

        def _compute_shared_output():
            return self.shared_experts(hidden_states)

        if self.shared_experts is None:
            result = _compute_routed_output()
        else:
            routed_output, shared_output = maybe_execute_in_parallel(
                _compute_routed_output,
                _compute_shared_output,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                self.aux_stream,
                disable_on_compile=True,
            )
            # In-place add into ``shared_output`` to avoid allocating a
            # temporary (matches DeepSeekV3 / GLM convention).
            result = shared_output.add_(routed_output)

        if self.allreduce is not None:
            result = self.allreduce(result, all_reduce_params=final_all_reduce_params)
        return result


class MiniMaxRMSNorm(nn.Module):
    """All-reduce RMSNorm used for TP-sharded Q/K hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        mapping: Mapping,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.mapping = mapping
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype), requires_grad=False)
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.all_reduce = AllReduce(mapping=self.mapping, strategy=AllReduceStrategy.NCCL)
        self.minimax_all_reduce_rms = MiniMaxAllReduceRMS(mapping=self.mapping)

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weight = load_weight_shard(
            weights[0]["weight"],
            tensor_parallel_size=self.mapping.tp_size,
            tensor_parallel_rank=self.mapping.tp_rank,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        copy_weight(self.weight, weight)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.contiguous()
        return self.minimax_all_reduce_rms(hidden_states, self.weight, self.eps)


def _extract_minimax_m3_attention_extra_attrs(layer_idx: str):
    """Resolve runtime metadata and the registered MiniMax-M3 layer."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Invalid MiniMax-M3 attention metadata"

    attn_layers = extra_attrs.get("attn_layers")
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()
    assert isinstance(attn_layer, MiniMaxM3Attention), "Invalid MiniMax-M3 attention layer"
    return metadata, attn_layer


@torch.library.custom_op("trtllm::minimax_m3_attn_custom_op_inplace", mutates_args=("output",))
def minimax_m3_attn_custom_op_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_q: Optional[torch.Tensor],
    idx_k: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Run MiniMax-M3 cache and attention work behind a compile boundary."""
    attn_metadata, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    num_tokens = attn_metadata.num_tokens
    attn_layer._attention_core(
        q[:num_tokens],
        k[:num_tokens],
        v[:num_tokens],
        idx_q[:num_tokens] if idx_q is not None else None,
        idx_k[:num_tokens] if idx_k is not None else None,
        attn_metadata,
        output[:num_tokens],
    )


class MiniMaxM3Attention(Attention):
    """M3 attention: dense (layers 0-2) or sparse (layers 3-59).

    Both branches share the same dense GQA scaffolding (``qkv_proj`` +
    ``o_proj`` + per-head Gemma Q/K norm + partial RoPE). Sparse layers
    additionally carry the MiniMax index branch (``index_q_proj``,
    ``index_k_proj`` and their per-head norms). The index value/output
    branch is omitted because the M3 checkpoint sets
    ``sparse_disable_index_value=True`` on every sparse layer; if a
    future config variant flips that flag the gate will catch the
    unmapped keys.
    """

    def __init__(
        self,
        *,
        model_config: "ModelConfig[PretrainedConfig]",
        layer_idx: Optional[int] = None,
        is_sparse_attention_layer: bool = False,
        disable_index_value: bool = False,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            rope_fusion=False,  # partial RoPE; rotate only rotary_dim of head_dim
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")
        if self.qk_norm_type != "per_head":
            raise ValueError(
                f"MiniMaxM3Attention only supports qk_norm_type='per_head', "
                f"got {self.qk_norm_type!r}"
            )
        self.use_gemma_norm = bool(getattr(config, "use_gemma_norm", False))
        self.head_dim_value = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )

        # Per-head Gemma RMSNorm — one set of weights shared across heads.
        self.q_norm = RMSNorm(
            hidden_size=self.head_dim_value,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=self.use_gemma_norm,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.head_dim_value,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=self.use_gemma_norm,
        )

        self.is_sparse_attention_layer = bool(is_sparse_attention_layer)
        self.disable_index_value = bool(disable_index_value)
        if self.is_sparse_attention_layer:
            sparse_cfg = getattr(config, "sparse_attention_config", None) or {}
            self.sparse_num_index_heads = int(sparse_cfg.get("sparse_num_index_heads", 4))
            self.sparse_index_dim = int(sparse_cfg.get("sparse_index_dim", 128))
            self.sparse_block_size = int(sparse_cfg.get("sparse_block_size", 128))
            self.sparse_topk_blocks = int(sparse_cfg.get("sparse_topk_blocks", 16))
            self.sparse_init_block = int(sparse_cfg.get("sparse_init_block", 0))
            self.sparse_local_block = int(sparse_cfg.get("sparse_local_block", 1))
            self.sparse_score_type = str(sparse_cfg.get("sparse_score_type", "max"))

            # index_q_proj is **replicated** across TP ranks. The sparse
            # forward reshapes idx_q to
            # ``[num_tokens, sparse_num_index_heads, sparse_index_dim]``,
            # which requires the rank-local idx_q to carry all heads.
            index_q_total = self.sparse_num_index_heads * self.sparse_index_dim
            self.index_q_proj = Linear(
                config.hidden_size,
                index_q_total,
                bias=False,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=None,
                quant_config=None,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            # index_k_proj is also replicated across TP ranks and
            # outputs ``sparse_index_dim`` channels — a single K per
            # token (not per-head), broadcast across index heads when
            # scoring blocks.
            self.index_k_proj = Linear(
                config.hidden_size,
                self.sparse_index_dim,
                bias=False,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=None,
                quant_config=None,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            # Per-head Gemma RMSNorm of width ``sparse_index_dim``;
            # applied to the projected index Q/K before partial RoPE in
            # the sparse forward path.
            self.index_q_norm = RMSNorm(
                hidden_size=self.sparse_index_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=self.use_gemma_norm,
            )
            self.index_k_norm = RMSNorm(
                hidden_size=self.sparse_index_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=self.use_gemma_norm,
            )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head Gemma RMSNorm on the main attention Q/K.

        Reshapes ``q`` (shape ``[..., num_heads * head_dim]``) and ``k``
        (shape ``[..., num_kv_heads * head_dim]``) to per-head rows of
        width ``head_dim``, applies the per-head RMSNorm
        (:attr:`q_norm` / :attr:`k_norm`, both ``use_gemma=True``), and
        reshapes back. Matches the SGLang reference's
        ``_qk_norm`` for ``qk_norm_type='per_head'``.
        """
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim_value)).reshape(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim_value)).reshape(k_shape)
        return q, k

    def apply_index_qk_norm(
        self, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head Gemma RMSNorm on the sparse-attention index Q/K.

        Mirrors :meth:`apply_qk_norm` for the index branch: reshapes
        ``idx_q`` (shape ``[..., num_index_heads * sparse_index_dim]``,
        possibly TP-sharded along the head axis) and ``idx_k`` (shape
        ``[..., sparse_index_dim]`` — single replicated head, not per
        index head) to per-head rows of width ``sparse_index_dim``,
        applies :attr:`index_q_norm` / :attr:`index_k_norm`, and reshapes
        back. The sparse forward path calls this on the projected index
        Q/K before partial RoPE.

        Raises:
            RuntimeError: if called on a dense layer (no index branch).
        """
        if not self.is_sparse_attention_layer:
            raise RuntimeError(
                f"apply_index_qk_norm is only valid on sparse attention layers "
                f"(layer_idx={self.layer_idx} is dense)"
            )
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = self.index_q_norm(idx_q.reshape(-1, self.sparse_index_dim)).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k.reshape(-1, self.sparse_index_dim)).reshape(idx_k_shape)
        return idx_q, idx_k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        """Run per-head QK norm before partial RoPE.

        The base ``Attention.apply_rope`` consumes split q/k/v. We split,
        apply per-head QK norm, then defer to the base partial-RoPE
        implementation (driven by ``RopeParams.dim < head_dim``).
        """
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)

    def forward(
        self,
        position_ids: Optional[torch.IntTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        **kwargs,
    ):
        """Dispatch sparse layers to the MiniMax-M3 sparse algorithm.

        Dense layers (0-2 in the M3 checkpoint) fall through to the base
        :class:`Attention` forward unchanged. Sparse layers (3+) drive
        the two-step block-sparse attention (index attention + top-k
        block selection + sparse GQA) implemented in
        :mod:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3`.

        The sparse forward expects ``attn_metadata.kv_cache_manager`` to
        be a :class:`MiniMaxM3KVCacheManagerV2` (the cache manager
        produced by
        :func:`get_sparse_attn_kv_cache_manager` for
        ``algorithm='minimax_m3'``). The cache manager owns both the
        standard paged main K/V buffer and a per-sparse-layer paged
        side index-K buffer.
        """
        if not self.is_sparse_attention_layer:
            return self._dense_forward(position_ids, hidden_states, attn_metadata, **kwargs)
        return self._sparse_forward(position_ids, hidden_states, attn_metadata, **kwargs)

    def _dense_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Dense MiniMax-M3 attention for layers 0-2.

        Mirrors :meth:`_sparse_forward` minus the index branch and minus
        the per-block top-k selection. Bypasses ``self.attn`` because the
        LLM API constructs ``MiniMaxM3SparseRuntimeBackend`` for every
        layer when ``sparse_attention_config=MiniMaxM3SparseAttentionConfig()``
        is set (the only way to get the matching :class:`MiniMaxM3KVCacheManagerV2`
        through the standard ``_create_kv_cache_manager`` factory), and
        that backend's ``forward`` strictly requires the M3 index branch
        and metadata which dense layers cannot supply.

        Steps:
          1. Project Q/K/V via fused ``qkv_proj``.
          2. Apply per-head Gemma RMSNorm to Q/K (same as
             :meth:`_sparse_forward` step 2 minus the index branch).
          3. Apply partial RoPE.
          4. Pull the paged main K/V cache from the M3 cache manager.
          5. Read the pre-built :class:`MiniMaxM3SparseAttentionMetadata`
             from ``attn_metadata.minimax_m3``. Production code paths
             build this attachment in
             :meth:`MiniMaxM3AttentionMetadata.prepare` (called by the
             pyexecutor outside any CUDA-graph capture window); test
             code paths attach it directly. The forward path **never**
             builds metadata itself and **never** migrates tensors
             between devices — both would trigger
             ``cudaErrorStreamCaptureUnsupported`` for the
             ``cuda_graph=True`` hard-path runs.
          6. Write new K/V to ``out_cache_loc`` slots.
          7. Gather per-request K/V padded to ``[batch, max_k]`` using
             :func:`_gather_paged_batched` and run standard dense GQA
             via :func:`torch.nn.functional.scaled_dot_product_attention`
             (causal for prefill, no-causal for decode).
          8. Apply ``o_proj``.
        """
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata; received None."
            )

        # 1. Projections (no index branch).
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. Per-head Gemma RMSNorm on Q/K (no index norm).
        q, k = self.apply_qk_norm(q, k)

        # 3. Partial RoPE on Q/K (no index branch).
        if self.rotary_emb is not None and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])

        # Keep token-wise projections and the output projection visible to
        # torch.compile. Only the metadata/cache-dependent attention core is
        # hidden behind the inplace custom op.
        o = self._forward_attention_core(q, k, v, None, None, attn_metadata)
        return self.o_proj(o)

    def _dense_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense cache updates and attention into ``output``."""
        from ..attention_backend.sparse.minimax_m3 import (
            _gather_paged_batched,
            _write_main_kv_slots_to_pool,
        )

        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata.kv_cache_manager to be a MiniMaxM3KVCacheManagerV2."
            )

        num_tokens = int(q.shape[0])
        q_view = q.view(num_tokens, self.num_heads, self.head_dim)
        k_view = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
        v_view = v.view(num_tokens, self.num_key_value_heads, self.head_dim)

        # 4. Paged-block main K/V pool. Keep the multi-dim view (do not
        # reshape) so writes propagate to the underlying pool storage:
        # ``kv_pool[:, 0]`` is a non-contiguous view (its dim-0 stride
        # is 2× the contiguous stride because dim 1 separates K from
        # V), so reshaping to ``[-1, num_kv_heads, head_dim]`` silently
        # forks a copy. Writing to the copy and then discarding it is
        # exactly the bug that drove dense layer-0 decode attention to
        # near-zero output: the next forward call would read zeros for
        # the prefilled positions because the pool was never updated.
        # Pass the 4-D view directly; the helpers below address slots
        # via ``(page, within)`` fancy indexing.
        kv_pool = kv_cache_manager.get_buffers(self.layer_idx)
        k_cache_view = kv_pool[:, 0]
        v_cache_view = kv_pool[:, 1]

        # 5. Read the pre-built M3 metadata. Building is the
        # responsibility of ``MiniMaxM3AttentionMetadata.prepare`` (or
        # the test path), so the forward stays CUDA-graph-capture safe.
        m3_attachment = getattr(attn_metadata, "minimax_m3", None)
        if m3_attachment is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata.minimax_m3 to be pre-built by "
                "MiniMaxM3AttentionMetadata.prepare(); the model_engine "
                "configures the M3 backend's Metadata class so this happens "
                "automatically. Test callers must attach minimax_m3 manually."
            )
        m3_meta = m3_attachment["metadata"]
        out_cache_loc = m3_attachment["out_cache_loc"]

        # 6. Write the new tokens' K/V to the pool. All inputs come
        # from prepare() which lands them on the cache device, so the
        # only ``.to(...)`` is a same-device dtype conversion (safe
        # under CUDA-graph capture).
        _write_main_kv_slots_to_pool(kv_pool, 0, out_cache_loc, k_view)
        _write_main_kv_slots_to_pool(kv_pool, 1, out_cache_loc, v_view)

        # 7. Gather padded K/V for every batch row and run dense GQA.
        batch = int(m3_meta.slot_ids.shape[0])
        max_k = int(m3_meta.max_seqlen_k)
        if max_k <= 0:
            max_k = 1
        # ``_gather_paged_batched`` decomposes the flat slot id into
        # ``(page, within)`` when handed a 4-D pool view; the result is
        # ``[batch, max_k, num_kv_heads, head_dim]`` exactly as the
        # legacy flat-slot path produced.
        k_padded = _gather_paged_batched(
            k_cache_view, m3_meta.req_to_token, m3_meta.slot_ids, max_k
        )
        v_padded = _gather_paged_batched(
            v_cache_view, m3_meta.req_to_token, m3_meta.slot_ids, max_k
        )

        # GQA via repeat_interleave on the KV head axis: each KV head is
        # shared by ``num_heads / num_kv_heads`` Q heads. SDPA does not
        # support GQA natively pre-PyTorch-2.4 on every backend, so we
        # expand K/V to match Q's head count. The expansion is O(num_q *
        # num_kv) memory; for the M3 geometry (Q=8, KV=1 per TP=8 rank)
        # it adds ~7x KV memory transiently — acceptable for the smoke
        # decode.
        if self.num_heads % max(self.num_key_value_heads, 1) != 0:
            raise RuntimeError(
                f"Dense GQA expects num_heads ({self.num_heads}) divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )
        group = self.num_heads // max(self.num_key_value_heads, 1)
        if group > 1:
            k_padded = k_padded.repeat_interleave(group, dim=2)
            v_padded = v_padded.repeat_interleave(group, dim=2)

        # Build the per-query attention mask that masks out padded KV
        # positions beyond each sequence's true ``seq_lens`` and (for
        # prefill) preserves causality. ``q_positions`` from the prefill
        # metadata names each Q token's K-side position; for decode
        # there is one Q token per request at position ``seq_lens - 1``.
        # The metadata tensors are produced by
        # :meth:`MiniMaxM3AttentionMetadata.prepare` on the cache
        # device, so ``.to(dtype=torch.long)`` is a same-device dtype
        # conversion (capture-safe).
        seq_lens_dev = m3_meta.seq_lens.to(dtype=torch.long)
        kv_positions = torch.arange(max_k, device=q.device).unsqueeze(0)  # [1, max_k]

        if m3_meta.is_prefill:
            # Prefill: build [total_q, max_k] mask using q_positions / q_batch_row.
            # Prefill never runs inside the CUDA-graph capture window
            # (capture is decode-only), so the per-batch Python loop and
            # the ``.tolist()`` below are acceptable.
            assert m3_meta.q_batch_row is not None and m3_meta.q_positions is not None, (
                "prefill metadata requires q_batch_row + q_positions; call .prepare()"
            )
            q_batch_row = m3_meta.q_batch_row.to(dtype=torch.long)
            q_positions = m3_meta.q_positions.to(dtype=torch.long)
            seq_lens_per_q = seq_lens_dev[q_batch_row]  # [total_q]
            # Each Q token attends to KV positions [0, q_position] (causal).
            # Also mask out KV positions >= seq_lens_per_q (padding).
            valid = (kv_positions <= q_positions.unsqueeze(-1)) & (
                kv_positions < seq_lens_per_q.unsqueeze(-1)
            )  # [total_q, max_k]
            # Build per-batch attention by routing each Q to its K via q_batch_row.
            # Easiest path: SDPA expects [batch, num_heads, q_len, head_dim].
            # We process each batch row independently to keep the math straightforward.
            output_view = output.view(-1, self.num_heads, self.head_dim)
            cu = m3_meta.cu_seqlens_q.to(torch.long).tolist()
            for b in range(batch):
                start, end = cu[b], cu[b + 1]
                if end <= start:
                    continue
                q_b = q_view[start:end].transpose(0, 1).unsqueeze(0)  # [1, H, q, d]
                k_b = k_padded[b].transpose(0, 1).unsqueeze(0)  # [1, H, k, d]
                v_b = v_padded[b].transpose(0, 1).unsqueeze(0)  # [1, H, k, d]
                mask_b = valid[start:end].unsqueeze(0).unsqueeze(0)  # [1, 1, q, k]
                # SDPA: when attn_mask is a bool tensor, True = attend, False = mask.
                with sdpa_kernel(_DENSE_SDPA_BACKENDS):
                    out_b = torch.nn.functional.scaled_dot_product_attention(
                        q_b.to(q.dtype),
                        k_b.to(q.dtype),
                        v_b.to(q.dtype),
                        attn_mask=mask_b,
                        dropout_p=0.0,
                        is_causal=False,
                    )  # [1, H, q, d]
                output_view[start:end].copy_(out_b.squeeze(0).transpose(0, 1))
        else:
            # Decode: one Q token per request at position seq_lens - 1.
            # Every input tensor here is already on q.device (set up by
            # prepare()), so SDPA captures cleanly.
            valid = kv_positions < seq_lens_dev.unsqueeze(-1)  # [batch, max_k]
            q_b = q_view.unsqueeze(1).transpose(1, 2)  # [batch, H, 1, d]
            k_b = k_padded.transpose(1, 2)  # [batch, H, k, d]
            v_b = v_padded.transpose(1, 2)  # [batch, H, k, d]
            mask_b = valid.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, k]
            with sdpa_kernel(_DENSE_SDPA_BACKENDS):
                out_b = torch.nn.functional.scaled_dot_product_attention(
                    q_b.to(q.dtype),
                    k_b.to(q.dtype),
                    v_b.to(q.dtype),
                    attn_mask=mask_b,
                    dropout_p=0.0,
                    is_causal=False,
                )  # [batch, H, 1, d]
            # Drop the singleton Q-length axis and write the resulting
            # ``[batch, num_heads, head_dim]`` tensor into the final buffer.
            # The prior ``.transpose(1, 2).reshape(batch, H, d)`` pattern
            # was wrong: with ``H != head_dim`` (M3 TP=8 has H=8, d=128)
            # the non-contiguous transpose forces ``reshape`` to copy the
            # data in C-order under its current ``[batch, d, H]`` shape,
            # then reinterpret as ``[batch, H, d]`` — which scrambles
            # ``(head, head_dim)`` ordering and feeds permuted activations
            # into ``o_proj``. Prefill is unaffected because its
            # ``transpose(0, 1)`` runs between q-len and num_heads axes
            # which the per-batch loop already laid out correctly.
            output.view(batch, self.num_heads, self.head_dim).copy_(out_b.squeeze(2))

        return output

    def _forward_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        output = q.new_empty((q.shape[0], self.num_heads * self.head_dim))
        if self.register_to_config and is_torch_compiling():
            minimax_m3_attn_custom_op_inplace(
                q,
                k,
                v,
                idx_q,
                idx_k,
                self.layer_idx_str,
                output,
            )
        else:
            self._attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        return output

    def _attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_sparse_attention_layer:
            assert idx_q is not None and idx_k is not None
            return self._sparse_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        assert idx_q is None and idx_k is None
        return self._dense_attention_core(q, k, v, attn_metadata, output)

    def _sparse_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Run a MiniMax-M3 sparse attention forward end-to-end.

        Steps:
          1. Project ``hidden_states`` to Q/K/V (fused ``qkv_proj``)
             plus index Q (per-head) and index K (single replicated).
          2. Apply per-head Gemma RMSNorm to both branches.
          3. Apply partial RoPE (``rotary_dim`` channels of ``head_dim``)
             to both branches.
          4. Pull paged main K/V cache (reshaped to flat-slot view) and
             paged side index-K cache from the
             :class:`MiniMaxM3KVCacheManagerV2`.
          5. Build a :class:`MiniMaxM3SparseAttentionMetadata` from the
             standard :class:`AttentionMetadata` (using ``request_ids``
             + ``seq_lens`` + ``num_cached_tokens_per_seq``).
          6. Write the new token's K/V/idx_K to the slots named by the
             per-token ``out_cache_loc`` derived from
             ``req_to_token``.
          7. Dispatch to :meth:`MiniMaxM3SparseRuntimeBackend.forward`
             which runs the sparse path end-to-end.
          8. Apply ``o_proj`` and return.

        Production callers (the LLM API path) drive
        :meth:`MiniMaxM3AttentionMetadata.prepare` outside any
        CUDA-graph capture window; that method attaches a pre-built
        :class:`MiniMaxM3SparseAttentionMetadata` and an
        ``out_cache_loc`` tensor as ``attn_metadata.minimax_m3``. Test
        callers attach the same dict manually.  This forward path
        always reads the pre-built attachment and never builds metadata
        itself — both would trigger
        ``cudaErrorStreamCaptureUnsupported`` for the
        ``cuda_graph=True`` hard path because the build does
        CPU->GPU copies.
        """
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata; received None."
            )
        # 1. Projections.
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        idx_q = self.index_q_proj(hidden_states)
        idx_k = self.index_k_proj(hidden_states)

        # 2. Per-head Gemma RMSNorm on both branches.
        q, k = self.apply_qk_norm(q, k)
        idx_q, idx_k = self.apply_index_qk_norm(idx_q, idx_k)

        # 3. Partial RoPE on both branches. The base ``Attention``
        # constructor created ``self.rotary_emb`` for the configured
        # partial ``rotary_dim`` because ``rope_fusion=False``.
        if self.rotary_emb is not None and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])
            idx_q, idx_k = self.rotary_emb(position_ids, [idx_q, idx_k])

        o = self._forward_attention_core(q, k, v, idx_q, idx_k, attn_metadata)
        return self.o_proj(o)

    def _sparse_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse cache updates and attention into ``output``."""
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata.kv_cache_manager to be a MiniMaxM3KVCacheManagerV2."
            )

        # 4. Get the paged-block main K/V cache + flat side index-K cache.
        # The base KVCacheManagerV2 layout is
        # ``[num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim]``
        # with NHD layout. Pass the multi-dim view ``kv_pool[:, kv_index]``
        # ``[num_pages, tokens_per_block, num_kv_heads, head_dim]`` directly:
        # ``_gather_paged_batched`` decomposes the flat slot id into
        # ``(page, within)`` for 4-D caches, and writes go through
        # :func:`_write_main_kv_slots_to_pool` which uses multi-dim
        # fancy assignment. The previously used reshape pattern
        # silently copied the K (or V) slice because ``kv_pool[:, 0]``
        # is non-contiguous, so writes never propagated to the pool —
        # the bug that drove the dense layer-0 decode attention to
        # near-zero output and produced the GSM8K-100 0.0 score.
        kv_pool = kv_cache_manager.get_buffers(self.layer_idx)
        # Index 0 = K, 1 = V on the kv_factor axis.
        k_cache = kv_pool[:, 0]
        v_cache = kv_pool[:, 1]
        if not hasattr(kv_cache_manager, "get_index_k_buffer"):
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires the "
                "kv_cache_manager to be a MiniMaxM3KVCacheManagerV2 "
                f"(got {type(kv_cache_manager).__name__})."
            )
        idx_k_cache = kv_cache_manager.get_index_k_buffer(self.layer_idx)
        idx_v_cache = (
            kv_cache_manager.get_index_v_buffer(self.layer_idx)
            if hasattr(kv_cache_manager, "get_index_v_buffer")
            else None
        )

        # 5. Read the pre-built M3 metadata. Building is the
        # responsibility of ``MiniMaxM3AttentionMetadata.prepare`` (or
        # the test path), so the forward stays CUDA-graph-capture safe.
        m3_attachment = getattr(attn_metadata, "minimax_m3", None)
        if m3_attachment is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata.minimax_m3 to be pre-built by "
                "MiniMaxM3AttentionMetadata.prepare(); the model_engine "
                "configures the M3 backend's Metadata class so this happens "
                "automatically. Test callers must attach minimax_m3 manually."
            )
        m3_meta = m3_attachment["metadata"]
        out_cache_loc = m3_attachment["out_cache_loc"]

        if not self.disable_index_value and idx_v_cache is not None:
            # The shared idx_v_proj is not part of the M3 checkpoint
            # bring-up (disable_index_value=True everywhere); future
            # variants would project idx_v from hidden_states the same
            # way idx_k is projected.
            raise NotImplementedError(
                "Index V projection is not implemented (M3 checkpoint sets "
                "disable_index_value=True on every sparse layer)."
            )

        # 6-7. Dispatch to the registered MiniMax-M3 sparse runtime
        # backend, which executes the sparse path end-to-end including
        # the cache writes. Production construction (LLM API with
        # ``sparse_attention_config=MiniMaxM3SparseAttentionConfig()``)
        # registers :class:`MiniMaxM3SparseRuntimeBackend` as
        # ``self.attn``; any other backend on a sparse layer is a
        # configuration error.
        from ..attention_backend.sparse.minimax_m3 import get_minimax_m3_attention_backend_cls

        m3_backend_cls = get_minimax_m3_attention_backend_cls()
        if not isinstance(self.attn, m3_backend_cls):
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                f"self.attn to be a MiniMaxM3SparseRuntimeBackend, got "
                f"{type(self.attn).__name__}. Construct the model with "
                "sparse_attention_config=MiniMaxM3SparseAttentionConfig(...) on "
                "ModelConfig so the standard attention-backend dispatch selects "
                "the M3 sparse runtime."
            )
        return self.attn.forward(
            q,
            k,
            v,
            None,
            output=output,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_v=None,  # disable_index_value=True for M3 checkpoint
            k_cache=k_cache,
            v_cache=v_cache,
            idx_k_cache=idx_k_cache,
            idx_v_cache=idx_v_cache,
            out_cache_loc=out_cache_loc,
            m3_metadata=m3_meta,
        )


class MiniMaxM3DecoderLayer(DecoderLayer):
    """One M3 transformer block (dense or sparse, MLP or MoE)."""

    def __init__(
        self,
        model_config: "ModelConfig[PretrainedConfig]",
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        _, sparse_layer_ids = get_sparse_layer_ids(config)
        disable_index_value_ids = set(get_sparse_disable_index_value_layer_ids(config))
        is_sparse = layer_idx in sparse_layer_ids
        disable_index_value = layer_idx in disable_index_value_ids

        self.self_attn = MiniMaxM3Attention(
            model_config=model_config,
            layer_idx=layer_idx,
            is_sparse_attention_layer=is_sparse,
            disable_index_value=disable_index_value,
        )

        _, moe_layer_ids = get_moe_layer_ids(config)
        if layer_idx in moe_layer_ids:
            self.block_sparse_moe = MiniMaxM3MoE(
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
            self.mlp = None
        else:
            dense_intermediate = int(
                getattr(config, "dense_intermediate_size", config.intermediate_size)
            )
            self.mlp = _build_swiglu_oai_dense_mlp(
                model_config=model_config, intermediate_size=dense_intermediate
            )
            self.block_sparse_moe = None

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=bool(getattr(config, "use_gemma_norm", False)),
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=bool(getattr(config, "use_gemma_norm", False)),
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.block_sparse_moe is not None:
            hidden_states = self.block_sparse_moe(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMaxM3Model(DecoderModel):
    """M3 text decoder model."""

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        super().__init__(model_config)
        quant_config = model_config.quant_config
        if quant_config is None or (
            (not quant_config.quant_mode.has_fp8_kv_cache())
            and (not quant_config.quant_mode.has_fp4_kv_cache())
        ):
            model_config.pretrained_config.torch_dtype = torch.bfloat16
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        # Two aux streams: one for MoE shared/routed parallel execution,
        # one for MoE chunking overlap inside the fused MoE kernel.
        # Matches the DeepSeekV3 convention.
        self.aux_stream_dict = {
            AuxStreamType.MoeShared: torch.cuda.Stream(),
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=bool(getattr(config, "use_gemma_norm", False)),
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# HF MiniMax-M3 stores the routed score-correction bias one level above
# the router weight (``block_sparse_moe.e_score_correction_bias``,
# sibling of ``block_sparse_moe.gate.weight``). The TRT-LLM module tree
# binds it to :class:`MiniMaxM3Gate`, so the generic loader expects to
# see it at ``block_sparse_moe.gate.e_score_correction_bias``. The
# regex below moves the key into the gate's prefix before the loader
# dispatches; this lets ``mark_consumed("...gate")`` cleanly remove
# both the weight and the bias together without disturbing the sibling
# ``block_sparse_moe.experts.*`` backend subtree.
_M3_GATE_BIAS_RENAME_MAP = {
    r"^(.*\.block_sparse_moe)\.e_score_correction_bias$": (r"\1.gate.e_score_correction_bias"),
}


@register_auto_model("MiniMaxM3SparseForCausalLM")
class MiniMaxM3ForCausalLM(DecoderModelForCausalLM[MiniMaxM3Model, PretrainedConfig]):
    """Text-only M3 model."""

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        raw_pretrained = model_config.pretrained_config
        if is_minimax_m3_vl_config(raw_pretrained):
            model_config = get_text_model_config(model_config)
        super().__init__(
            MiniMaxM3Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        # Merge the M3-specific gate-bias rename into any caller-
        # supplied ``params_map`` so the VL wrapper and any downstream
        # tooling that already passes one keep working.
        params_map = kwargs.pop("params_map", None) or {}
        merged = {**_M3_GATE_BIAS_RENAME_MAP, **params_map}
        return super().load_weights(weights, *args, params_map=merged, **kwargs)


def _strip_language_model_prefix(
    weights: TMapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Strip the ``language_model.`` prefix and drop deferred multimodal/MTP keys.

    Returns ``(text_weights, ignored)`` where ``ignored`` maps each
    deferred-or-multimodal source key to the documented reason it was
    skipped. ``text_weights`` is suitable to pass to the underlying
    ``DecoderModelForCausalLM`` weight loader.
    """
    text_weights: Dict[str, Any] = {}
    ignored: Dict[str, str] = {}
    for key, value in weights.items():
        if key.startswith(_VL_PREFIXES):
            for prefix in _VL_PREFIXES:
                if key.startswith(prefix):
                    ignored[key] = f"deferred multimodal branch: {prefix.rstrip('.')}"
                    break
            continue
        if not key.startswith("language_model."):
            # Unknown top-level keys are kept for the inner loader so it can
            # surface a clear error rather than us silently dropping them.
            text_weights[key] = value
            continue
        rest = key[len("language_model.") :]
        if rest.startswith("model.mtp."):
            ignored[key] = "MTP (multi-token prediction) not implemented"
            continue
        text_weights[rest] = value
    return text_weights, ignored


def _build_minimax_m3_vl_input_processor_registration():
    """Compose the ``register_input_processor`` decorator for the M3 VL class."""
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.registry import (
        MultimodalPlaceholderMetadata,
        MultimodalPlaceholderPlacement,
        register_input_processor,
    )

    from .modeling_minimaxm3_vl import get_minimax_m3_vl_input_processor_cls

    return register_input_processor(
        get_minimax_m3_vl_input_processor_cls(),
        model_type="minimax_m3_vl",
        placeholder_metadata=MultimodalPlaceholderMetadata(
            placeholder_map={
                "image": "]<]start of image[>[]<]image[>[]<]end of image[>[",
                "video": "]<]start of image[>[]<]video[>[]<]end of image[>[",
            },
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            content_format=ContentFormat.STRING,
        ),
    )


@register_auto_model("MiniMaxM3SparseForConditionalGeneration")
class MiniMaxM3VLForConditionalGeneration(MiniMaxM3ForCausalLM):
    """MiniMax-M3 VL: text decoder + vision tower + multimodal projector + patch merger.

    Wraps the text-only :class:`MiniMaxM3ForCausalLM` (which owns the
    language model and lm head) and adds a ``vision_tower`` attribute
    that owns the ViT, the multi-modal projector, and the patch-merge
    MLP. Weight loading splits the checkpoint into text and vision
    streams: text weights flow through ``language_model.``-prefix
    stripping into the BF16 weight loader; vision-side weights are
    routed to ``self.vision_tower`` after re-anchoring the standalone
    ``multi_modal_projector.*`` and ``patch_merge_mlp.*`` checkpoint
    blobs under that namespace.
    """

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        # Capture the raw VL config before ``super().__init__`` swaps in
        # the text subconfig so we can read ``vision_config`` /
        # ``projector_hidden_size`` for the vision tower construction.
        raw_pretrained = model_config.pretrained_config
        super().__init__(model_config)

        # Expose multimodal token IDs at the top-level model so the
        # production runtime (``model_engine._prepare_multimodal_indices``)
        # can locate the placeholder positions in flat in-flight-batched
        # ``input_ids`` via ``getattr(self.model, "mm_token_ids", None)``.
        # MiniMax-M3's image/video tokens are in-vocab so the engine's
        # OOV fallback (``input_ids >= vocab_size``) does not find them.
        # The HF config carries ``image_token_index`` / ``video_token_index``
        # — read them directly; a missing field is a checkpoint contract
        # violation.
        if not hasattr(raw_pretrained, "image_token_index"):
            raise ValueError("MiniMax-M3 VL config is missing required field 'image_token_index'")
        if not hasattr(raw_pretrained, "video_token_index"):
            raise ValueError("MiniMax-M3 VL config is missing required field 'video_token_index'")
        image_token_id = int(raw_pretrained.image_token_index)
        video_token_id = int(raw_pretrained.video_token_index)
        self.register_buffer(
            "mm_token_ids",
            torch.tensor(
                [image_token_id, video_token_id],
                dtype=torch.int32,
            ),
            persistent=False,
        )

        vision_config_raw = getattr(raw_pretrained, "vision_config", None)
        if vision_config_raw is None:
            # Pure-text checkpoint registered under the VL architecture
            # name. Stay backward compatible by exposing a None
            # vision_tower attribute.
            self.vision_tower = None
            self.last_loaded_vision_keys: List[str] = []
            self.last_missing_vision_keys: List[str] = []
            return

        from .modeling_minimaxm3_vl import CLIPVisionConfig, MiniMaxVLVisionModel

        vision_config = CLIPVisionConfig.from_dict_or_obj(vision_config_raw)
        text_cfg = get_text_config(raw_pretrained)
        text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
        projector_hidden_size = getattr(raw_pretrained, "projector_hidden_size", None)
        # Vision branch is BF16 in the checkpoint.
        vision_dtype = getattr(raw_pretrained, "torch_dtype", torch.bfloat16)
        if not isinstance(vision_dtype, torch.dtype):
            vision_dtype = getattr(torch, str(vision_dtype).split(".")[-1], torch.bfloat16)

        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            dtype=vision_dtype,
        )
        self.last_loaded_vision_keys = []
        self.last_missing_vision_keys = []

    def load_weights(self, weights, *args, **kwargs):
        text_cfg = self.config
        if is_minimax_m3_vl_config(text_cfg):
            text_cfg = get_text_config(text_cfg)

        # Split off the vision branches (``vision_tower.*`` plus
        # standalone ``multi_modal_projector.*`` / ``patch_merge_mlp.*``)
        # so the text loader only sees ``language_model.*`` + the small
        # set of unknown top-level keys it surfaces as errors.
        from .modeling_minimaxm3_vl import load_minimax_m3_vl_state_dict, split_multimodal_weights

        text_or_other_weights, vision_weights = split_multimodal_weights(weights)

        text_weights, ignored = _strip_language_model_prefix(text_or_other_weights)
        self.last_ignored_weight_keys: Dict[str, str] = ignored

        if self.vision_tower is not None and vision_weights:
            loaded, missing = load_minimax_m3_vl_state_dict(
                self.vision_tower, vision_weights, strict=False
            )
            self.last_loaded_vision_keys = loaded
            self.last_missing_vision_keys = missing

        return super().load_weights(text_weights, *args, **kwargs)

    def forward(
        self,
        attn_metadata: "AttentionMetadata",
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Multimodal-aware forward that merges visual features when present.

        When ``kwargs`` carries a ``multimodal_params`` list (TRT-LLM's
        standard plumbing for multimodal-bearing requests — see
        :class:`~tensorrt_llm.inputs.multimodal.MultimodalParams`), run
        the vision tower on the carried ``pixel_values`` /
        ``image_grid_thw`` (and the video equivalents) and splice the
        merged visual features into ``embed_tokens(input_ids)`` at the
        positions where the already-expanded placeholder tokens live.
        The fused tensor is then forwarded as ``inputs_embeds`` to the
        underlying text decoder so the per-position embedding the
        decoder sees at a placeholder index is the *visual* feature for
        that position, not the placeholder-token's learnt text embedding.

        When ``multimodal_params`` is absent (text-only request, or
        decode/generation phase after the multimodal embeddings were
        consumed during prefill), the call passes through to the text
        :meth:`MiniMaxM3ForCausalLM.forward` unchanged.

        Mirrors the Qwen2-VL pattern in TRT-LLM (see
        ``modeling_qwen2vl.py``): vision encoder runs locally,
        ``embed_tokens`` is invoked here, and the underlying causal LM
        is then driven with the fused ``inputs_embeds`` (``input_ids``
        is set to ``None`` so the text decoder does not re-embed).
        """
        multimodal_params = kwargs.pop("multimodal_params", None) or []
        # Filter to only the params that actually carry vision data; a
        # decode-phase batch frequently includes pure-text entries.
        mm_params_with_data: List[Any] = []
        if multimodal_params:
            for param in multimodal_params:
                if param is None:
                    continue
                data = getattr(param, "multimodal_data", None) or {}
                img = data.get("image") if isinstance(data, dict) else None
                vid = data.get("video") if isinstance(data, dict) else None
                has_image_pv = isinstance(img, dict) and img.get("pixel_values") is not None
                has_video_pv = isinstance(vid, dict) and (
                    vid.get("pixel_values_videos") is not None
                    or vid.get("pixel_values") is not None
                )
                if has_image_pv or has_video_pv:
                    mm_params_with_data.append(param)

        if (
            mm_params_with_data
            and self.vision_tower is not None
            and inputs_embeds is None
            and input_ids is not None
        ):
            # Use the existing TRT-LLM multimodal fusion contract
            # (``fuse_input_embeds``) so this works correctly under the
            # production LLM API flow, including the case where the
            # runtime has rewritten placeholder positions with per-item
            # radix-attention pad_value hashes (the
            # ``MultiModalityDataPaddingPatternMultimodalTokens``
            # contract). ``fuse_input_embeds`` consumes either explicit
            # ``mm_token_indices`` / ``text_token_indices`` provided by
            # the runtime via kwargs, or falls back to scanning
            # ``input_ids`` for ``mm_token_ids`` — exactly the right
            # semantics for visual-input requests.
            from .modeling_minimaxm3_vl import extract_multimodal_items_in_request_order
            from .modeling_multimodal_utils import fuse_input_embeds

            # Walk ``multimodal_params`` in left-to-right request order
            # and produce ``mm_embeds`` per-item so the concatenated rows
            # match ``mm_token_indices`` in prompt order — required for
            # mixed image+video prompts.
            items = extract_multimodal_items_in_request_order(mm_params_with_data)
            embed_tokens = self.model.embed_tokens
            flat_ids = input_ids.flatten()
            mm_embeds: List[torch.Tensor] = []
            vision_dtype = getattr(self.vision_tower, "dtype", embed_tokens.weight.dtype)
            vision_device = embed_tokens.weight.device
            for item in items:
                feats = self.vision_tower(
                    pixel_values=item["pixel_values"].to(device=vision_device, dtype=vision_dtype),
                    grid_thw=item["grid_thw"],
                )
                mm_embeds.append(feats.to(dtype=embed_tokens.weight.dtype, device=vision_device))

            # ``self.mm_token_ids`` covers both image and video
            # placeholders and is registered as a buffer in __init__ so
            # the production runtime can read it. Move to the input's
            # device for the fuse path (registry buffer device follows
            # ``model.to(device)`` automatically, but the standalone
            # unit-test path may use a different device).
            fused_ids, fused_embeds = fuse_input_embeds(
                embedding_layer=embed_tokens,
                input_ids=flat_ids,
                mm_embeds=mm_embeds,
                mm_token_ids=self.mm_token_ids.to(flat_ids.device),
                **kwargs,
            )
            input_ids = fused_ids
            inputs_embeds = fused_embeds

        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            **kwargs,
        )


# Register the M3 VL input processor for the production LLM API.
MiniMaxM3VLForConditionalGeneration = _build_minimax_m3_vl_input_processor_registration()(
    MiniMaxM3VLForConditionalGeneration
)
