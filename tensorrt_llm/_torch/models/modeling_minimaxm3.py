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
import os
from typing import Any, Dict, List, Optional, Tuple
from typing import Mapping as TMapping

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig

from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    AttentionForwardArgs,
    PositionalEmbeddingParams,
    RopeParams,
)
from ..attention_backend.sparse.minimax_m3 import (
    MiniMaxM3MsaSparseAttention,
    MiniMaxM3SparseRuntimeBackend,
    _gather_paged_batched,
    _write_main_kv_slots_to_pool,
)
from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams, MiniMaxAllReduceRMS
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM3MoeRoutingMethod, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
    copy_weight,
    load_weight_shard,
)
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import (
    ActivationType,
    AuxStreamType,
    EventType,
    get_model_extra_attrs,
    is_torch_compiling,
)
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.minimaxm3_weight_mapper import MINIMAX_M3_PARAMS_MAP, MiniMaxM3HfWeightMapper
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, ModelConfig, filter_weights, register_auto_model

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
    # SwiGLU-OAI is plain SwiGLU with an alpha gain and (up + 1) offset, so it
    # routes through the fused silu_and_mul kernel (one launch, optional fp8
    # epilogue) instead of the eager elementwise fallback. Mirrors the
    # routed-expert SwigluBias path. See _minimax_m3_swiglu_oai for the math.
    return GatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        activation=torch.nn.functional.silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=1.0,
        swiglu_limit=swiglu_limit,
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
        # Router matmul runs against the full-precision fp32 weight to match
        # SGLang. The fused GEMM widens the bf16/fp16 activation to fp32 on-chip
        # with fp32 accumulation, so no separate fp32 copy of the activation is
        # materialized. The eager cast plus linear covers fp32 activations and
        # non-CUDA tensors.
        if hidden_states.is_cuda and hidden_states.dtype in (torch.bfloat16, torch.float16):
            return torch.ops.trtllm.moe_router_gemm_op(
                hidden_states, self.weight, out_dtype=torch.float32
            )
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
    attn_layer._dispatch_attention_backend(
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
    additionally carry the MiniMax index branch: a fused index_qk_proj (output
    [idx_q | idx_k]) plus per-head index norms. The index value/output branch
    is omitted because the M3 checkpoint sets sparse_disable_index_value=True on
    every sparse layer; the gate catches the unmapped keys if that ever flips.
    """

    def __init__(
        self,
        *,
        model_config: "ModelConfig[PretrainedConfig]",
        layer_idx: Optional[int] = None,
        is_sparse_attention_layer: bool = False,
        disable_index_value: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
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

        # Dtype of the q/k/v activations fed into norm+RoPE. This is always the
        # model compute dtype (bf16) for every M3 config. KV-cache quantization
        # (fp8/fp4) only changes cache storage, not the activation dtype.
        self.attn_activation_dtype = config.torch_dtype

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

        # Aux stream and events to run the per-head q-norm and k-norm
        # concurrently on the separate-norm fallback path. The stream is shared
        # across layers via aux_stream_dict, matching DeepSeekV3; it falls back
        # to a private stream when constructed standalone (for example in tests).
        # The bf16 fast path fuses norm and RoPE into one kernel and does not
        # use these.
        self.aux_stream = aux_stream if aux_stream is not None else torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

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

            # Index Q and K are both replicated (no head sharding) and project
            # the same hidden_states, so fuse them into one index_qk_proj GEMM
            # with output [idx_q | idx_k]. idx_q holds all num_index_heads heads;
            # idx_k is a single K per token, broadcast across heads when scoring.
            self.index_q_size = self.sparse_num_index_heads * self.sparse_index_dim
            self.index_k_size = self.sparse_index_dim
            self.index_qk_proj = Linear(
                config.hidden_size,
                self.index_q_size + self.index_k_size,
                bias=False,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=None,
                quant_config=None,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
                ),
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

        def _q_norm():
            return self.q_norm(q.reshape(-1, self.head_dim_value)).reshape(q_shape)

        def _k_norm():
            return self.k_norm(k.reshape(-1, self.head_dim_value)).reshape(k_shape)

        # Run the independent q-norm and k-norm concurrently on the aux stream.
        # Falls back to sequential execution under torch.compile.
        q, k = maybe_execute_in_parallel(
            _q_norm,
            _k_norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
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

        def _idx_q_norm():
            return self.index_q_norm(idx_q.reshape(-1, self.sparse_index_dim)).reshape(idx_q_shape)

        def _idx_k_norm():
            return self.index_k_norm(idx_k.reshape(-1, self.sparse_index_dim)).reshape(idx_k_shape)

        # Run the independent index q-norm and k-norm concurrently on the aux
        # stream. Falls back to sequential execution under torch.compile.
        idx_q, idx_k = maybe_execute_in_parallel(
            _idx_q_norm,
            _idx_k_norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
        return idx_q, idx_k

    def _fused_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        *,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
    ) -> Optional[torch.Tensor]:
        """Fuse per-head Gemma RMSNorm and partial RoPE into one kernel.

        Runs torch.ops.trtllm.fused_qk_norm_rope in place over the fused qkv
        (Q heads then K heads, optional V heads left untouched) and returns the
        mutated tensor. The kernel norms the full head_dim and rotates only the
        first rotary_dim channels read from RopeParams.dim, which matches M3's
        whole-head norm with front partial RoPE, and applies the same Gemma
        (1 + weight) scaling as apply_qk_norm.

        Returns None, leaving qkv untouched, when the fused path does not apply
        so callers fall back to separate norm and RoPE. This happens when
        activations are not bf16 (the kernel is bf16-only), when RoPE has no
        position_ids, or when no partial-RoPE rotary_emb exists.
        """
        if position_ids is None or qkv.dtype != torch.bfloat16:
            return None
        if (
            self.rotary_emb is None
            or self.pos_embd_params is None
            or self.pos_embd_params.rope is None
        ):
            return None

        # Partial-RoPE dim comes from RopeParams (M3 rotates 64 of 128).
        rotary_dim = int(self.pos_embd_params.rope.dim)
        # The kernel assumes a contiguous [num_tokens, total_heads * head_dim].
        qkv = qkv.contiguous()
        torch.ops.trtllm.fused_qk_norm_rope(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            q_norm.variance_epsilon,
            q_norm.weight,
            k_norm.weight,
            self.pos_embd_params.rope.theta,
            self.pos_embd_params.is_neox,
            position_ids.reshape(-1).contiguous().to(torch.int32),
            1.0,  # factor: no YARN (M3 has no rope_scaling)
            0.0,  # low
            0.0,  # high
            1.0,  # attention_factor
            True,  # is_qk_norm
            self.use_gemma_norm,  # use_gemma
            False,  # use_mrope
            0,  # mrope_section1
            0,  # mrope_section2
        )
        return qkv

    def _expect_fused_qk_norm_rope(self, position_ids: Optional[torch.Tensor]) -> bool:
        """Whether the fused path must run rather than fall back.

        The fused kernel runs when attention activations are bf16 and RoPE has
        position_ids. Every M3 config keeps bf16 attention activations, so a
        fallback is a regression. Guards the forward asserts.
        """
        return self.attn_activation_dtype == torch.bfloat16 and position_ids is not None

    def forward(
        self,
        position_ids: Optional[torch.IntTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
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
            return self._dense_forward(
                position_ids,
                hidden_states,
                attn_metadata,
                all_reduce_params=all_reduce_params,
                **kwargs,
            )
        return self._sparse_forward(
            position_ids,
            hidden_states,
            attn_metadata,
            all_reduce_params=all_reduce_params,
            **kwargs,
        )

    def _dense_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
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
          2-3. Apply per-head Gemma RMSNorm and partial RoPE to Q/K. The bf16
             fast path fuses both into one fused_qk_norm_rope kernel over the
             fused qkv; the non-bf16-activation fallback runs apply_qk_norm then
             rotary_emb separately.
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

        # Projections (no index branch).
        qkv = self.qkv_proj(hidden_states)

        # Per-head Gemma RMSNorm and partial RoPE on Q/K. The bf16 fast
        # path fuses both into one kernel over the fused qkv; otherwise fall
        # back to separate norm and RoPE.
        fused_qkv = self._fused_qk_norm_rope(
            qkv,
            position_ids,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_key_value_heads,
            num_heads_v=self.num_key_value_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        if fused_qkv is not None:
            q, k, v = fused_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Match the fallback contiguity; V stays a column-slice view.
            q, k = q.contiguous(), k.contiguous()
        else:
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 dense attention (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, head_dim="
                f"{self.head_dim}) but fell back to the separate path; qkv dtype "
                f"is {qkv.dtype} (expected {self.attn_activation_dtype})."
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])

        # Keep token-wise projections and the output projection visible to
        # torch.compile. Only the metadata/cache-dependent attention core is
        # hidden behind the inplace custom op.
        o = self._forward_attention_core(q, k, v, None, None, attn_metadata)
        # all_reduce_params lets the decoder defer the o_proj output AllReduce so
        # it can be fused with post_attention_layernorm (RESIDUAL_RMS_NORM).
        # Passing None preserves the standalone o_proj reduction used by the
        # single-GPU, attention-DP, and fusion-disabled paths.
        return self.o_proj(o, all_reduce_params=all_reduce_params)

    def _sdpa_dense_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense cache updates and attention into ``output`` (SDPA path).

        This is the non-MSA dense reference path: it runs standard GQA via
        ``torch.nn.functional.scaled_dot_product_attention`` (memory-efficient
        / math backends). It is selected only when ``self.attn`` is **not** a
        :class:`MiniMaxM3MsaSparseAttention`; the MSA backend handles dense
        layers itself inside :meth:`_msa_attention_core`.
        """
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
        # Graph capture bakes the gather/mask width, and max_seqlen_k is a
        # per-step host bound later replays can outgrow. Bake
        # min(page-table width, engine max_seq_len): the raw table width
        # alone is inflated far past max_seq_len by the KV-estimation pass
        # and would OOM the [batch, max_k, heads] gather. The seq_lens mask
        # below invalidates the slack.
        if attn_metadata.is_cuda_graph:
            capacity = int(m3_meta.req_to_token.shape[1])
            max_k = min(capacity, int(attn_metadata.max_seq_len or capacity))
        else:
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

        # Build the per-query attention mask that masks out padded KV
        # positions beyond each sequence's true ``seq_lens`` and (for
        # prefill) preserves causality. ``q_positions`` from the prefill
        # metadata names each Q token's K-side position.
        # The metadata tensors are produced by
        # :meth:`MiniMaxM3AttentionMetadata.prepare` on the cache
        # device, so ``.to(dtype=torch.long)`` is a same-device dtype
        # conversion (capture-safe).
        seq_lens_dev = m3_meta.seq_lens.to(dtype=torch.long)
        kv_positions = torch.arange(max_k, device=q.device).unsqueeze(0)  # [1, max_k]

        if m3_meta.is_prefill:
            if group > 1:
                k_padded = k_padded.repeat_interleave(group, dim=2)
                v_padded = v_padded.repeat_interleave(group, dim=2)
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
            # Decode: qo_len query tokens per request; token t of request b
            # attends seq_lens[b] - qo_len + t + 1 positions (the causal
            # ladder; qo_len=1 is the classic one-token mask). Every input
            # tensor here is already on q.device (set up by prepare()), so
            # SDPA captures cleanly.
            qo_len = int(m3_meta.decode_qo_len)
            ladder = torch.arange(1 - qo_len, 1, device=q.device, dtype=torch.long)
            # eff[b, t] = attendable position count for token t of row b.
            eff = seq_lens_dev.unsqueeze(-1) + ladder  # [batch, qo]
            valid = kv_positions.unsqueeze(1) < eff.unsqueeze(-1)  # [batch, qo, max_k]
            q_b = q_view.view(batch, qo_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # [batch, H, qo, d]
            mask_b = valid.unsqueeze(1)  # [batch, 1, qo, k]
            # Expand K/V one KV head at a time: the all-heads transient is
            # O(batch * max_k * num_heads) inside the CUDA-graph pool and
            # exceeds the pool budget under attention DP (unsharded heads);
            # with TP-sharded KV heads this is one iteration.
            out_b = q.new_empty(batch, self.num_heads, qo_len, self.head_dim)
            with sdpa_kernel(_DENSE_SDPA_BACKENDS):
                for h in range(max(self.num_key_value_heads, 1)):
                    qh = slice(h * group, (h + 1) * group)
                    k_h = k_padded[:, :, h : h + 1].repeat_interleave(group, dim=2)
                    v_h = v_padded[:, :, h : h + 1].repeat_interleave(group, dim=2)
                    out_b[:, qh] = torch.nn.functional.scaled_dot_product_attention(
                        q_b[:, qh].to(q.dtype),
                        k_h.transpose(1, 2).to(q.dtype),
                        v_h.transpose(1, 2).to(q.dtype),
                        attn_mask=mask_b,
                        dropout_p=0.0,
                        is_causal=False,
                    )  # [batch, group, qo, d]
            # Copy through a token-major [batch, qo, H, dh] view rather
            # than transpose(1, 2).reshape, which (with H != head_dim)
            # copies in C-order and scrambles (head, head_dim) into o_proj.
            output.view(batch, qo_len, self.num_heads, self.head_dim).copy_(out_b.transpose(1, 2))

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
            self._dispatch_attention_backend(q, k, v, idx_q, idx_k, attn_metadata, output)
        return output

    def _dispatch_attention_backend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Route the attention core to the configured backend.

        ``self.attn`` is either a :class:`MiniMaxM3MsaSparseAttention` (MSA
        backend, which handles both dense and sparse layers) or a
        :class:`MiniMaxM3SparseRuntimeBackend` (Triton backend). This method
        only selects the backend-specific core:

        * MSA → :meth:`_msa_attention_core`
        * Triton sparse → :meth:`_triton_sparse_attention_core`
        * SDPA dense → :meth:`_sdpa_dense_attention_core`
        """
        if isinstance(self.attn, MiniMaxM3MsaSparseAttention):
            return self._msa_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        if self.is_sparse_attention_layer:
            assert idx_q is not None and idx_k is not None
            return self._triton_sparse_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        assert idx_q is None and idx_k is None
        return self._sdpa_dense_attention_core(q, k, v, attn_metadata, output)

    def _msa_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MSA backend (:class:`MiniMaxM3MsaSparseAttention`).

        The MSA backend runs the sparse GQA or dense paged GQA through its
        inherited FMHA forward; this layer selects the top-k blocks (sparse
        only) and builds the ``forward_args`` the FMHA reads.
        """
        if self.is_sparse_attention_layer:
            assert idx_q is not None and idx_k is not None
            # Publish the selected blocks so the FMHA runs the sparse path.
            kv_block_indexes = self.attn.run_indexer(idx_q, idx_k, attn_metadata)
            forward_args = AttentionForwardArgs(output=output, topk_indices=kv_block_indexes)
        else:
            assert idx_q is None and idx_k is None
            # No top-k selection means the FMHA attends the full page table.
            forward_args = AttentionForwardArgs(output=output)
        self.attn.forward(q, k, v, attn_metadata, forward_args=forward_args)
        return output

    def _sparse_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run a MiniMax-M3 sparse attention forward end-to-end.

        Steps:
          1. Project ``hidden_states`` to Q/K/V (fused ``qkv_proj``)
             plus index Q (per-head) and index K (single replicated).
          2-3. Apply per-head Gemma RMSNorm and partial RoPE to the main and
             index branches. The bf16 fast path fuses each branch into one
             fused_qk_norm_rope kernel (the index branch passes num_heads_v=0
             and norms only the concatenated idx_q/idx_k); the non-bf16-activation
             fallback runs the norm helpers then rotary_emb.
          4. Pull paged main K/V cache (reshaped to flat-slot view) and
             paged side index-K cache from the
             :class:`MiniMaxM3KVCacheManagerV2`.
          5. Build a :class:`MiniMaxM3TritonSparseAttentionMetadata` from the
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
        :class:`MiniMaxM3TritonSparseAttentionMetadata` and an
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

        # Project, per-head Gemma RMSNorm, and partial RoPE for the main and
        # index branches. Each branch fuses norm and RoPE into one bf16 kernel,
        # falling back to separate norm and RoPE otherwise. The branches read
        # only hidden_states and write disjoint outputs, so each runs its
        # projection, norm, and RoPE concurrently on the aux stream when
        # multi-stream is enabled, joining before the attention core.
        def _main_norm_rope():
            qkv = self.qkv_proj(hidden_states)
            fused_qkv = self._fused_qk_norm_rope(
                qkv,
                position_ids,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_key_value_heads,
                num_heads_v=self.num_key_value_heads,
                head_dim=self.head_dim,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
            )
            if fused_qkv is not None:
                q, k, v = fused_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                return q.contiguous(), k.contiguous(), v
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse attention (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, head_dim="
                f"{self.head_dim}) but fell back to the separate path; qkv dtype "
                f"is {qkv.dtype} (expected {self.attn_activation_dtype})."
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])
            return q, k, v

        def _index_norm_rope():
            idx_qk = self.index_qk_proj(hidden_states)
            fused_idx = self._fused_qk_norm_rope(
                idx_qk,
                position_ids,
                num_heads_q=self.sparse_num_index_heads,
                num_heads_k=1,
                num_heads_v=0,
                head_dim=self.sparse_index_dim,
                q_norm=self.index_q_norm,
                k_norm=self.index_k_norm,
            )
            if fused_idx is not None:
                idx_q, idx_k = fused_idx.split(
                    [self.sparse_num_index_heads * self.sparse_index_dim, self.sparse_index_dim],
                    dim=-1,
                )
                return idx_q.contiguous(), idx_k.contiguous()
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse index branch (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, index_dim="
                f"{self.sparse_index_dim}) but fell back to the separate path; idx "
                f"dtype is {idx_qk.dtype} (expected {self.attn_activation_dtype})."
            )
            idx_q, idx_k = self.apply_index_qk_norm(idx_q, idx_k)
            if self.rotary_emb is not None and position_ids is not None:
                idx_q, idx_k = self.rotary_emb(position_ids, [idx_q, idx_k])
            return idx_q, idx_k

        (q, k, v), (idx_q, idx_k) = maybe_execute_in_parallel(
            _main_norm_rope,
            _index_norm_rope,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )

        o = self._forward_attention_core(q, k, v, idx_q, idx_k, attn_metadata)
        # all_reduce_params lets the decoder defer the o_proj output AllReduce so
        # it can be fused with post_attention_layernorm (RESIDUAL_RMS_NORM).
        # Passing None preserves the standalone o_proj reduction used by the
        # single-GPU, attention-DP, and fusion-disabled paths.
        return self.o_proj(o, all_reduce_params=all_reduce_params)

    def _triton_sparse_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse cache updates and attention into ``output`` (Triton path).

        This is the non-MSA sparse path: it dispatches to
        :class:`MiniMaxM3SparseRuntimeBackend` (the Triton sparse runtime). It
        is selected only when ``self.attn`` is that backend; the MSA backend
        handles sparse layers itself inside :meth:`_msa_attention_core`.
        """
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
        if not isinstance(self.attn, MiniMaxM3SparseRuntimeBackend):
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
        self.enable_attention_dp = self.mapping.enable_attention_dp

        _, sparse_layer_ids = get_sparse_layer_ids(config)
        disable_index_value_ids = set(get_sparse_disable_index_value_layer_ids(config))
        is_sparse = layer_idx in sparse_layer_ids
        disable_index_value = layer_idx in disable_index_value_ids

        self.self_attn = MiniMaxM3Attention(
            model_config=model_config,
            layer_idx=layer_idx,
            is_sparse_attention_layer=is_sparse,
            disable_index_value=disable_index_value,
            aux_stream=aux_stream_dict[AuxStreamType.Attention],
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

        # Layer-boundary RMSNorms are plain (non-Gemma) norms so they can drive
        # the fused AllReduce+residual+RMSNorm epilogue
        # (AllReduceFusionOp.RESIDUAL_RMS_NORM), whose kernel applies a plain
        # weight * x scaling with no Gemma (1 + weight) offset. When the
        # checkpoint stores Gemma norms (use_gemma_norm=True), the loader folds
        # (1 + weight) into the stored weight at load time (see
        # _fold_gemma_boundary_norm_weights), so the runtime norm is numerically
        # identical to the original Gemma norm on every path. The per-head
        # q/k/index norms keep use_gemma because they are consumed by the
        # separate fused_qk_norm_rope kernel, which handles Gemma directly.
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )

        # DeepSeek-V3-style layer-boundary AllReduce fusion. Each layer folds
        # the attention o_proj output AllReduce into post_attention_layernorm
        # (PRE fusion) and the MoE/MLP output AllReduce into the next layer's
        # input_layernorm (POST fusion, wired via next_layer_layernorm in
        # setup_aliases). Fusion is only meaningful when there is a real
        # cross-rank reduction to fold, i.e. TP>1 and not attention-DP (each DP
        # rank owns independent tokens, so no attention/MoE AllReduce happens
        # there). An env override matches the DeepSeek-V3 escape hatch.
        self.enable_fusion = os.environ.get("TRTLLM_MINIMAX_M3_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= (not self.enable_attention_dp) and self.mapping.tp_size > 1
        self.pre_feed_forward_fusion = self.enable_fusion
        self.post_feed_forward_fusion = self.enable_fusion

        self.allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=config.torch_dtype,
            )

        # Wired by MiniMaxM3ForCausalLM.setup_aliases after weight load to the
        # next layer's input_layernorm (or the final model norm for the last
        # layer). None disables POST fusion and boundary-norm folding.
        self.next_layer_layernorm: Optional[RMSNorm] = None

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NVTX markers below are emitted only when TLLM_NVTX_DEBUG=1 (or
        # TLLM_LLMAPI_ENABLE_NVTX=1) is set; otherwise they are no-ops.
        attn_kind = "sparse_attn" if self.self_attn.is_sparse_attention_layer else "dense_attn"

        # Layer-0 prologue only. For every subsequent layer the input_layernorm
        # (an add+RMSNorm at the layer boundary) was already applied by the
        # previous layer as its next_layer_layernorm, so residual is not None
        # here and this block is skipped (matches DeepSeek-V3).
        if residual is None:
            with nvtx_range_debug(f"layer{self.layer_idx}.input_layernorm"):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

        # When PRE fusion is active the attention defers its o_proj AllReduce so
        # it can be fused into post_attention_layernorm below; otherwise the
        # o_proj reduces as usual (all_reduce_params=None preserves the
        # single-GPU, attention-DP, and fusion-disabled behavior exactly).
        attn_all_reduce_params = (
            AllReduceParams(enable_allreduce=False) if self.pre_feed_forward_fusion else None
        )
        with nvtx_range_debug(f"layer{self.layer_idx}.{attn_kind}"):
            hidden_states = self.self_attn(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                all_reduce_params=attn_all_reduce_params,
                **kwargs,
            )

        if self.block_sparse_moe is not None:
            hidden_states, residual = self.forward_MoE(hidden_states, attn_metadata, residual)
        else:
            hidden_states, residual = self.forward_mlp(hidden_states, residual)

        # hidden_states is fully TP-reduced at layer exit (no cross-layer
        # allreduce+norm fusion).
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual

    def _apply_pre_feed_forward_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AllReduce(+residual+RMSNorm) between attention and the feed-forward.

        On the PRE-fusion path the deferred attention o_proj AllReduce is fused
        with post_attention_layernorm into one kernel; otherwise it is the plain
        add+RMSNorm and the attention already reduced its own output.
        """
        if self.pre_feed_forward_fusion:
            return self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        return self.post_attention_layernorm(hidden_states, residual)

    def _apply_next_layer_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the next layer's input_layernorm at the layer boundary.

        On the POST-fusion path the deferred feed-forward output AllReduce is
        fused with next_layer_layernorm (the next layer's input_layernorm, or
        the final model norm for the last layer). Off the fusion path it is the
        plain add+RMSNorm. When next_layer_layernorm has not been wired (e.g. a
        standalone unit test that never ran setup_aliases) the
        (hidden_states, residual) pair is returned unchanged so the model can
        apply the final norm itself.
        """
        if self.next_layer_layernorm is None:
            return hidden_states, residual
        if self.post_feed_forward_fusion:
            return self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        return self.next_layer_layernorm(hidden_states, residual)

    def _feed_forward_all_reduce_params(self) -> Optional[AllReduceParams]:
        """AllReduce params handed to the MoE or dense-MLP output projection.

        Disables the module's internal output AllReduce when POST fusion will
        fold it into next_layer_layernorm; otherwise None preserves the module's
        own reduction (single-GPU, attention-DP, fusion-disabled).
        """
        if self.post_feed_forward_fusion:
            return AllReduceParams(enable_allreduce=False)
        return None

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with nvtx_range_debug(f"layer{self.layer_idx}.post_attention_layernorm"):
            hidden_states, residual = self._apply_pre_feed_forward_norm(hidden_states, residual)

        with nvtx_range_debug(f"layer{self.layer_idx}.moe"):
            hidden_states = self.block_sparse_moe(
                hidden_states,
                attn_metadata,
                final_all_reduce_params=self._feed_forward_all_reduce_params(),
            )

        with nvtx_range_debug(f"layer{self.layer_idx}.next_layer_layernorm"):
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with nvtx_range_debug(f"layer{self.layer_idx}.post_attention_layernorm"):
            hidden_states, residual = self._apply_pre_feed_forward_norm(hidden_states, residual)

        with nvtx_range_debug(f"layer{self.layer_idx}.mlp"):
            hidden_states = self.mlp(
                hidden_states,
                final_all_reduce_params=self._feed_forward_all_reduce_params(),
            )

        with nvtx_range_debug(f"layer{self.layer_idx}.next_layer_layernorm"):
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)
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
        # Aux streams shared across layers, matching the DeepSeekV3 convention:
        # one for the per-head q/k norm overlap in attention, one for MoE
        # shared/routed parallel execution, and one for MoE chunking overlap
        # inside the fused MoE kernel.
        self.aux_stream_dict = {
            AuxStreamType.Attention: torch.cuda.Stream(),
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
        # Final norm is a plain (non-Gemma) RMSNorm for the same reason as the
        # layer-boundary norms (see MiniMaxM3DecoderLayer.__init__): it doubles
        # as the last layer's next_layer_layernorm, so the last MoE/MLP output
        # AllReduce folds into it via RESIDUAL_RMS_NORM. The Gemma (1 + weight)
        # offset is folded into the stored weight at load time (see
        # _fold_gemma_boundary_norm_weights).
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        residual = None
        for layer_idx, decoder_layer in enumerate(self.layers):
            # Per-layer NVTX range (layer0, layer1, ...). Emitted only when
            # TLLM_NVTX_DEBUG=1 (or TLLM_LLMAPI_ENABLE_NVTX=1) is set.
            with nvtx_range_debug(f"MiniMaxM3.layer{layer_idx}"):
                hidden_states, residual = decoder_layer(
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    residual=residual,
                    spec_metadata=spec_metadata,
                )

        # When setup_aliases has chained the final norm into the last decoder
        # layer (next_layer_layernorm = self.norm), the last layer's boundary
        # step already applied it (fused or plain), so hidden_states is normed
        # and this is skipped. The fallback covers paths that never ran
        # setup_aliases (e.g. standalone unit tests): there the last layer
        # returns the unnormed (hidden_states, residual) pair and the final
        # add+RMSNorm is applied here.
        if self.layers[-1].next_layer_layernorm is None:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _load_index_qk_proj_weights(model: nn.Module, weights) -> None:
    """Fuse checkpoint index_q_proj and index_k_proj into index_qk_proj.

    The shared weight loader only auto-fuses qkv_proj and gate_up_proj by
    name, so walk each fused index module and load the sibling checkpoint
    tensors through the FUSED_GATE_UP_LINEAR row-cat path. Sources are
    marked consumed so the generic pass does not treat them as unused.
    """
    for name, module in model.named_modules():
        if name.split(".")[-1] != "index_qk_proj":
            continue
        parent = name.rsplit(".", 1)[0]
        q_weights = filter_weights(f"{parent}.index_q_proj", weights)
        k_weights = filter_weights(f"{parent}.index_k_proj", weights)
        # Missing sources make Linear.load_weights assert rather than leave
        # the fused module silently uninitialized.
        module.load_weights(weights=[q_weights, k_weights])
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed(f"{parent}.index_q_proj")
            weights.mark_consumed(f"{parent}.index_k_proj")
        else:
            for key in list(weights.keys()):
                if key.startswith(f"{parent}.index_q_proj.") or key.startswith(
                    f"{parent}.index_k_proj."
                ):
                    del weights[key]


# Layer-boundary RMSNorms whose Gemma (1 + weight) scaling is folded into the
# stored weight at load time so the runtime norm is a plain RMSNorm (see
# MiniMaxM3DecoderLayer.__init__ / MiniMaxM3Model.__init__). These are exactly
# the norms that drive the DeepSeek-V3-style fused AllReduce+residual+RMSNorm
# epilogue, whose kernel has no Gemma offset. The per-head q/k/index norms are
# intentionally excluded: they feed the separate fused_qk_norm_rope kernel,
# which handles Gemma directly and stays use_gemma=True.
_M3_BOUNDARY_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
)
_M3_FINAL_NORM_KEY = "model.norm.weight"


def _fold_gemma_boundary_norm_weights(weights):
    """Fold Gemma (1 + weight) into the layer-boundary RMSNorm weights.

    MiniMax-M3 stores every RMSNorm as a Gemma norm (use_gemma_norm=True), which
    computes (1 + weight) * x. The layer-boundary norms are constructed as plain
    norms (use_gemma=False, weight * x) so they can drive the fused
    AllReduce+RMSNorm kernels, so their stored weights must be pre-incremented by
    1.0. This is a numerically exact, load-time-only rewrite; the resulting norm
    is identical to the original Gemma norm on every path.

    Only the decoder input_layernorm / post_attention_layernorm and the final
    model.norm are touched. A no-op for keys that are absent (partial load) so it
    is safe to call unconditionally, but it must only run when the checkpoint
    actually uses Gemma norms (guarded by the caller).
    """
    for key in list(weights.keys()):
        if key.endswith(_M3_BOUNDARY_NORM_SUFFIXES) or key == _M3_FINAL_NORM_KEY:
            w = weights[key]
            w = w[:] if hasattr(w, "__getitem__") else w
            weights[key] = w + 1.0
    return weights


@register_auto_model("MiniMaxM3SparseForCausalLM")
class MiniMaxM3ForCausalLM(SpecDecOneEngineForCausalLM[MiniMaxM3Model, PretrainedConfig]):
    """Text-only M3 model."""

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        raw_pretrained = model_config.pretrained_config
        if is_minimax_m3_vl_config(raw_pretrained):
            model_config = get_text_model_config(model_config)
        super().__init__(MiniMaxM3Model(model_config), model_config)

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
        # Fuse index_q/index_k into each index_qk_proj module (also covers
        # the VL path, which routes text weights through here).
        _load_index_qk_proj_weights(self, weights)
        # Fold Gemma (1 + weight) into the layer-boundary RMSNorm weights so the
        # runtime norms can be plain (non-Gemma) and drive the fused
        # AllReduce+residual+RMSNorm epilogue. Only when the checkpoint actually
        # stores Gemma norms; otherwise the boundary norms are already plain.
        if bool(getattr(self.config, "use_gemma_norm", False)):
            weights = _fold_gemma_boundary_norm_weights(weights)
        if weight_mapper is None:
            weight_mapper = MiniMaxM3HfWeightMapper()
        weight_mapper.init_model_and_config(self, self.model_config)
        merged_params_map = {**MINIMAX_M3_PARAMS_MAP, **(params_map or {})}
        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map=merged_params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def setup_aliases(self) -> None:
        """Chain each decoder layer's next_layer_layernorm for POST fusion.

        Wired after weight load (the generic loader skips next_layer_layernorm
        aliases). Each layer's MoE/MLP output AllReduce is fused into the next
        layer's input_layernorm; the last layer chains the final model norm so
        its output AllReduce folds the final normalization too.
        """
        layers = self.model.layers
        num_layers = len(layers)
        for idx, layer in enumerate(layers):
            if idx == num_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = layers[idx + 1].input_layernorm


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

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
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

        super().load_weights(
            weights=text_weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

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
