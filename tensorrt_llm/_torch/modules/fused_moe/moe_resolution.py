# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve the MoE implementation from backend preference and capabilities.

Records rejected candidates; falls back when the requested backend cannot serve.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Tuple, Type, Union

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...model_config import ModelConfig
from ...peft.lora.validation import has_moe_lora_targets
from ...utils import ActivationType
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepGemmFusedMoE
from .fused_moe_densegemm import DenseGEMMFusedMoE
from .fused_moe_marlin import MarlinFusedMoE
from .fused_moe_triton import TritonFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejection,
    MoERejectReason,
    MoEResolutionReport,
    canonical_activation,
    canonical_quant,
    canonical_routing,
)
from .impl_environment import collect_moe_environment
from .mega_moe import MegaMoECuteDsl, MegaMoEDeepGemm
from .moe_load_balancer import get_moe_load_balancer

if TYPE_CHECKING:
    from .routing import BaseMoeRoutingMethod, RoutingMethodType

WIDEEP_DEPRECATION_MESSAGE = (
    "The WIDEEP MoE backend is deprecated and can no longer be selected. Wide "
    "expert parallelism and EPLB are supported by the other backends: use "
    "DEEPGEMM for FP8 block-scale checkpoints, or TRTLLM / CUTEDSL / CUTLASS "
    "otherwise."
)

# Global priority: specialized first, broad fallbacks last.
IMPL_PRIORITY: Tuple[Type, ...] = (
    CuteDslB12xFusedMoE,  # SM120/121 NVFP4 decode only -- narrowest, so first
    MegaMoEDeepGemm,  # ahead of plain CuteDSL / DeepGEMM: better perf when eligible
    MegaMoECuteDsl,
    CuteDslFusedMoE,
    TRTLLMGenFusedMoE,
    DeepGemmFusedMoE,
    DenseGEMMFusedMoE,
    MarlinFusedMoE,
    TritonFusedMoE,
    CutlassFusedMoE,  # widest coverage, hence the fallback
    VanillaMoE,  # reference implementation, never preferred
)

# Family membership only; IMPL_PRIORITY decides try order.
BACKEND_FAMILY: Dict[str, FrozenSet[Type]] = {
    "CUTLASS": frozenset({CutlassFusedMoE}),
    "VANILLA": frozenset({VanillaMoE}),
    "MARLIN": frozenset({MarlinFusedMoE}),
    "CUTEDSL": frozenset({CuteDslB12xFusedMoE, CuteDslFusedMoE}),
    "DEEPGEMM": frozenset({DeepGemmFusedMoE}),
    "DENSEGEMM": frozenset({DenseGEMMFusedMoE}),
    "TRTLLM": frozenset({TRTLLMGenFusedMoE}),
    "TRITON": frozenset({TritonFusedMoE}),
    "MEGAMOE_DEEPGEMM": frozenset({MegaMoEDeepGemm}),
    "MEGAMOE_CUTEDSL": frozenset({MegaMoECuteDsl}),
}

# Catch table drift at import time.
_UNRANKED = {cls for family in BACKEND_FAMILY.values() for cls in family} - set(IMPL_PRIORITY)
if _UNRANKED:
    raise RuntimeError(
        f"MoE impls named by BACKEND_FAMILY but absent from IMPL_PRIORITY: "
        f"{sorted(cls.__name__ for cls in _UNRANKED)}"
    )

# Widest coverage; default degradation target.
FALLBACK_IMPL: Type = CutlassFusedMoE


def _legacy_backend_name(impl_cls: Type) -> str:
    """Diagnostic name used until each leaf class owns one fixed impl id."""
    return impl_cls.__name__


# ---------------------------------------------------------------------------
# Building the question
# ---------------------------------------------------------------------------


# HF configs use different names for the same fields; ModelConfig does not unify them.
_NUM_EXPERTS_ATTRS = ("num_experts", "n_routed_experts", "num_local_experts")
_TOP_K_ATTRS = ("num_experts_per_tok", "experts_per_token")


@dataclass(frozen=True)
class MoELayerShapes:
    """Resolved shapes for MoE construction / selection."""

    num_experts: Optional[int]
    hidden_size: Optional[int]
    intermediate_size: Optional[int]
    dtype: Optional[torch.dtype]
    top_k: Optional[int]


def derive_moe_layer_shapes(
    model_config: ModelConfig,
    *,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    top_k: Optional[int] = None,
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"] = None,
) -> MoELayerShapes:
    """Fill unset fields from ``pretrained_config`` (and routing for ``top_k``).

    Explicit args win. ``top_k`` order: explicit, then routing object's
    ``experts_per_token``, then pretrained. A bare ``RoutingMethodType`` has
    no k, so it falls through.
    """
    from .routing import BaseMoeRoutingMethod

    pretrained = model_config.pretrained_config

    if dtype is None and pretrained is not None:
        dtype = getattr(pretrained, "torch_dtype", None)
    if hidden_size is None and pretrained is not None:
        hidden_size = getattr(pretrained, "hidden_size", None)
    if intermediate_size is None and pretrained is not None:
        # Prefer MoE width; getattr so a present-but-None field still falls through.
        intermediate_size = getattr(pretrained, "moe_intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(pretrained, "intermediate_size", None)
    if num_experts is None and pretrained is not None:
        for attr in _NUM_EXPERTS_ATTRS:
            value = getattr(pretrained, attr, None)
            if value is not None:
                num_experts = value
                break

    if top_k is None and isinstance(routing, BaseMoeRoutingMethod):
        top_k = routing.experts_per_token
    if top_k is None and pretrained is not None:
        for attr in _TOP_K_ATTRS:
            value = getattr(pretrained, attr, None)
            if value is not None:
                top_k = value
                break

    return MoELayerShapes(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        top_k=top_k,
    )


def build_moe_problem(
    model_config: ModelConfig,
    *,
    override_quant_config: Optional[QuantConfig] = None,
    dtype: Optional[torch.dtype] = None,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    top_k: Optional[int] = None,
    swiglu_gptoss_style: Optional[bool] = None,
    bias: Optional[bool] = None,
    activation_type: Optional[ActivationType] = None,
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"] = None,
) -> MoEProblem:
    """Assemble the problem half of a selection question.

    Explicit args win over ``pretrained_config``. Missing fields stay ``None``
    (unknown): shape gates abstain instead of rejecting on absent info.
    """
    shapes = derive_moe_layer_shapes(
        model_config,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        top_k=top_k,
        routing=routing,
    )
    quant_config = override_quant_config or model_config.quant_config
    quant_algo = None if quant_config is None else quant_config.quant_algo

    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=shapes.dtype if shapes.dtype is not None else torch.bfloat16,
        hidden_size=shapes.hidden_size,
        intermediate_size=shapes.intermediate_size,
        num_experts=shapes.num_experts,
        top_k=shapes.top_k,
        swiglu_gptoss_style=swiglu_gptoss_style,
        bias=bias,
        activation=canonical_activation(activation_type),
        routing=canonical_routing(routing),
    )


def infer_swiglu_gptoss_style(
    *,
    bias: bool = False,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    activation_type: Optional[Union[ActivationType, int]] = None,
) -> bool:
    """True for the gpt-oss / MiniMax SwiGLU package (bias, alpha/beta, or SwigluBias).

    ``swiglu_limit`` alone is not enough — DeepSeek-V4 uses a plain clamp and
    must not be treated as gpt-oss.

    ``activation_type`` is normalized because ``MoE`` stores the activation as a
    plain ``int``, which no identity check against an enum member can match.
    """
    if activation_type is not None and ActivationType(activation_type) is ActivationType.SwigluBias:
        return True
    return bool(bias or swiglu_alpha is not None or swiglu_beta is not None)


def build_moe_deployment(
    model_config: ModelConfig,
    *,
    num_experts: Optional[int] = None,
    environment: Optional[MoEEnvironment] = None,
) -> MoEDeployment:
    """Assemble the deployment half, reading the same mapping ``MoE.__init__`` does."""
    mapping = model_config.mapping
    # MoE._init_load_balancer only adopts the config's slot count once a
    # balancer is registered; without one the layer keeps num_slots ==
    # num_experts, so the deployment must say the same.
    eplb_enabled = get_moe_load_balancer() is not None
    balancer_config = getattr(model_config, "moe_load_balancer", None)
    num_slots = getattr(balancer_config, "num_slots", None) if eplb_enabled else None
    if num_slots is None:
        num_slots = num_experts if num_experts is not None else 0
    lora_config = getattr(model_config, "lora_config", None)
    return MoEDeployment(
        ep_size=mapping.moe_ep_size,
        tp_size=mapping.moe_tp_size,
        # Same meaning as MoE.parallel_size (mapping.tp_size).
        parallel_size=mapping.tp_size,
        cluster_size=mapping.moe_cluster_size,
        use_dp=mapping.enable_attention_dp,
        num_slots=num_slots,
        env=environment if environment is not None else collect_moe_environment(),
        # Registered balancer only; config alone does not enable EPLB.
        eplb_enabled=eplb_enabled,
        # Routed-expert LoRA only; attention-only LoRA stays False.
        moe_lora_enabled=has_moe_lora_targets(lora_config),
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


# Backends whose whole point is to be the one that runs: silently degrading
# them to Cutlass would hand back the very numbers the caller asked to compare
# against. They fail with the rejection trail instead.
NO_FALLBACK_BACKENDS: FrozenSet[str] = frozenset({"VANILLA"})


def _candidates_for(backend: str) -> List[Type]:
    """Requested family in priority order, then fallback."""
    normalized = backend.upper()
    if normalized == "WIDEEP":
        raise ValueError(WIDEEP_DEPRECATION_MESSAGE)
    family = BACKEND_FAMILY.get(normalized)
    if family is None:
        raise ValueError(f"Unsupported moe backend: {backend}")
    candidates = [impl_cls for impl_cls in IMPL_PRIORITY if impl_cls in family]
    if FALLBACK_IMPL not in family and normalized not in NO_FALLBACK_BACKENDS:
        candidates.append(FALLBACK_IMPL)
    return candidates


def resolve_moe_impl(
    model_config: ModelConfig,
    *,
    problem: Optional[MoEProblem] = None,
    deployment: Optional[MoEDeployment] = None,
    override_quant_config: Optional[QuantConfig] = None,
    dtype: Optional[torch.dtype] = None,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    swiglu_gptoss_style: Optional[bool] = None,
    bias: Optional[bool] = None,
    activation_type: Optional[ActivationType] = None,
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"] = None,
    layer_idx: Optional[int] = None,
) -> MoEResolutionReport:
    """Resolve a MoE backend and return the full eligibility report.

    Raises ValueError for unknown or deprecated backend literals.
    """
    if problem is None:
        problem = build_moe_problem(
            model_config,
            override_quant_config=override_quant_config,
            dtype=dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            swiglu_gptoss_style=swiglu_gptoss_style,
            bias=bias,
            activation_type=activation_type,
            routing=routing,
        )
    if deployment is None:
        deployment = build_moe_deployment(model_config, num_experts=problem.num_experts)

    requested = model_config.moe_backend
    candidates = _candidates_for(requested)
    in_family = BACKEND_FAMILY[requested.upper()]

    # Ask all candidates so the report lists alternatives.
    rejected = []
    eligible: List[Type] = []
    for candidate in candidates:
        if deployment.moe_lora_enabled and not candidate.capabilities.supports_moe_lora:
            rejected.append(
                MoERejection(
                    _legacy_backend_name(candidate),
                    MoERejectReason.LORA_UNSUPPORTED,
                    f"{candidate.__name__} does not fuse routed-expert LoRA",
                )
            )
            continue
        eligibility = candidate.can_implement(problem, deployment)
        if eligibility.eligible:
            eligible.append(candidate)
            continue
        rejected.append(
            MoERejection(
                _legacy_backend_name(candidate),
                eligibility.reject_reason,
                eligibility.detail,
            )
        )

    # candidates is already priority-ordered.
    winner_cls = eligible[0] if eligible else None

    if winner_cls is None:
        selected_by = "failed"
    elif winner_cls in in_family:
        # Another family member still counts as pinned.
        selected_by = "pinned"
    else:
        selected_by = "heuristic"

    report = MoEResolutionReport(
        problem=problem,
        deployment=deployment,
        winner=None if winner_cls is None else _legacy_backend_name(winner_cls),
        rejected=tuple(rejected),
        eligible=tuple(_legacy_backend_name(impl_cls) for impl_cls in eligible),
        selected_by=selected_by,
        requested=requested,
        env_fingerprint=deployment.env.fingerprint(),
    )

    if report.degraded and winner_cls is not None:
        cause = report.degraded_from
        location = "" if layer_idx is None else f" [layer_idx={layer_idx}]"
        logger.warning(
            f"MoE backend {requested} cannot serve this layer{location} "
            f"({cause.reason.value}: {cause.detail}); running "
            f"{winner_cls.__name__} instead. Full trail: {report.describe()}"
        )
    else:
        logger.debug(report.describe())

    return report


def impl_class_for(report: MoEResolutionReport) -> Type:
    """The class a report's winner names, or raise with the whole trail."""
    if report.winner is None:
        raise ValueError(f"no MoE implementation can serve this layer. {report.describe()}")
    for candidate in IMPL_PRIORITY:
        if _legacy_backend_name(candidate) == report.winner:
            return candidate
    raise ValueError(
        f"resolution report names legacy backend {report.winner!r}, which no candidate class claims"
    )


def resolve_moe_cls(model_config: ModelConfig, **kwargs) -> Type:
    """Resolve and return only the implementation class."""
    return impl_class_for(resolve_moe_impl(model_config, **kwargs))
