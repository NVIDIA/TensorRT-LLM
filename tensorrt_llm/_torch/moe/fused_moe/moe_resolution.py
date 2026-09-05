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
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.peft.lora.validation import has_moe_lora_targets
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from .activation import ActivationParamShape, MoEActivation, activation_constant_names
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .fused_moe_deepgemm import DeepgemmCudaCppFp8BlockScalesImpl
from .fused_moe_densegemm import DenseGEMMFusedMoE
from .fused_moe_marlin import MarlinFusedMoE
from .fused_moe_triton import TritonFusedMoE
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .impl_base import MoEImplBase
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
from .impl_identity import MOE_IMPL_REGISTRY, MoEImplId, MoEImplQuery
from .interface import MoE
from .mega_moe import DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl, MegaMoECuteDsl
from .moe_load_balancer import get_moe_load_balancer

if TYPE_CHECKING:
    from .routing import BaseMoeRoutingMethod, RoutingMethodType

WIDEEP_DEPRECATION_MESSAGE = (
    "The WIDEEP MoE backend is deprecated and can no longer be selected. Wide "
    "expert parallelism and EPLB are supported by the other backends: use "
    "DEEPGEMM for FP8 block-scale checkpoints, or TRTLLM / CUTEDSL / CUTLASS "
    "otherwise."
)

#: What ``resolve_moe_cls`` can hand back. Three shapes, not two: ``MoE`` is a
#: complete layer, ``MoEImplBase`` an execution unit legal only as
#: ``ConfigurableMoE.backend``, and ``VanillaMoE`` the PyTorch reference path --
#: an ``nn.ModuleList`` that inherits neither, so a two-way union would exclude
#: the backend the other two are checked against. ``create_moe`` re-exports it.
MoEImplClass = type[MoE] | type[MoEImplBase] | type[VanillaMoE]

# Global priority: specialized first, broad fallbacks last. Both entrances
# intersect their candidate set with this tuple -- ``_candidates_for`` against a
# BACKEND_FAMILY entry, ``_candidates_for_impl_id`` against the registry -- so a
# class missing from here resolves to an empty candidate list.
# The DeepGEMM entries use the identity-derived names rather than the
# ``DeepGemmFusedMoE`` / ``MegaMoEDeepGemm`` aliases, so what is ranked here
# reads the same as what a resolution report prints.
IMPL_PRIORITY: Tuple[MoEImplClass, ...] = (
    CuteDslB12xFusedMoE,  # SM120/121 NVFP4 decode only -- narrowest, so first
    DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl,  # ahead of plain CuteDSL / DeepGEMM: better perf when eligible
    MegaMoECuteDsl,
    CuteDslFusedMoE,
    TRTLLMGenFusedMoE,
    DeepgemmCudaCppFp8BlockScalesImpl,
    DenseGEMMFusedMoE,
    MarlinFusedMoE,
    TritonFusedMoE,
    CutlassFusedMoE,  # widest coverage, hence the fallback
    VanillaMoE,  # reference implementation, never preferred
)

# Family membership only; IMPL_PRIORITY decides try order. The coarse
# ``moe_backend`` literal and a pinned identity reach the same class for the
# DeepGEMM families, so a run reports one name either way.
BACKEND_FAMILY: Dict[str, FrozenSet[MoEImplClass]] = {
    "CUTLASS": frozenset({CutlassFusedMoE}),
    "VANILLA": frozenset({VanillaMoE}),
    "MARLIN": frozenset({MarlinFusedMoE}),
    "CUTEDSL": frozenset({CuteDslB12xFusedMoE, CuteDslFusedMoE}),
    "DEEPGEMM": frozenset({DeepgemmCudaCppFp8BlockScalesImpl}),
    "DENSEGEMM": frozenset({DenseGEMMFusedMoE}),
    "TRTLLM": frozenset({TRTLLMGenFusedMoE}),
    "TRITON": frozenset({TritonFusedMoE}),
    "MEGAMOE_DEEPGEMM": frozenset({DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl}),
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
FALLBACK_IMPL: MoEImplClass = CutlassFusedMoE


def _legacy_backend_name(impl_cls: MoEImplClass) -> str:
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
    activation: Optional[MoEActivation] = None,
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"] = None,
) -> MoEProblem:
    """Assemble the problem half of a selection question.

    Explicit args win over ``pretrained_config``. Missing fields stay ``None``
    (unknown): shape gates abstain instead of rejecting on absent info.

    ``activation`` is the whole activation package, and both halves of it are
    read: the kind, and *which constants* the caller supplies -- a question the
    kind alone cannot answer, since clamped and unclamped SwiGLU share one
    ``ActivationType``. Pass it wherever it is in hand; without it the problem
    says "some activation" and the activation gate abstains.
    """
    activation_kind = None if activation is None else ActivationType(activation.kind)
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
        activation=canonical_activation(activation_kind),
        activation_constants=activation_constant_names(activation),
        routing=canonical_routing(routing),
    )


def infer_swiglu_gptoss_style(
    *,
    bias: bool = False,
    activation_type: Optional[Union[ActivationType, int]] = None,
) -> bool:
    """True for the gpt-oss / MiniMax SwiGLU package (expert bias, or SwigluBias).

    Keyed off the kind alone. The older form also answered True whenever an
    alpha/beta constant was merely *present*, which SiTU satisfies too, and that
    silently downgraded a MegaMoE request to Cutlass. A clamp was never part of
    it either -- DeepSeek-V4 uses a plain clamp.

    ``activation_type`` is still normalized even though ``MoE`` now stores an
    ``ActivationType``: the parameter is also reached from call sites that pass a
    bare int, and no identity check against an enum member would match one.
    """
    if activation_type is not None and ActivationType(activation_type) is ActivationType.SwigluBias:
        return True
    return bool(bias)


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
    locality_domain_policy = getattr(model_config, "locality_domain_policy", None)
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
        # Same expression the impls use to set ``use_fused_finalize``; any
        # LoRA counts, not just routed-expert LoRA.
        fused_finalize_enabled=(
            not getattr(model_config, "moe_disable_finalize_fusion", False) and lora_config is None
        ),
        locality_domain_requested=bool(
            locality_domain_policy is not None and locality_domain_policy.enabled
        ),
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


# Backends whose whole point is to be the one that runs: silently degrading
# them to Cutlass would hand back the very numbers the caller asked to compare
# against. They fail with the rejection trail instead.
NO_FALLBACK_BACKENDS: FrozenSet[str] = frozenset({"VANILLA"})


def _candidates_for(backend: str) -> List[MoEImplClass]:
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


def _coerce_impl_query(impl_id: Union[str, MoEImplId, MoEImplQuery]) -> MoEImplQuery:
    """Accept a full identity, a partial query, or the text form of either.

    Text goes through ``parse_query`` rather than field assignment because that
    is the door which rejects unknown tokens, and a pinned identity most often
    arrives as something a human wrote. Segment order carries no meaning:
    tokens are matched to fields by value.
    """
    if isinstance(impl_id, MoEImplQuery):
        return impl_id
    text = impl_id.canonical() if isinstance(impl_id, MoEImplId) else impl_id
    return MOE_IMPL_REGISTRY.parse_query(text)


def _candidates_for_impl_id(query: MoEImplQuery) -> List[MoEImplClass]:
    """Registered impls a pinned identity names, in global priority order.

    No fallback is appended, which is the whole difference from
    ``_candidates_for``. A caller that named an implementation is asking for
    that one; running a substitute would attribute another kernel's numbers to
    the identity that was pinned. So a pin whose gates all decline produces a
    ``winner is None`` report and ``impl_class_for`` raises with the trail.

    Matching nothing is the other failure, and it is a different one: it raises
    here, before any candidate is considered, for the same reason an unknown
    backend literal does -- there is no report worth returning when the request
    named nothing that exists.
    """
    matched = {impl_cls for _, impl_cls in MOE_IMPL_REGISTRY.find(query)}
    if not matched:
        raise ValueError(
            f"MoE impl {query.describe()!r} matches no registered implementation. "
            f"Registered: {sorted(str(identity) for identity in MOE_IMPL_REGISTRY.identities())}"
        )
    ranked = [impl_cls for impl_cls in IMPL_PRIORITY if impl_cls in matched]
    unranked = sorted(impl_cls.__name__ for impl_cls in matched.difference(ranked))
    if unranked:
        raise RuntimeError(
            f"MoE impls are registered but absent from IMPL_PRIORITY, so a pinned "
            f"request can never reach them: {unranked}"
        )
    return ranked


#: ABI register name -> the ``MoEActivationSupport`` field declaring its shape.
_CONSTANT_SHAPE_FIELDS: Dict[str, str] = {
    "alpha": "alpha_beta",
    "beta": "alpha_beta",
    "clamp": "limit",
}


def _reject_unsupported_activation(
    candidate: MoEImplClass, problem: MoEProblem
) -> Optional[MoERejection]:
    """Decline a candidate whose declaration cannot carry this activation.

    Central rather than repeated in eleven ``can_implement`` gates, because the
    answer is already written down: ``activation_support`` is the same
    declaration ``materialize_activation_params`` reads. A backend that forgot
    to re-derive it would not run the layer anyway -- it would raise from the
    adapter at construction, well past the point where another candidate could
    still have been chosen.

    Reads the *class* declaration. TRTLLM-Gen overrides
    ``resolve_activation_support`` per instance, but only to narrow
    ``PER_EXPERT_TENSOR`` to ``UNIFORM_SCALAR``; no instance turns a shape into
    ``UNSUPPORTED``, so no instance refuses a constant its class admits.
    """
    support = getattr(candidate, "activation_support", None)
    if support is None:
        return None

    kind = problem.activation_type
    if kind not in support.kinds:
        executes = ", ".join(sorted(k.name for k in support.kinds))
        return MoERejection(
            _legacy_backend_name(candidate),
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{candidate.__name__} does not execute {kind.name} (executes: {executes})",
        )

    # Sorted so the rejection names the same register every run.
    for constant in sorted(problem.activation_constants):
        field = _CONSTANT_SHAPE_FIELDS.get(constant)
        if field is None:
            continue
        if getattr(support, field) is ActivationParamShape.UNSUPPORTED:
            return MoERejection(
                _legacy_backend_name(candidate),
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{candidate.__name__} kernels take no activation {constant}, "
                f"which this layer's {kind.name} supplies",
            )
    return None


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
    activation: Optional[MoEActivation] = None,
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"] = None,
    layer_idx: Optional[int] = None,
    allow_degradation: bool = True,
    impl_id: Optional[Union[str, MoEImplId, MoEImplQuery]] = None,
) -> MoEResolutionReport:
    """Resolve a MoE backend and return the full eligibility report.

    ``allow_degradation=False`` turns the usual substitution warning into a
    hard failure. A caller that is measuring one specific backend needs that:
    silently running the fallback returns numbers attributed to the backend it
    asked for. The rejection trail says which gate declined and why, so the
    caller does not have to re-derive the winner and compare classes.

    ``impl_id`` pins a canonical implementation identity and, when given,
    replaces ``model_config.moe_backend`` as the request. The two are separate
    tracks on purpose: a backend literal names a coarse family and may degrade
    to the fallback, while an identity names one registered implementation and
    never degrades, so ``allow_degradation`` has nothing to act on here. A
    partial identity is legal and is resolved among its matches by
    ``IMPL_PRIORITY``, the same order the family path uses.

    Raises ValueError for unknown or deprecated backend literals, for unknown
    identity tokens, and for an identity that matches nothing registered.
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
            activation=activation,
            routing=routing,
        )
    if deployment is None:
        deployment = build_moe_deployment(model_config, num_experts=problem.num_experts)

    if impl_id is None:
        requested = model_config.moe_backend
        candidates = _candidates_for(requested)
        requested_impls: FrozenSet[MoEImplClass] = BACKEND_FAMILY[requested.upper()]
    else:
        query = _coerce_impl_query(impl_id)
        requested = query.describe()
        candidates = _candidates_for_impl_id(query)
        # Every candidate IS what was asked for, since no fallback was
        # appended, so a winner on this path is always "pinned".
        requested_impls = frozenset(candidates)

    # Ask all candidates so the report lists alternatives.
    rejected = []
    eligible: List[MoEImplClass] = []
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
        if not eligibility.eligible:
            rejected.append(
                MoERejection(
                    _legacy_backend_name(candidate),
                    eligibility.reject_reason,
                    eligibility.detail,
                )
            )
            continue
        # After can_implement, so a backend's own more specific reason wins.
        activation_rejection = _reject_unsupported_activation(candidate, problem)
        if activation_rejection is not None:
            rejected.append(activation_rejection)
            continue
        eligible.append(candidate)

    # candidates is already priority-ordered.
    winner_cls = eligible[0] if eligible else None

    if winner_cls is None:
        selected_by = "failed"
    elif winner_cls in requested_impls:
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

    if report.degraded and not allow_degradation:
        location = "" if layer_idx is None else f" [layer_idx={layer_idx}]"
        raise ValueError(
            f"MoE backend {requested} was requested with degradation disallowed "
            f"but cannot serve this layer{location}. {report.describe()}"
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


def impl_class_for(report: MoEResolutionReport) -> MoEImplClass:
    """The class a report's winner names, or raise with the whole trail."""
    if report.winner is None:
        # describe() prints reason codes only. With nothing left to run, the
        # operator needs the details too -- that is all the error can offer.
        details = "; ".join(
            f"{rejection.legacy_backend}: {rejection.detail}"
            for rejection in report.rejected
            if rejection.detail
        )
        raise ValueError(
            f"no MoE implementation can serve this layer. {report.describe()}"
            + (f" Details: {details}" if details else "")
        )
    for candidate in IMPL_PRIORITY:
        if _legacy_backend_name(candidate) == report.winner:
            return candidate
    raise ValueError(
        f"resolution report names legacy backend {report.winner!r}, which no candidate class claims"
    )


def resolve_moe_cls(model_config: ModelConfig, **kwargs) -> MoEImplClass:
    """Resolve and return only the implementation class."""
    return impl_class_for(resolve_moe_impl(model_config, **kwargs))
