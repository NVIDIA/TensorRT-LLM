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

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Collection, Dict, FrozenSet, List, Optional, Tuple, Union

import torch

from tensorrt_llm._errors import UnsupportedError
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.peft.lora.validation import has_moe_lora_targets
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from .activation import MoEActivation, SwigluActivation, activation_constant_names
from .cutedsl import (
    TrtllmCutedslDenseGemmNvfp4Impl,
    TrtllmCutedslFusedFc12Nvfp4Impl,
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslGroupedGemmNvfp4Impl,
    TrtllmCutedslGroupedGemmNvfp4PlusImpl,
)
from .cutlass import CUTLASS_LEAVES
from .fused_moe_cute_dsl_b12x import FlashinferCutedslNvfp4Impl, FlashinferCutedslW4a16Nvfp4Impl
from .fused_moe_deepgemm import DeepgemmCudaFp8BlockScalesImpl
from .fused_moe_triton import TritonFusedMoE
from .fused_moe_vanilla import VanillaMoE
from .impl_base import MoEImplBase
from .impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejection,
    MoERejectReason,
    MoEResolutionReport,
    activation_rejection_detail,
    canonical_activation,
    canonical_quant,
    canonical_routing,
    identity_quant_of,
    normalize_quant,
)
from .impl_environment import collect_moe_environment
from .impl_identity import MOE_IMPL_REGISTRY, MoEImplDescriptor, MoEImplId, MoEImplQuery
from .interface import MoE
from .marlin import MarlinCudaNvfp4Impl, MarlinCudaW4a16Nvfp4Impl
from .mega_moe import DeepgemmCudaW4a8Mxfp4Mxfp8Impl, TrtllmCutedslMegaMoeNvfp4Impl
from .moe_load_balancer import get_moe_load_balancer
from .trtllm_gen import (
    TrtllmTrtllmGenBf16Impl,
    TrtllmTrtllmGenFp8BlockScalesImpl,
    TrtllmTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenW4a8Mxfp4Fp8Impl,
    TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
    TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    TrtllmTrtllmGenW4a16Mxfp4Impl,
)

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

# Global priority: specialized first, broad fallbacks last. Both halves of
# ``moe_impl_request`` order their candidates by this tuple -- a backend literal
# its BACKEND_FAMILY entry, a tactic its registry matches -- so every such class
# is ranked here: a BACKEND_FAMILY member missing from it fails at import, and a
# registered class missing from it fails the first tactic that matches it.
# The registered entries use the identity-derived names rather than the
# ``DeepGemmFusedMoE`` / ``MegaMoEDeepGemm`` / ``MegaMoECuteDsl`` aliases, so
# what is ranked here reads the same as what a resolution report prints.
IMPL_PRIORITY: Tuple[MoEImplClass, ...] = (
    # The SM120/121 b12x leaves, one format each -- narrowest, so first.
    FlashinferCutedslNvfp4Impl,
    FlashinferCutedslW4a16Nvfp4Impl,
    DeepgemmCudaW4a8Mxfp4Mxfp8Impl,  # ahead of plain CuteDSL / DeepGEMM: better perf when eligible
    TrtllmCutedslMegaMoeNvfp4Impl,
    # Disjoint ``quant`` segments, so the order between them does not matter.
    TrtllmCutedslGroupedGemmNvfp4Impl,
    TrtllmCutedslGroupedGemmNvfp4PlusImpl,
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslFusedFc12Nvfp4Impl,
    # The TRTLLM-Gen leaves. Order among them does not matter: the ``quant``
    # segments are disjoint, so at most one leaf can admit a given problem.
    TrtllmTrtllmGenBf16Impl,
    TrtllmTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenFp8BlockScalesImpl,
    TrtllmTrtllmGenW4a16Mxfp4Impl,
    TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
    TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    TrtllmTrtllmGenW4a8Mxfp4Fp8Impl,
    DeepgemmCudaFp8BlockScalesImpl,
    TrtllmCutedslDenseGemmNvfp4Impl,
    MarlinCudaNvfp4Impl,
    MarlinCudaW4a16Nvfp4Impl,
    TritonFusedMoE,
    *CUTLASS_LEAVES,
    VanillaMoE,  # reference implementation, never preferred
)

# Family membership only; IMPL_PRIORITY decides try order. The coarse
# ``moe_backend`` literal and a pinned identity reach the same class for the
# DeepGEMM families, so a run reports one name either way.
BACKEND_FAMILY: Dict[str, FrozenSet[MoEImplClass]] = {
    # Includes the two ``fp8_block_scales`` leaves, whose technique is not
    # ``cutlass``: the literal serves FP8 block scales on SM90 and SM120/121.
    "CUTLASS": frozenset(CUTLASS_LEAVES),
    "VANILLA": frozenset({VanillaMoE}),
    "MARLIN": frozenset({MarlinCudaNvfp4Impl, MarlinCudaW4a16Nvfp4Impl}),
    "CUTEDSL": frozenset(
        {
            FlashinferCutedslNvfp4Impl,
            FlashinferCutedslW4a16Nvfp4Impl,
            TrtllmCutedslGroupedGemmNvfp4Impl,
            TrtllmCutedslGroupedGemmNvfp4PlusImpl,
            TrtllmCutedslGroupedGemmBf16Impl,
        }
    ),
    "CUTEDSL_FC12": frozenset({TrtllmCutedslFusedFc12Nvfp4Impl}),
    "DEEPGEMM": frozenset({DeepgemmCudaFp8BlockScalesImpl}),
    "DENSEGEMM": frozenset({TrtllmCutedslDenseGemmNvfp4Impl}),
    # The coarse literal still names the whole family, so ``moe_backend:
    # TRTLLM`` keeps meaning "any TRTLLM-Gen leaf" and the format picks
    # which. A pinned ``impl_id`` names exactly one of them.
    "TRTLLM": frozenset(
        {
            TrtllmTrtllmGenBf16Impl,
            TrtllmTrtllmGenNvfp4Impl,
            TrtllmTrtllmGenFp8BlockScalesImpl,
            TrtllmTrtllmGenW4a16Mxfp4Impl,
            TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
            TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
            TrtllmTrtllmGenW4a8Mxfp4Fp8Impl,
        }
    ),
    "TRITON": frozenset({TritonFusedMoE}),
    "MEGAMOE_DEEPGEMM": frozenset({DeepgemmCudaW4a8Mxfp4Mxfp8Impl}),
    "MEGAMOE_CUTEDSL": frozenset({TrtllmCutedslMegaMoeNvfp4Impl}),
}

# Catch table drift at import time.
_UNRANKED = {cls for family in BACKEND_FAMILY.values() for cls in family} - set(IMPL_PRIORITY)
if _UNRANKED:
    raise RuntimeError(
        f"MoE impls named by BACKEND_FAMILY but absent from IMPL_PRIORITY: "
        f"{sorted(cls.__name__ for cls in _UNRANKED)}"
    )


def backend_family_of(impl_cls: MoEImplClass) -> Optional[str]:
    """The ``moe_backend`` literal whose family contains ``impl_cls``.

    The families above are disjoint, so at most one name matches. ``None``
    means the class is reachable only through a pinned identity, never through
    the coarse literal.
    """
    for name, family in BACKEND_FAMILY.items():
        if impl_cls in family:
            return name
    return None


# Default degradation target: the Cutlass leaves, whose coverage is the
# widest between them.
FALLBACK_IMPLS: tuple[MoEImplClass, ...] = CUTLASS_LEAVES


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
        clamp_after_silu=(
            activation.clamp_after_silu if isinstance(activation, SwigluActivation) else False
        ),
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
    eplb_mode = getattr(balancer_config, "mode", "standard") if eplb_enabled else "disabled"
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
        eplb_mode=eplb_mode,
        # Routed-expert LoRA only; attention-only LoRA stays False.
        moe_lora_enabled=has_moe_lora_targets(lora_config),
        force_dynamic_quantization=model_config.force_dynamic_quantization,
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


def _published_quant(impl_cls: MoEImplClass) -> Optional[str]:
    """The format ``impl_cls`` publishes; ``None`` for a class without an
    identity (``TritonFusedMoE``, ``VanillaMoE``)."""
    if getattr(impl_cls, "descriptor", None) is None:
        return None
    return identity_quant_of(impl_cls)


def _implementations_of(quant: str) -> Tuple[str, ...]:
    """The ids of the registered implementations of ``quant``, in priority order."""
    matched = {impl_cls for _, impl_cls in MOE_IMPL_REGISTRY.find(MoEImplQuery(quant=quant))}
    return tuple(impl_cls.descriptor.impl_id for impl_cls in IMPL_PRIORITY if impl_cls in matched)


def _covering_tactics(query: MoEImplQuery, quants: Collection[str]) -> Tuple[str, ...]:
    """Tactics that drop the fewest segments of ``query`` and name
    implementations of every format in ``quants``, all of one backend family.

    Only dropping keeps what the tactic pins pinned, and at least one segment
    stays; the empty result means no such tactic exists.
    """
    pinned = [
        name
        for name in ("quant", "provider", "kernel_name", "technique")
        if getattr(query, name) is not None
    ]
    for size in range(1, len(pinned)):
        found = []
        for names in itertools.combinations(pinned, size):
            widened = replace(query, **{name: None for name in names})
            matched = [
                impl_cls
                for _, impl_cls in MOE_IMPL_REGISTRY.find(widened)
                if _published_quant(impl_cls) in quants
            ]
            families = {backend_family_of(impl_cls) for impl_cls in matched}
            if (
                {_published_quant(impl_cls) for impl_cls in matched} >= set(quants)
                and len(families) == 1
                and None not in families
            ):
                found.append(widened.compact())
        if found:
            return tuple(found)
    return ()


def _one_family_tactics(
    query: MoEImplQuery, impls: Collection[MoEImplClass]
) -> Dict[str, Tuple[str, ...]]:
    """Tactics that each name one backend family among ``impls``, by family.

    ``impls`` are what ``query`` matches, spread over several families. For
    each of them the tactic is ``query`` plus the fewest segments of that
    implementation's own id that leave only its family among ``impls``. Ties go
    to ``technique``, then ``kernel_name``, ``provider`` and ``quant``: a
    provider segment would drop the other provider's leaves of the same family,
    such as the native TRTLLM-Gen leaf next to its FlashInfer twin. A family no
    literal reaches is left out, since a tactic naming only it cannot load.
    """
    identities = {impl_cls: impl_cls.descriptor.identity for impl_cls in impls}
    open_fields = [
        name
        for name in ("technique", "kernel_name", "provider", "quant")
        if getattr(query, name) is None
    ]
    tactics: Dict[str, List[str]] = {}
    for impl_cls, identity in identities.items():
        family = backend_family_of(impl_cls)
        if family is None:
            continue
        # Fewest added segments first; adding every open one gives the full id,
        # which matches only ``impl_cls``, so ``next`` always finds one.
        narrowings = (
            replace(query, **{name: getattr(identity, name) for name in names})
            for size in range(len(open_fields) + 1)
            for names in itertools.combinations(open_fields, size)
        )
        tactic = next(
            narrowing.compact()
            for narrowing in narrowings
            if all(
                backend_family_of(other) == family
                for other, other_identity in identities.items()
                if narrowing.matches(other_identity)
            )
        )
        if tactic not in tactics.setdefault(family, []):
            tactics[family].append(tactic)
    return {family: tuple(found) for family, found in tactics.items()}


@dataclass(frozen=True)
class MoEImplRequest:
    """The implementations a MoE layer is asked to run, built only by
    :func:`moe_impl_request`.

    A ``moe_config.tactic`` and a ``moe_backend`` literal are two spellings of
    one request. A tactic is a pin: its matches and nothing else, and when none
    of them can serve the layer resolution fails with the rejection trail. A
    literal names its family and may degrade to ``fallback``, a legacy behavior
    that is deprecated together with ``moe_backend``.

    Attributes:
        label: The request as diagnostics and ``MoEResolutionReport.requested``
            print it: the literal, or the tactic's query.
        impls: What was asked for, in ``IMPL_PRIORITY`` order.
        fallback: Tried after ``impls`` when none of them accepts; empty for a
            pin and for a literal with no fallback. Legacy ``moe_backend``
            behavior only, deprecated with it: it runs an implementation the
            user did not ask for. New request kinds must leave it empty, so
            that what is configured is what runs, and a configuration no
            implementation serves fails with an explicit error.
        query: The tactic, when one built the request.
    """

    label: str
    impls: Tuple[MoEImplClass, ...]
    fallback: Tuple[MoEImplClass, ...] = ()
    query: Optional[MoEImplQuery] = None

    @property
    def pinned(self) -> bool:
        """Whether a tactic built the request, so that it never degrades."""
        return self.query is not None

    @property
    def candidates(self) -> Tuple[MoEImplClass, ...]:
        """What resolution asks, in order: the request, then its fallback."""
        return self.impls + self.fallback

    def identity_quants(self) -> FrozenSet[str]:
        """The formats the candidates publish, spelled as identities spell them."""
        published = (_published_quant(impl_cls) for impl_cls in self.candidates)
        return frozenset(quant for quant in published if quant is not None)

    def narrowed_to(self, quants: Collection[str]) -> "MoEImplRequest":
        """The same request with only the implementations of ``quants``."""
        return replace(
            self,
            impls=tuple(
                impl_cls for impl_cls in self.impls if _published_quant(impl_cls) in quants
            ),
        )

    def families(self) -> Dict[Optional[str], Tuple[str, ...]]:
        """``{backend family: (impl id, ...)}`` of ``impls``, in priority order.

        Family membership is a declaration, so this asks no gate.
        """
        families: Dict[Optional[str], List[str]] = {}
        for impl_cls in self.impls:
            descriptor = getattr(impl_cls, "descriptor", None)
            name = impl_cls.__name__ if descriptor is None else descriptor.impl_id
            families.setdefault(backend_family_of(impl_cls), []).append(name)
        return {family: tuple(names) for family, names in families.items()}

    def backend_for(self, quants: Collection[str]) -> str:
        """The ``moe_backend`` literal model code is built for on a checkpoint of ``quants``.

        Only the implementations of the checkpoint's formats can run, so
        ``impls`` are narrowed to ``quants`` first: the tactic ``mega_moe`` is
        MegaMoE CuteDSL on an NVFP4 checkpoint and MegaMoE DeepGEMM on an
        MXFP4 one. When none of them serves those formats, the family of all of
        them. A literal's request is its own family either way, as the
        families are disjoint. Family membership is a declaration, so this
        asks no gate.

        TODO: Remove together with ``moe_backend``. This is a transitional
        bridge, not part of selection: a tactic alone selects the
        implementations, and the literal is derived only for the model code
        that still branches on it (gate and router dtypes, checkpoint weight
        mappers, weight preloading, quant-config defaults). Once those
        branches read the resolved implementation's descriptor, no caller
        needs a literal. The refusal of a request spanning several families
        exists for the same reason, since that model code is built for one.

        Args:
            quants: The checkpoint's formats, spelled as identities spell them.

        Raises:
            ValueError: The narrowed request names implementations of several
                families (``grouped_gemm.nvfp4`` names a CUTLASS and a CuteDSL
                leaf), or ones no literal reaches.
        """
        narrowed = self.narrowed_to(quants)
        impls = narrowed.impls or self.impls
        families = narrowed.families() or self.families()
        if len(families) == 1 and None not in families:
            return next(iter(families))
        listing = "; ".join(f"{family}: {', '.join(ids)}" for family, ids in families.items())
        if len(families) == 1:
            raise ValueError(
                f"MoE request {self.label!r} names only implementations no "
                f"moe_backend literal reaches ({listing}), and model code is "
                "built for a literal before any layer is resolved."
            )
        advice = "give one of the ids listed"
        if self.query is not None:
            by_family = _one_family_tactics(self.query, impls)
            advice = (
                "for example "
                + "; ".join(
                    f"{' or '.join(repr(tactic) for tactic in tactics)} for {family}"
                    for family, tactics in by_family.items()
                )
                + ", or give one of the ids listed"
            )
        raise ValueError(
            f"MoE request {self.label!r} names MoE implementations of several "
            f"backend families for this checkpoint (quant {sorted(quants)}): "
            f"{listing}. Model code is built for one family before any layer is "
            f"resolved, so name implementations of one family: {advice}."
        )


def moe_impl_request(
    backend: Optional[str] = None,
    *,
    tactic: Optional[Union[str, MoEImplId, MoEImplQuery]] = None,
) -> MoEImplRequest:
    """Turn a ``moe_backend`` literal or a ``moe_config.tactic`` into implementations.

    The one place a backend literal is translated, so that resolution, model
    loading and the MXFP4 format rule read the same request. The two are
    alternatives and exactly one must be given. ``MoeConfig`` still accepts
    its default ``AUTO`` next to a tactic, but callers drop the backend
    whenever a tactic is set, so no ``AUTO`` reaches this function.

    Args:
        backend: A ``moe_backend`` literal; ``AUTO`` must be resolved first
            (``ModelConfig.resolve_moe_backend``).
        tactic: An implementation id, a dotted subset of one, or a query.

    Raises:
        ValueError: Both or neither given; an unknown or deprecated literal; a
            tactic with unknown tokens, matching nothing registered, or
            constraining no field.
    """
    if backend is not None and tactic is not None:
        raise ValueError(
            f"a MoE impl request takes a backend literal or a tactic, not both "
            f"(backend={backend!r}, tactic={tactic!r})"
        )
    if tactic is not None:
        query = _coerce_impl_query(tactic)
        return MoEImplRequest(
            label=query.describe(), impls=tuple(_candidates_for_impl_id(query)), query=query
        )
    if backend is None:
        raise ValueError("a MoE impl request needs a backend literal or a tactic")
    normalized = backend.upper()
    if normalized == "WIDEEP":
        raise ValueError(WIDEEP_DEPRECATION_MESSAGE)
    family = BACKEND_FAMILY.get(normalized)
    if family is None:
        raise ValueError(f"Unsupported moe backend: {backend}")
    # A request for CUTLASS already holds every fallback leaf.
    degrades = family.isdisjoint(FALLBACK_IMPLS) and normalized not in NO_FALLBACK_BACKENDS
    return MoEImplRequest(
        label=backend,
        impls=tuple(impl_cls for impl_cls in IMPL_PRIORITY if impl_cls in family),
        fallback=FALLBACK_IMPLS if degrades else (),
    )


# The formats MXFP4 weights run as, most preferred first. Blackwell and later
# quantize the activations too; earlier GPUs keep them in 16 bits.
#
# The order decides only for a request that leaves the format open: every
# backend literal, whose family and CUTLASS fallback leaves together serve all
# three formats (VANILLA publishes none); an unresolved AUTO; and a tactic such
# as "cutlass" that matches several formats. A tactic naming one format, by its
# quant segment or by matching implementations of one format only, runs as
# that format.
_MXFP4_FORMATS_SM100_AND_LATER = (
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W4A8_MXFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
)
_MXFP4_FORMATS_BEFORE_SM100 = (
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W4A8_MXFP4_FP8,
)


def mxfp4_quant_algo(request: Optional[MoEImplRequest], sm_version: int) -> QuantAlgo:
    """The format an MXFP4 checkpoint's experts run as.

    MXFP4 weights run as several formats that differ in how activations are
    quantized, so the checkpoint does not fix one. The SM's most preferred
    format among those the request's candidates publish wins, so a request
    naming one MXFP4 format runs as that format; a request publishing none
    (``None`` stands for an ``AUTO`` backend not resolved yet) takes the SM's
    first choice.
    """
    preference = (
        _MXFP4_FORMATS_SM100_AND_LATER if sm_version >= 100 else _MXFP4_FORMATS_BEFORE_SM100
    )
    published = frozenset() if request is None else request.identity_quants()
    for algo in preference:
        if normalize_quant(canonical_quant(algo)) in published:
            return algo
    return preference[0]


def _coerce_impl_query(impl_id: Union[str, MoEImplId, MoEImplQuery]) -> MoEImplQuery:
    """Accept a full identity, a partial query, or the text form of either.

    Text goes through ``parse_query``, which is the door that rejects unknown
    tokens. Segment order carries no meaning; tokens match fields by value.
    """
    if isinstance(impl_id, MoEImplQuery):
        return impl_id
    text = impl_id.canonical() if isinstance(impl_id, MoEImplId) else impl_id
    return MOE_IMPL_REGISTRY.parse_query(text)


def _candidates_for_impl_id(query: MoEImplQuery) -> List[MoEImplClass]:
    """Registered impls a pinned identity names, in global priority order.

    No fallback is appended -- the whole difference from a backend literal.
    Substituting would attribute another kernel's numbers to the pinned
    identity, so a pin whose gates all decline yields a ``winner is None``
    report and ``impl_class_for`` raises with the trail.

    Two requests raise here instead, before any candidate is considered: one
    matching nothing registered, and one pinning no field at all, which would
    silently widen into "any registered impl" while still discarding the
    caller's backend literal.
    """
    if query.is_empty:
        raise ValueError(
            "MoE impl query constrains no field, so it names every registered "
            "implementation rather than one. Pin at least one segment, or pin "
            "nothing to select by backend."
        )
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


def _reject_unsupported_activation(
    candidate: MoEImplClass, problem: MoEProblem
) -> Optional[MoERejection]:
    """Decline a candidate whose declaration cannot carry this activation.

    Central rather than repeated in every ``can_implement`` gate, because the
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
    detail = activation_rejection_detail(candidate, problem)
    if detail is None:
        return None
    return MoERejection(
        _legacy_backend_name(candidate), MoERejectReason.ACTIVATION_UNSUPPORTED, detail
    )


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
    replaces ``model_config.moe_backend`` as the request. A backend literal
    names a coarse family and may degrade to the fallback; an identity never
    degrades, so ``allow_degradation`` is inert on this track. A partial
    identity is legal and is disambiguated among its matches by
    ``IMPL_PRIORITY``; any winner inside that match set still counts as
    pinned, so ``report.degraded`` stays False for pins of either width.

    Without ``impl_id`` the request is ``model_config.moe_tactic``, the
    user's ``moe_config.tactic`` (or ``speculative_config.moe_tactic`` on a
    draft copy), so every caller resolving a layer honours the same pin. A
    layer in a format none of its implementations serves is refused, since
    selecting it by ``moe_backend`` would run something the tactic does not
    name. An ``impl_id`` argument pins every layer it is passed for, and a
    layer it cannot serve gets a failed report.

    Raises ValueError for unknown or deprecated backend literals, for unknown
    identity tokens, for an identity that matches nothing registered, for
    one that constrains no field and so would name everything, and for a
    model-level tactic that names no implementation of the layer's format or
    runs one outside the backend family the model was built for.
    """
    pinned_by_caller = impl_id is not None
    tactic = impl_id if pinned_by_caller else model_config.moe_tactic

    location = "" if layer_idx is None else f" [layer_idx={layer_idx}]"

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

    # Loading derives ``moe_backend`` from the model's tactic
    # (``pinned_moe_backend``), so a tactic is the whole request.
    request = (
        moe_impl_request(model_config.moe_backend)
        if tactic is None
        else moe_impl_request(tactic=tactic)
    )
    if request.pinned and not pinned_by_caller:
        pinned_quants = request.identity_quants()
        if problem.identity_quant not in pinned_quants:
            implementations = _implementations_of(problem.identity_quant)
            serving = (
                f"the implementations of that format are {', '.join(implementations)}"
                if implementations
                else "no registered implementation serves that format"
            )
            covering = _covering_tactics(request.query, pinned_quants | {problem.identity_quant})
            widen = (
                "Widen it so that it names that format too, for example to "
                + " or ".join(repr(tactic) for tactic in covering)
                if covering
                else "No tactic made by dropping its segments names that format too "
                "within one backend family"
            )
            raise ValueError(
                f"MoE tactic {request.label} names no implementation of "
                f"quant={problem.identity_quant} (it names {sorted(pinned_quants)}), so "
                f"it cannot build the MoE layers in that format{location}. {widen}; "
                f"{serving}. Draft models and MTP layers are selected by "
                f"speculative_config.moe_tactic or speculative_config.moe_backend."
            )

    requested = request.label
    # A pin has no fallback, so every candidate IS what was asked for and a
    # winner on that path is always "pinned".
    requested_impls = frozenset(request.impls)

    # Ask all candidates so the report lists alternatives.
    rejected = []
    eligible: List[MoEImplClass] = []
    for candidate in request.candidates:
        if (
            deployment.eplb_mode == "per_layer_dynamic"
            and not candidate.capabilities.supports_per_layer_dynamic_eplb
        ):
            rejected.append(
                MoERejection(
                    _legacy_backend_name(candidate),
                    MoERejectReason.EPLB_UNSUPPORTED,
                    f"{candidate.__name__} does not consume per-layer dynamic EPLB plans",
                )
            )
            continue
        if deployment.eplb_enabled and not candidate.capabilities.supports_eplb:
            rejected.append(
                MoERejection(
                    _legacy_backend_name(candidate),
                    MoERejectReason.EPLB_UNSUPPORTED,
                    f"{candidate.__name__} does not support EPLB slot layouts",
                )
            )
            continue
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

    # The candidates are already priority-ordered.
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
        requested_impls=frozenset(_legacy_backend_name(impl_cls) for impl_cls in requested_impls),
    )

    if report.degraded and not allow_degradation:
        raise UnsupportedError(
            f"MoE backend {requested} was requested with degradation disallowed "
            f"but cannot serve this layer{location}. {report.describe()}"
            f" Each candidate's reason: {report.describe_rejections()}"
        )

    if winner_cls is None:
        # Gates abstain rather than raise, so with all of them abstaining this
        # is the only place the individual reasons surface: ``describe()``
        # carries reason codes alone, and the caller may swallow
        # ``impl_class_for``'s error.
        logger.warning(
            f"No MoE implementation can serve this layer{location} "
            f"(requested {requested}). Each candidate's reason: "
            f"{report.describe_rejections()}"
        )
    elif report.degraded:
        cause = report.degraded_from
        others = ", ".join(
            f"{rejection.legacy_backend}={rejection.reason.value}"
            for rejection in report.requested_rejections
            if rejection is not cause
        )
        logger.warning(
            f"MoE backend {requested} cannot serve this layer{location} "
            f"({'no candidate was considered' if cause is None else cause.describe()}); "
            f"running {winner_cls.__name__} instead."
            + (f" Also turned down: {others}." if others else "")
        )
    else:
        # Only the undegraded path reaches here, so ``describe()`` is built at
        # most once per resolution: the warnings above carry their own text.
        logger.debug(report.describe())

    if request.pinned and winner_cls is not None:
        # Model code outside selection still branches on the
        # ``model_config.moe_backend`` literal (router and bias dtypes, finalize
        # fusion, weight preloading). Loading derives that literal from the
        # checkpoint's expert formats before any layer exists
        # (``ModelConfig.pinned_moe_backend``), an approximation this layer
        # checks: model code built for another family than the one running
        # would be wrong, so a model-level tactic is refused here. A caller's
        # ``impl_id`` measures one implementation on purpose and only warns.
        winner_family = backend_family_of(winner_cls)
        configured = model_config.moe_backend
        if winner_family != configured.upper():
            family = (
                "no backend family" if winner_family is None else f"backend family {winner_family}"
            )
            if not pinned_by_caller:
                raise ValueError(
                    f"MoE tactic {requested} runs {winner_cls.__name__} ({family}) "
                    f"on this layer{location}, but the model was built for "
                    f"moe_backend {configured}, the family loading derived from "
                    f"the checkpoint's expert formats. Model code that still "
                    f"branches on the backend name would follow {configured}. "
                    f"Narrow the tactic so that it names implementations of one "
                    f"backend family, for example by giving one of its full ids."
                )
            logger.warning_once(
                f"MoE impl pin {requested} runs {winner_cls.__name__} ({family}), "
                f"while the model was configured for moe_backend {configured}. "
                f"Model code that still branches on the backend name follows "
                f"{configured}. Narrow the pin so that it names implementations "
                f"of one backend family.",
                key=("moe_impl_pin_family", requested, winner_cls, configured),
            )

    return report


def impl_class_for(report: MoEResolutionReport, *, advice: str = "") -> MoEImplClass:
    """The class a report's winner names, or raise with the whole trail.

    ``advice`` is what the caller knows about the request and the report does
    not, such as the config field that set it; the error puts it ahead of the
    trail.
    """
    if report.winner is None:
        # describe() prints reason codes only. With nothing left to run, the
        # operator needs the details too -- that is all the error can offer.
        head = "no MoE implementation can serve this layer."
        if advice:
            head += f" {advice}"
        raise UnsupportedError(
            f"{head} {report.describe()} Each candidate's reason: {report.describe_rejections()}"
        )
    for candidate in IMPL_PRIORITY:
        if _legacy_backend_name(candidate) == report.winner:
            return candidate
    raise ValueError(
        f"resolution report names legacy backend {report.winner!r}, which no candidate class claims"
    )


def resolve_moe_cls(model_config: ModelConfig, **kwargs) -> MoEImplClass:
    """Resolve and return only the implementation class.

    Every MoE layer is built from what this returns, so a run says here, once
    per implementation, which one it builds and the id a MoE tactic names it
    by: running once without a tactic shows what to pin.
    """
    report = resolve_moe_impl(model_config, **kwargs)
    advice = ""
    if report.winner is None and kwargs.get("impl_id") is None and model_config.moe_tactic:
        advice = (
            f"The MoE tactic {report.requested} names only implementations that "
            "decline this layer, and a tactic runs nothing else: change "
            "moe_config.tactic (speculative_config.moe_tactic for a draft model or "
            "MTP layers) to name one that accepts it."
        )
    impl_cls = impl_class_for(report, advice=advice)
    family = backend_family_of(impl_cls) or "no backend family"
    descriptor = getattr(impl_cls, "descriptor", None)
    if isinstance(descriptor, MoEImplDescriptor):
        logger.info_once(
            f"MoE layers run {impl_cls.__name__}, MoE implementation id "
            f"{descriptor.impl_id} (moe_backend {family}). A MoE tactic "
            f"(moe_config.tactic, speculative_config.moe_tactic) pins it by this "
            f"id or by a dotted subset of it.",
            key=("moe_impl_built", impl_cls),
        )
    else:
        logger.info_once(
            f"MoE layers run {impl_cls.__name__} (moe_backend {family}), which "
            f"has no MoE implementation id, so a MoE tactic cannot pin it.",
            key=("moe_impl_built", impl_cls),
        )
    return impl_cls
