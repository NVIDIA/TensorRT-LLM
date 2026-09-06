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
"""Contracts for declaring, selecting, and executing MoE implementations."""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.utils import ActivationType
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    from .moe_load_balancer import SingleLayerMoeLoadBalancer
    from .routing import BaseMoeRoutingMethod, RoutingMethodType

# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEStaticCapability:
    """Static abilities used during selection. Defaults are conservative."""

    # Legacy gate: ``moe.backend.__class__ == CutlassFusedMoE`` in MoEScheduler.
    supports_moe_lora: bool = False
    # Legacy gate: CuteDslFusedMoE isinstance check in ConfigurableMoE DWDP.
    supports_dwdp: bool = False
    # Legacy gate: ``assert moe_cls in supported_load_balancer_backends`` in
    # ``create_moe_backend``. Not the same question as the instance-level
    # ``_supports_load_balancer()``, which TRTLLMGenFusedMoE overrides to mean
    # "separated routing is used".
    supports_eplb: bool = False
    # Legacy gate: the ``assert moe_cls in [...]`` bias allow-list in
    # ``create_moe_backend``. Per-expert FC bias from the checkpoint, added
    # before the activation functor runs -- not an activation constant.
    supports_expert_bias: bool = False
    # Legacy gate: the three ``assert not apply_router_weight_on_input`` checks
    # keyed on ``moe_cls`` in ``create_moe_backend``. The fold itself belongs to
    # MoEScheduler (``x = x * token_final_scales``), so what a backend declares
    # here is whether it handles what the fold leaves behind: ``None`` scales,
    # or all-ones under a DeepEP / NCCL comm strategy. Backends that reject the
    # flag in their own constructor keep doing so; that check guards direct
    # construction, which never reaches the factory.
    supports_apply_router_weight_on_input: bool = False


@dataclass(frozen=True)
class MoEInputRequirement:
    """Caller-side inputs the scheduler must prepare; not a selection filter."""

    # Legacy: the ``token_final_scales`` bfloat16 / float32 casts in
    # MoEScheduler.
    routing_scales_dtype: Optional[torch.dtype] = None
    # Legacy: the DeepGemmFusedMoE isinstance branches in MoEScheduler.
    requires_run_moe_workspace: bool = False
    # Legacy: the DeepEP-comm plus TRTLLMGenFusedMoE special case in
    # MoEScheduler.
    requires_sanitized_expert_ids: bool = False
    # Overrides the NVLink one-sided combine payload dtype; None means the
    # buffer follows the model output dtype.
    onesided_workspace_dtype: Optional[torch.dtype] = None

    # No sentinel for invalid_token_expert_id: every Communication uses -1.

    # There is deliberately no ``requires_router_logits`` field either, though
    # the design sketched one to replace the scheduler's router-logits filter.
    # A class-level bool cannot express that condition: it also depends on the
    # routing method instance and on an environment override, neither of which
    # is known per class. ``TRTLLMGenFusedMoE._routes_outside_the_kernel``
    # answers it instead, next to the kernel whose contract it describes.


# ---------------------------------------------------------------------------
# Selection inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEProblem:
    """Reusable tuning-key inputs for a MoE layer.

    Eligibility gates abstain when an optional field is unknown.
    """

    quant: Optional[str]  # QuantAlgo value, or None for unquantized
    dtype_act: torch.dtype  # activation dtype BEFORE quantization
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    #: Tri-state because some call sites cannot distinguish gpt-oss SwiGLU.
    swiglu_gptoss_style: Optional[bool] = None
    #: Expert FC bias, distinct from ``swiglu_gptoss_style``. MiniMax sets
    #: SwigluBias + alpha/beta/limit with ``bias=False``; gpt-oss sets both.
    bias: Optional[bool] = None
    #: ``ActivationType`` member name; omitted values canonicalize to SwiGLU.
    activation: str = "Swiglu"
    #: Which of the ``alpha`` / ``beta`` / ``clamp`` ABI registers the caller's
    #: activation fills; the kind alone does not say, since clamped and
    #: unclamped SwiGLU share one ``ActivationType``. Empty if the call site
    #: supplied no activation carrier.
    activation_constants: frozenset[str] = frozenset()
    #: ``RoutingMethodType`` member name; None means the call site did not say.
    routing: Optional[str] = None

    @property
    def routing_method_type(self) -> Optional["RoutingMethodType"]:
        """``routing`` as the enum member, or ``None`` when unknown."""
        from .routing import RoutingMethodType

        if self.routing is None:
            return None
        return RoutingMethodType[self.routing]

    @property
    def activation_type(self) -> "ActivationType":
        """Return ``activation`` as an enum member."""
        from tensorrt_llm._torch.utils import ActivationType

        return ActivationType[self.activation]

    @property
    def quant_algo(self) -> Optional["QuantAlgo"]:
        """Return ``quant`` as an enum member."""
        if self.quant is None:
            return None
        from tensorrt_llm.models.modeling_utils import QuantAlgo

        return QuantAlgo(self.quant)

    @property
    def is_fully_specified(self) -> bool:
        """Whether this problem can key a persisted tuning result."""
        return None not in (self.hidden_size, self.intermediate_size, self.num_experts, self.top_k)


def canonical_quant(quant_algo: Optional["QuantAlgo"]) -> Optional[str]:
    """Canonicalize a quantization algorithm for the tuning key."""
    if quant_algo is None:
        return None
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    aliases = {
        # Calibration recipes; the weights and the kernel are plain NVFP4.
        QuantAlgo.NVFP4_AWQ: QuantAlgo.NVFP4,
        QuantAlgo.NVFP4_ARC: QuantAlgo.NVFP4,
        # MIXED_PRECISION is a model-level marker, not a layer format.
        QuantAlgo.MIXED_PRECISION: None,
        QuantAlgo.NO_QUANT: None,
    }
    resolved = aliases.get(quant_algo, quant_algo)
    return None if resolved is None else str(resolved.value)


def canonical_activation(activation_type: Optional["ActivationType"]) -> str:
    """Canonicalize an activation for the tuning key."""
    from tensorrt_llm._torch.utils import ActivationType

    if activation_type is None:
        return ActivationType.Swiglu.name
    return ActivationType(activation_type).name


def canonical_routing(
    routing: Optional["BaseMoeRoutingMethod | RoutingMethodType"],
) -> Optional[str]:
    """Canonicalize a routing method or method type for the tuning key."""
    from .routing import RoutingMethodType

    if routing is None:
        return None
    if not isinstance(routing, RoutingMethodType):
        routing = routing.routing_method_type
    return RoutingMethodType(routing).name


@dataclass(frozen=True)
class MoEEnvironment:
    """Everything probed from the machine, collected ONCE and then frozen.

    This structure exists to make the probes explicit inputs. Once selection
    moves onto it, ``can_implement`` must not call ``get_sm_version()``, must
    not import optional deps to test for their presence, and must not read
    ``os.environ`` -- it reads ``d.env`` instead.
    """

    sm: int  # e.g. 100, 103
    available_deps: Tuple[str, ...] = ()
    env_flags: Tuple[Tuple[str, str], ...] = ()  # sorted (name, value) pairs

    def has_dep(self, name: str) -> bool:
        return name in self.available_deps

    def fingerprint(self) -> str:
        """Return a stable fingerprint for the selection environment."""
        payload = repr((self.sm, sorted(self.available_deps), self.env_flags))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MoEDeployment:
    """Topology and slot layout used for eligibility."""

    ep_size: int
    tp_size: int
    use_dp: bool
    num_slots: int
    env: MoEEnvironment
    # Whole parallel-group width from mapping.tp_size.
    parallel_size: int
    # mapping.moe_cluster_size; values above one enable the smart router.
    cluster_size: int = 1
    # True only when an EPLB load balancer is registered.
    eplb_enabled: bool = False
    # True only for routed-expert LoRA targets.
    moe_lora_enabled: bool = False
    # False when ``moe_disable_finalize_fusion`` is set or any LoRA is
    # configured: both need an unfused FC2 so a seam is left for the LoRA GEMM.
    fused_finalize_enabled: bool = True
    # ``model_config.locality_domain_policy.enabled``. Whether the machine can
    # actually serve it is ``env.has_dep(MoEDep.LOCALITY_DOMAIN)``.
    locality_domain_requested: bool = False

    @property
    def smart_router(self) -> bool:
        """Mirrors ``MoE.smart_router`` (``interface.py``), its only definition."""
        return self.cluster_size > 1


# ---------------------------------------------------------------------------
# Selection results
# ---------------------------------------------------------------------------


class MoERejectReason(str, Enum):
    """Closed enum. Tests assert on these, never on log substrings.

    Closed because the reasons are an API: a test that wants "this request was
    turned down for the right cause" must be able to name the cause, and a
    free-form string cannot be named without pinning the wording too.
    """

    QUANT_UNSUPPORTED = "quant_unsupported"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    SM_UNSUPPORTED = "sm_unsupported"
    DEP_MISSING = "dep_missing"
    SHAPE_UNALIGNED = "shape_unaligned"
    SLOTS_NOT_DIVISIBLE_BY_EP = "slots_not_divisible_by_ep"
    TOPOLOGY_UNSUPPORTED = "topology_unsupported"
    LORA_UNSUPPORTED = "lora_unsupported"
    # Activation shape the impl cannot serve (today: swiglu_gptoss_style, i.e.
    # bias plus custom swiglu alpha/beta/limit).
    ACTIVATION_UNSUPPORTED = "activation_unsupported"
    # The routing method produces scores in a form the impl's kernel cannot
    # consume. Distinct from TOPOLOGY_UNSUPPORTED: nothing about the parallel
    # layout is wrong, the impl just fuses one routing shape and no other.
    ROUTING_UNSUPPORTED = "routing_unsupported"
    # EPLB is registered for this layer and the impl cannot lay out slots for
    # it. Distinct from TOPOLOGY_UNSUPPORTED: the parallel sizes are fine.
    EPLB_UNSUPPORTED = "eplb_unsupported"
    # The impl only has a fused-finalize FC2 epilogue, and the caller disabled
    # finalize fusion (explicitly, or implicitly by configuring LoRA).
    FINALIZE_FUSION_REQUIRED = "finalize_fusion_required"
    # Not a capability verdict: the impl could run, but the resolver refuses to
    # route production traffic there. Kept separate so that "we chose not to"
    # never reads as "it cannot".
    PATH_NOT_ENABLED = "path_not_enabled"
    # The named backend no longer exists (today: WIDEEP).
    BACKEND_DEPRECATED = "backend_deprecated"
    # A wrapper or aggregate that is never itself an execution unit.
    NOT_AN_IMPL = "not_an_impl"


@dataclass(frozen=True)
class MoEEligibility:
    """Return value of ``can_implement(problem, deployment)``.

    It carries the verdict and, on rejection, a closed-enum reason. It holds no
    execution parameters and no identity -- those live in :class:`MoERunContext`
    and, after the leaf-class migration, ``MoEImplDescriptor``.
    """

    eligible: bool
    reject_reason: Optional[MoERejectReason] = None
    detail: str = ""  # human-facing only

    def __post_init__(self) -> None:
        # Kills the "silent False with no reason" anti-pattern at type level.
        if self.eligible and self.reject_reason is not None:
            raise ValueError("eligible=True must not carry a reject_reason")
        if not self.eligible and self.reject_reason is None:
            raise ValueError("a rejection must name a MoERejectReason")

    def __bool__(self) -> bool:
        return self.eligible

    @classmethod
    def ok(cls) -> "MoEEligibility":
        return cls(eligible=True)

    @classmethod
    def no(cls, reason: MoERejectReason, detail: str) -> "MoEEligibility":
        """Reject with a machine-readable cause and a human-readable detail.

        ``detail`` is required rather than optional: a reason code narrows the
        cause to a category, and the operator still needs the specific value
        that failed the gate.
        """
        return cls(eligible=False, reject_reason=reason, detail=detail)


def nvfp4_fc1_row_alignment_rejection(
    p: "MoEProblem", d: "MoEDeployment"
) -> Optional[MoEEligibility]:
    """NVFP4 block scales swizzle in 128x4 tiles, so a gated FC1 buffer rounded
    up to that tile splits gate/up at the wrong row. Returns the rejection when
    this shard would be rounded up, else None (unknown shapes abstain).
    """
    from tensorrt_llm._torch.utils import is_gated_activation
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    if p.quant_algo != QuantAlgo.NVFP4 or p.intermediate_size is None:
        return None
    # Non-gated FC1 is one block with no gate/up split, so the rows the loader
    # adds stay a zero tail that the kernel never reads as an operand.
    if not is_gated_activation(p.activation_type):
        return None
    tp_size = max(d.tp_size, 1)
    if p.intermediate_size % tp_size != 0:
        # Uneven shards are a different concern; do not answer for them.
        return None
    fc1_rows_full = p.intermediate_size * 2
    fc1_rows = fc1_rows_full // tp_size
    if fc1_rows % 128 == 0:
        return None
    if fc1_rows_full % 128 == 0:
        hint = (
            f"moe_tp_size must divide {fc1_rows_full // 128}; raise "
            f"moe_expert_parallel_size to shrink moe_tp_size, which "
            f"non-ULYSSES CP multiplies by cp_size"
        )
    else:
        hint = "no moe_tp_size satisfies this for this intermediate_size"
    return MoEEligibility.no(
        MoERejectReason.SHAPE_UNALIGNED,
        f"NVFP4 MoE requires gated FC1 rows "
        f"(2 * intermediate_size_per_partition) to be a multiple of 128, but "
        f"moe_tp_size={tp_size} gives {fc1_rows}. {hint}.",
    )


@dataclass(frozen=True)
class MoERejection:
    """One candidate that did not win, and why."""

    legacy_backend: str
    reason: MoERejectReason
    detail: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "legacy_backend": self.legacy_backend,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MoEResolutionReport:
    """Selected implementation, eligible alternatives, and rejection reasons."""

    problem: MoEProblem
    deployment: MoEDeployment
    winner: Optional[str]  # legacy backend class name; None => hard failure
    # Selection mode: pinned, heuristic fallback, or failed.
    selected_by: str
    rejected: Tuple[MoERejection, ...] = ()
    # Eligible candidates in priority order; eligible[0] is the winner.
    eligible: Tuple[str, ...] = ()
    requested: Optional[str] = None  # backend literal, as written
    env_fingerprint: str = ""

    @property
    def alternatives(self) -> Tuple[str, ...]:
        """Eligible impls that lost to the winner on priority alone.

        Not rejections: nothing is wrong with these, they simply ranked lower.
        Keeping the two lists apart is the point -- "could not run" and "ran
        second" call for opposite responses from whoever reads the report.
        """
        return self.eligible[1:]

    @property
    def degraded(self) -> bool:
        """Whether the caller got something other than what it asked for."""
        return self.selected_by == "heuristic"

    @property
    def degraded_from(self) -> Optional[MoERejection]:
        """The rejection that caused the substitution, if there was one."""
        if not self.degraded or not self.rejected:
            return None
        # The last family rejection best explains the fallback.
        return self.rejected[-1]

    def to_dict(self) -> Dict[str, object]:
        """Serializable form. Field names are part of the artifact format."""
        return {
            "winner": self.winner,
            "requested": self.requested,
            "selected_by": self.selected_by,
            "env_fingerprint": self.env_fingerprint,
            "eligible": list(self.eligible),
            "rejected": [rejection.to_dict() for rejection in self.rejected],
            "problem": {
                "quant": self.problem.quant,
                "dtype_act": str(self.problem.dtype_act),
                "hidden_size": self.problem.hidden_size,
                "intermediate_size": self.problem.intermediate_size,
                "num_experts": self.problem.num_experts,
                "top_k": self.problem.top_k,
                "swiglu_gptoss_style": self.problem.swiglu_gptoss_style,
                "bias": self.problem.bias,
                "activation": self.problem.activation,
                "activation_constants": sorted(self.problem.activation_constants),
                "routing": self.problem.routing,
            },
            "deployment": {
                "ep_size": self.deployment.ep_size,
                "tp_size": self.deployment.tp_size,
                "parallel_size": self.deployment.parallel_size,
                "cluster_size": self.deployment.cluster_size,
                "use_dp": self.deployment.use_dp,
                "num_slots": self.deployment.num_slots,
                "eplb_enabled": self.deployment.eplb_enabled,
                "moe_lora_enabled": self.deployment.moe_lora_enabled,
                "sm": self.deployment.env.sm,
                "env_flags": dict(self.deployment.env.env_flags),
            },
        }

    def describe(self) -> str:
        """One line for the log. Reads as a sentence, not as a dict dump."""
        winner = "none" if self.winner is None else self.winner
        head = f"MoE resolution: {winner} (via {self.selected_by}"
        if self.requested is not None:
            head += f", requested {self.requested}"
        head += f", env {self.env_fingerprint})"
        if self.alternatives:
            # Named in the same line as the winner, because this is the list an
            # operator retries one by one when the default is not fast enough.
            head += f"; also eligible: {', '.join(self.alternatives)}"
        if not self.rejected:
            return head
        turned_down = ", ".join(
            f"{rejection.legacy_backend}={rejection.reason.value}" for rejection in self.rejected
        )
        return f"{head}; turned down: {turned_down}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoECommPlan:
    """What the comm layer decided for THIS forward. Facts, not capabilities.

    Single producer, so the value ``run_moe`` sees and the value the following
    ``comm.combine()`` acts on cannot drift apart. That producer is currently
    ``ExternalCommMoEScheduler._build_comm_plan`` rather than the comm strategy
    itself; moving it onto the strategy needs ``combine()`` to read
    ``payload_in_workspace`` off the plan instead of off the strategy, which
    changes the dispatch and combine signatures and stays with TRTLLM-14972.
    """

    input_sf_swizzled: bool  # what quantize_input actually produced
    enable_alltoall: bool
    moe_output: Optional[torch.Tensor]  # workspace-backed buffer, or None
    # No impl reads this one yet: ``combine()`` still takes the value off the
    # strategy attribute, which the producer assigns from this same field so the
    # two cannot disagree. It is here because the plan is where the decision is
    # made; TRTLLM-14972 makes ``combine()`` read it from here and drops the
    # attribute.
    payload_in_workspace: bool


@dataclass(frozen=True)
class MoERunContext:
    """Everything ``run_moe`` needs that the CALLER genuinely produces.

    ``workspace``, ``tuner_num_tokens`` and ``tuner_top_k`` are excluded on
    purpose: they are the impl's own resources, produced inside the impl rather
    than handed to it.
    """

    # produced by routing. Expert IDs [num_tokens, top_k], or expert slots of
    # the same shape when EPLB is enabled. None when the impl routes internally
    # from ``router_logits`` instead.
    token_selected_experts: Optional[torch.Tensor]
    token_final_scales: Optional[torch.Tensor]  # routing weights [num_tokens, top_k]
    # produced by quantize_input
    x: torch.Tensor  # activations [num_tokens, hidden_size]
    x_sf: Optional[torch.Tensor]  # scale factors, when the input is quantized
    # produced by the outer forward
    output_dtype: Optional[torch.dtype] = None
    do_finalize: bool = True
    lora_params: Optional[dict] = None
    router_logits: Optional[torch.Tensor] = None
    # scheduling info the impl may need for tuner-visible shapes
    all_rank_num_tokens: Optional[List[int]] = None
    # produced by the comm strategy, one object per forward
    comm_plan: Optional[MoECommPlan] = None


def require_comm_plan(impl: object, ctx: MoERunContext) -> MoECommPlan:
    """Return the required external-communication plan for this forward."""
    assert ctx.comm_plan is not None, (
        f"{type(impl).__name__}.run_moe needs ctx.comm_plan, and the scheduler "
        "that drives it always supplies one. A missing plan means run_moe was "
        "called without going through ExternalCommMoEScheduler."
    )
    return ctx.comm_plan


@dataclass(frozen=True, kw_only=True)
class MoEEplbBinding:
    """EPLB expert layout required before weight creation.

    Keyword-only on purpose: the leading fields are all ``int``, so positional
    construction would let a field inserted in the middle silently shift every
    later argument, surfacing only as wrong expert-weight shapes.
    """

    layer_idx: int
    # The checkpoint expert count, not a slot-layout value, but needed here for
    # the same reason: ``register_all_parameter_slot_and_to_fix_weight_fns``
    # iterates over it to set up host-tensor sharing before weights exist.
    num_experts: int
    num_slots: int
    slot_start: int
    slot_end: int
    expert_size_per_partition: int
    initial_local_expert_ids: Tuple[int, ...]
    initial_global_assignments: Tuple[int, ...]
    layer_load_balancer: Optional["SingleLayerMoeLoadBalancer"] = None

    @property
    def eplb_enabled(self) -> bool:
        return self.layer_load_balancer is not None
