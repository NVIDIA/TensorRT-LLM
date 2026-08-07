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
"""Declaration, selection and execution contracts for MoE implementations.

Every type here is a frozen, GPU-free dataclass: it can be constructed and
asserted on a machine with no device, which is what lets selection be unit
tested and lets offline tuning enumerate candidates ahead of deployment.

The types split along three axes, and putting a field on the wrong one is the
mistake this file is shaped to prevent:

- What an impl *can do* regardless of input      -> MoEStaticCapability
- What an impl *demands of its caller*           -> MoEInputRequirement
- Whether an impl fits *one concrete question*   -> MoEEligibility

A capability is a class-level declaration. An eligibility is a verdict about a
single (problem, deployment) pair. An input requirement is neither: it never
disqualifies an impl, it just tells the scheduler what to prepare.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .impl_identity import MoEImplId
    from .moe_load_balancer import SingleLayerMoeLoadBalancer

# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEStaticCapability:
    """What this impl CAN do, independent of any particular input.

    Feeds SELECTION. Every field defaults to the conservative answer, so an
    impl that forgets to declare gets excluded rather than silently accepted.

    Two rules keep this class from absorbing everything: a condition that
    depends on the actual problem shape belongs in ``can_implement``, and a
    condition the caller can simply satisfy by preparing its input belongs in
    :class:`MoEInputRequirement`.
    """

    # Legacy gate: ``moe.backend.__class__ == CutlassFusedMoE`` in MoEScheduler.
    supports_moe_lora: bool = False
    # Legacy gate: the CuteDslFusedMoE isinstance check in
    # ``ConfigurableMoE._should_enable_dwdp``.
    supports_dwdp: bool = False


@dataclass(frozen=True)
class MoEInputRequirement:
    """What this impl REQUIRES THE CALLER to hand it.

    Read by MoEScheduler while assembling :class:`MoERunContext`, and by the
    comm strategy while building :class:`MoECommPlan`. Deliberately NOT part of
    selection: an unmet input requirement is the scheduler's job to satisfy,
    never a reason to pick a different impl.
    """

    # Legacy: the ``token_final_scales`` bfloat16 / float32 casts in
    # MoEScheduler.
    routing_scales_dtype: Optional[torch.dtype] = None
    # Legacy: the DeepGemmFusedMoE isinstance branches in MoEScheduler.
    requires_run_moe_workspace: bool = False
    # Legacy: the DeepEP-comm plus TRTLLMGenFusedMoE special case in
    # MoEScheduler.
    requires_sanitized_expert_ids: bool = False
    requires_router_logits: bool = False
    # Legacy: the bfloat16 combine workspace picked in
    # ``MoEScheduler._get_nvlink_onesided_moe_output``.
    onesided_workspace_dtype: Optional[torch.dtype] = None

    # There is deliberately no sentinel field here. Every Communication sets
    # ``invalid_token_expert_id = -1`` in its own __init__, and TRTLLM-Gen
    # kernels accept nothing else, so the value is a comm-side invariant rather
    # than something an impl needs the caller to supply.


# ---------------------------------------------------------------------------
# Selection inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEProblem:
    """The part of the question that is REUSABLE ACROSS DEPLOYMENTS.

    Normative rule for adding a field here: if changing the field must
    invalidate an already-persisted tuning result, it belongs in MoEProblem; if
    it only changes WHICH impls are eligible, it belongs in
    :class:`MoEDeployment`. This is also the tuning key, so field order and
    types are part of the on-disk format.

    ``quant`` / ``dtype_act`` / ``swiglu_gptoss_style`` are exactly today's
    three arguments of today's ``MoE.can_implement``. The shape fields are
    new: the tuning winner depends on them, yet today they are only checked
    later, inside ``__init__`` / ``validate``.
    """

    quant: Optional[str]  # canonical quant name, or None for bf16
    dtype_act: torch.dtype  # activation dtype BEFORE quantization
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    swiglu_gptoss_style: bool = False


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
        """Provenance stamp for a tuning result.

        Records WHICH machine state produced a given winner, so that replaying
        under a different environment is detectable instead of silently
        selecting someone else.
        """
        payload = repr((self.sm, sorted(self.available_deps), self.env_flags))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MoEDeployment:
    """Topology and slot layout.

    Changing these changes WHICH impls are eligible, but never invalidates a
    tuning result.
    """

    ep_size: int
    tp_size: int
    use_dp: bool
    num_slots: int
    env: MoEEnvironment

    @property
    def parallel_size(self) -> int:
        # Matches today's ``self.use_dp and self.parallel_size > 1`` test in
        # ``TRTLLMGenFusedMoE._supports_load_balancer``.
        return self.ep_size * self.tp_size


# ---------------------------------------------------------------------------
# Selection results
# ---------------------------------------------------------------------------


class MoERejectReason(str, Enum):
    """Closed enum. Tests assert on these, never on log substrings."""

    QUANT_UNSUPPORTED = "quant_unsupported"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    SM_UNSUPPORTED = "sm_unsupported"
    DEP_MISSING = "dep_missing"
    SHAPE_UNALIGNED = "shape_unaligned"
    SLOTS_NOT_DIVISIBLE_BY_EP = "slots_not_divisible_by_ep"
    TOPOLOGY_UNSUPPORTED = "topology_unsupported"
    LORA_UNSUPPORTED = "lora_unsupported"


@dataclass(frozen=True)
class MoEEligibility:
    """Return value of ``can_implement(problem, deployment)``.

    It carries the verdict and, on rejection, a closed-enum reason. It holds no
    execution parameters and no identity -- those live in :class:`MoERunContext`
    and ``MoEImplId``.
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


@dataclass(frozen=True)
class MoEResolutionReport:
    """Structured answer to "who got picked, and why was everyone else out".

    Replaces today's read-the-logs workflow. Two consumers: offline tuning
    (this IS the candidate set) and tests (assert on ``reject_reason``).
    """

    problem: MoEProblem
    deployment: MoEDeployment
    winner: Optional["MoEImplId"]  # None => hard failure
    rejected: Tuple[Tuple["MoEImplId", MoERejectReason], ...] = ()
    selected_by: str = "auto"  # "auto" | "pin"
    env_fingerprint: str = ""

    @property
    def eligible(self) -> Tuple["MoEImplId", ...]:
        return (self.winner,) if self.winner is not None else ()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoECommPlan:
    """What the comm layer decided for THIS forward. Facts, not capabilities.

    Single producer: the comm strategy builds it at the end of dispatch, and
    both ``run_moe`` and the following ``comm.combine()`` read the same object.
    """

    input_sf_swizzled: bool  # what quantize_input actually produced
    enable_alltoall: bool
    moe_output: Optional[torch.Tensor]  # workspace-backed buffer, or None
    payload_in_workspace: bool  # combine() reads the same field


@dataclass(frozen=True)
class MoERunContext:
    """Everything ``run_moe`` needs that the CALLER genuinely produces.

    ``workspace``, ``tuner_num_tokens`` and ``tuner_top_k`` are excluded on
    purpose: they are the impl's own resources, produced inside the impl rather
    than handed to it.
    """

    # produced by routing
    token_selected_experts: torch.Tensor
    token_final_scales: Optional[torch.Tensor]
    # produced by quantize_input
    x: torch.Tensor
    x_sf: Optional[torch.Tensor]
    # produced by the outer forward
    output_dtype: Optional[torch.dtype] = None
    do_finalize: bool = True
    lora_params: Optional[dict] = None
    router_logits: Optional[torch.Tensor] = None
    # scheduling info the impl may need for tuner-visible shapes
    all_rank_num_tokens: Optional[List[int]] = None
    # produced by the comm strategy, one object per forward
    comm_plan: Optional[MoECommPlan] = None


@dataclass(frozen=True)
class MoEEplbBinding:
    """Everything an impl needs to lay out and load its expert weights.

    Computed once by whoever owns the load-balancer registration, then passed as
    an explicit constructor argument -- never ``setattr``'d after construction.
    That is the entire point: weight shapes depend on these values, so they must
    be known BEFORE ``create_weights()``, not patched in afterwards.

    Excludes ``repeat_idx`` / ``repeat_count`` on purpose: those are
    forward-time scheduling state owned by the wrapper.
    """

    layer_idx: int
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
