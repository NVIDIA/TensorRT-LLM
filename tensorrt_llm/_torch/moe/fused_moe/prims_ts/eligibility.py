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
"""The eligibility gates the PrimsTS leaves compose into ``can_implement``.

Each gate reads only ``cls``, ``MoEProblem`` and ``MoEDeployment``: no
``get_sm_version()``, no ``os.environ``, no import probe.
"""

from ..cutlass.eligibility import check_dtype, check_sm
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    check_activation_support,
    check_quant_matches_identity,
)
from ..impl_environment import MoEDep
from ..interface import _reject
from ..trtllm_gen.eligibility import check_intermediate_shard_alignment, check_no_expert_bias


def check_prims_ts_shared(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
    """Gates every PrimsTS leaf shares, or ``None`` to admit."""
    if not d.env.has_dep(MoEDep.PRIMS_TS):
        return _reject(
            MoERejectReason.DEP_MISSING,
            f"{cls.__name__} requires the CuTe DSL primitives and task-scheduling modules",
        )
    if d.env.sm == 107 and not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
        return _reject(
            MoERejectReason.DEP_MISSING,
            f"{cls.__name__} on SM107 requires Rubin support in CuTe DSL",
        )
    if d.eplb_enabled:
        return _reject(
            MoERejectReason.EPLB_UNSUPPORTED,
            f"{cls.__name__} has not been validated with EPLB",
        )
    if d.moe_lora_enabled:
        return _reject(MoERejectReason.LORA_UNSUPPORTED, f"{cls.__name__} has no MoE LoRA path")
    if d.smart_router:
        return _reject(
            MoERejectReason.TOPOLOGY_UNSUPPORTED,
            f"{cls.__name__} has no smart-router path (moe_cluster_size={d.cluster_size})",
        )
    if p.swiglu_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not support swiglu_gptoss_style",
        )
    # The FC1 input and the FC2 output must keep the layer's hidden width: the
    # finalize writes FC2's rows straight into the layer output.
    if p.hidden_size is not None and p.hidden_size % cls.hidden_size_alignment != 0:
        return _reject(
            MoERejectReason.SHAPE_UNALIGNED,
            f"{cls.__name__} requires hidden_size ({p.hidden_size}) to be a multiple of "
            f"{cls.hidden_size_alignment}",
        )
    return check_intermediate_shard_alignment(
        cls, p, d, weight_alignment=cls.intermediate_shard_alignment
    )


def check_prims_ts_leaf(
    cls: type, p: MoEProblem, d: MoEDeployment, *leaf_gates: MoEEligibility | None
) -> MoEEligibility:
    """Compose one leaf's verdict: identity, SM and dtype, the shared gates, then its own."""
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_sm(cls, d),
        check_dtype(cls, p),
        check_prims_ts_shared(cls, p, d),
        check_no_expert_bias(cls, p),
        check_activation_support(cls, p),
        *leaf_gates,
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()
