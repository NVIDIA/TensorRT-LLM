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
"""The checks each CuteDSL grouped-GEMM ``can_implement`` composes."""

from ...cutlass.eligibility import check_dtype, check_sm
from ...impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    check_quant_matches_identity,
)
from ...impl_environment import MoEDep
from ...interface import _reject


def check_grouped_gemm_leaf(
    cls: type,
    p: MoEProblem,
    d: MoEDeployment,
    *leaf_gates: MoEEligibility | None,
) -> MoEEligibility:
    """Run the family gates in order, then the leaf's own, and admit.

    Identity, SM and dtype come first, so a wrong format, SM or dtype is
    reported as such rather than as a later gate's failure. ``leaf_gates`` are already-evaluated
    verdicts that abstain (``None``) on a format they do not police.
    """
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_sm(cls, d),
        check_dtype(cls, p),
        check_grouped_gemm_shared(cls, p, d),
        *leaf_gates,
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()


def check_grouped_gemm_shared(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
    """Gates both grouped-GEMM leaves share, or ``None`` to admit."""
    sm_version = d.env.sm

    if sm_version == 107 and not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
        return _reject(
            MoERejectReason.DEP_MISSING,
            f"{cls.__name__} on SM107 requires Rubin support in CuTe DSL",
        )

    if p.swiglu_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not support swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit)",
        )

    # Localized weight shards are built once from the loaded weights, so
    # they cannot follow EPLB expert migration.
    if d.locality_domain_requested and d.env.has_dep(MoEDep.LOCALITY_DOMAIN) and d.eplb_enabled:
        return _reject(
            MoERejectReason.EPLB_UNSUPPORTED,
            "locality domain MoE cannot follow EPLB expert migration",
        )

    # SM107 has no unfused FC2: NVFP4 has no plain grouped GEMM there, and
    # the BF16 op always fuses finalize.
    if sm_version == 107 and not d.fused_finalize_enabled:
        return _reject(
            MoERejectReason.FINALIZE_FUSION_REQUIRED,
            f"{cls.__name__} on SM107 only has a fused-finalize FC2",
        )

    return None
