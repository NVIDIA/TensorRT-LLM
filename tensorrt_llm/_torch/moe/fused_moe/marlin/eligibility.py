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
"""The eligibility gates the Marlin leaves compose into ``can_implement``.

Free functions rather than methods on :class:`.MarlinFusedMoEBase`: the family
base must not implement ``can_implement``, or both leaves would inherit an
answer that belongs to no single identity.

Each gate reads only ``cls``, ``MoEProblem`` and ``MoEDeployment``. No
``get_sm_version()``, no ``os.environ``, no import probe, so an offline tuner
on a GPU-less host gets the same verdict a serving process does.
"""

import torch

from tensorrt_llm._torch.utils import is_nvfp4_marlin_supported_sm

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    check_quant_matches_identity,
)
from ..interface import _reject


def check_marlin_capabilities(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
    """Capability gates both Marlin leaves share, or ``None`` to admit."""
    # ``d.env.sm`` passed explicitly: the helper probes the live device when
    # called with no argument.
    if not is_nvfp4_marlin_supported_sm(d.env.sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} only supports SM89-SM99 (Ada/Hopper), got SM{d.env.sm}",
        )

    # The kernel dequantizes FP4 weights into BF16 registers and issues a BF16
    # MMA, so there is no path for another activation dtype.
    if p.dtype_act != torch.bfloat16:
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} W4A16 requires bfloat16 activations, got {p.dtype_act}",
        )

    if p.swiglu_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not support swiglu_gptoss_style",
        )

    # Sorted-token dispatch has no EPLB slot layout, so a layer that registered
    # a load balancer cannot run here.
    if d.eplb_enabled:
        return _reject(
            MoERejectReason.EPLB_UNSUPPORTED,
            f"{cls.__name__} has no EPLB slot layout",
        )

    return None


def check_marlin_leaf(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
    """Compose one leaf's verdict: identity first, then the shared capabilities."""
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_marlin_capabilities(cls, p, d),
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()
