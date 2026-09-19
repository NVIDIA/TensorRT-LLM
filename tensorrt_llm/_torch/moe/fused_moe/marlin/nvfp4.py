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
"""``marlin.cuda.fused_moe.nvfp4``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import MarlinFusedMoEBase
from .eligibility import check_marlin_leaf
from .identity import marlin_descriptor


@register_moe_impl
class MarlinCudaNvfp4Impl(MarlinFusedMoEBase):
    """``marlin.cuda.fused_moe.nvfp4``.

    The W4A4-labelled checkpoint, run W4A16: Marlin dequantizes in registers
    and never quantizes the activations, which is what lets an NVFP4
    checkpoint run on architectures that have no NVFP4 tensor cores.

    Executes identically to the ``w4a16_nvfp4`` sibling -- see
    :mod:`.base` for why one implementation carries two identities.
    """

    descriptor = marlin_descriptor(
        "nvfp4",
        "Marlin fused W4A16 MoE GEMM over NVFP4 weights, SM89-SM99 (Ada/Hopper).",
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_marlin_leaf(cls, p, d)
