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
"""TRTLLM-Gen's values of the identity types ``..impl_identity`` defines.

Every leaf imports this module, so nothing that varies by quantization format
belongs here.
"""

import dataclasses

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId
from ..interface import MoESchedulerKind

# Also the op backend registry keys (``moe_op_backend.get_op_backend``).
PROVIDER_TRTLLM = "trtllm"
PROVIDER_FLASHINFER = "flashinfer"

TECHNIQUE_TRTLLM_GEN = "trtllm_gen"
KERNEL_FUSED_MOE = "fused_moe"

# Every leaf publishes these two, so they are declared family-wide rather
# than per leaf.
TRTLLM_GEN_CAPABILITIES = MoEStaticCapability(supports_expert_bias=True, supports_eplb=True)

# A leaf whose ``run_moe`` takes its expert window from a per-layer dynamic EPLB
# replica plan; the scheduler orders its kernel after the helper-weight copy.
TRTLLM_GEN_PER_LAYER_DYNAMIC_EPLB_CAPABILITIES = dataclasses.replace(
    TRTLLM_GEN_CAPABILITIES, supports_per_layer_dynamic_eplb=True
)

TRTLLM_GEN_INPUT_REQUIREMENT = MoEInputRequirement(
    # The kernels read bf16 scales, and DeepEP dispatch must mark unfilled rows
    # before they arrive.
    routing_scales_dtype=torch.bfloat16,
    requires_sanitized_expert_ids=True,
    # Combine reduces in bf16 whatever the model's output dtype, so the NVLink
    # one-sided payload buffer has to match.
    onesided_workspace_dtype=torch.bfloat16,
)


def trtllm_gen_descriptor(
    provider: str,
    quant: str,
    doc: str,
    capabilities: MoEStaticCapability = TRTLLM_GEN_CAPABILITIES,
) -> MoEImplDescriptor:
    """Build one leaf's descriptor; provider and quant differ, capabilities rarely."""
    return MoEImplDescriptor(
        identity=MoEImplId(provider, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=capabilities,
        input_requirement=TRTLLM_GEN_INPUT_REQUIREMENT,
        doc=doc,
    )
