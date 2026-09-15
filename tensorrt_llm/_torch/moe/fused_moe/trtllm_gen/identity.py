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

TRTLLM_GEN_INPUT_REQUIREMENT = MoEInputRequirement(
    # The kernels read bf16 scales, and DeepEP dispatch must mark unfilled rows
    # before they arrive.
    routing_scales_dtype=torch.bfloat16,
    requires_sanitized_expert_ids=True,
    # Combine reduces in bf16 whatever the model's output dtype, so the NVLink
    # one-sided payload buffer has to match.
    onesided_workspace_dtype=torch.bfloat16,
)


class TrtllmProviderTraits:
    """The native TRT-LLM cubins, reached through ``TRTLLMOpBackend``."""

    provider = PROVIDER_TRTLLM
    use_flashinfer = False


class FlashinferProviderTraits:
    """The same algorithm as shipped in the FlashInfer wheel."""

    provider = PROVIDER_FLASHINFER
    use_flashinfer = True


# The two trait classes carry values only, for attributes
# ``TrtllmGenFusedMoEBase`` declares and leaves unset. A leaf must list its
# traits first so they win the MRO over the family base's defaults.


def trtllm_gen_descriptor(provider: str, quant: str, doc: str) -> MoEImplDescriptor:
    """Build one leaf's descriptor; only provider and quant ever differ."""
    return MoEImplDescriptor(
        identity=MoEImplId(provider, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=TRTLLM_GEN_CAPABILITIES,
        input_requirement=TRTLLM_GEN_INPUT_REQUIREMENT,
        doc=doc,
    )
