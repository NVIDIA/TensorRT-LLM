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
"""Marlin's values of the identity types ``..impl_identity`` defines.

Both leaves import this module, so nothing that varies by quantization format
belongs here.
"""

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId
from ..interface import MoESchedulerKind

# Marlin is the provider and not the technique: the kernel is vendored into
# this repo but its lineage is upstream's, which the device code still declares
# (``MARLIN_NAMESPACE_NAME marlin_moe_wna16`` in
# cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_moe_gemm.cu).
PROVIDER_MARLIN = "marlin"

TECHNIQUE_CUDA = "cuda"
# One launch covers scatter, both GEMMs and the routing-weight multiply, so the
# kernel name is the family one rather than a GEMM name.
KERNEL_FUSED_MOE = "fused_moe"

# ``supports_eplb`` stays at its conservative default: sorted-token dispatch
# has no slot layout.
MARLIN_CAPABILITIES = MoEStaticCapability(supports_apply_router_weight_on_input=True)

MARLIN_INPUT_REQUIREMENT = MoEInputRequirement(routing_scales_dtype=torch.float32)


def marlin_descriptor(quant: str, doc: str) -> MoEImplDescriptor:
    """Build one leaf's descriptor; only quant ever differs."""
    return MoEImplDescriptor(
        identity=MoEImplId(PROVIDER_MARLIN, TECHNIQUE_CUDA, KERNEL_FUSED_MOE, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MARLIN_CAPABILITIES,
        input_requirement=MARLIN_INPUT_REQUIREMENT,
        doc=doc,
    )
