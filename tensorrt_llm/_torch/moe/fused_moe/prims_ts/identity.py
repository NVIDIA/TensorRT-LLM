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
"""PrimsTS's values of the identity types ``..impl_identity`` defines.

Every leaf imports this module, so nothing that varies by quantization format
belongs here.
"""

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId, MoEImplQuery
from ..interface import MoESchedulerKind

# FlashInfer is the provider: the kernels are vendored into this repo
# (``tensorrt_llm/_torch/moe/kernels/prims_ts``) but their lineage is
# FlashInfer's, as with Marlin.
PROVIDER_FLASHINFER = "flashinfer"

# CuTe DSL primitives with task scheduling (PrimsTS) is the technology, and the
# segment that tells these leaves from FlashInfer's other CuTe DSL kernels.
TECHNIQUE_PRIMS_TS = "prims_ts"
# FC1 with its fused activation and FC2, as two batched GEMMs over the
# experts, launched over TRT-LLM's routing tables.
KERNEL_BATCHED_GEMM = "batched_gemm"

#: Every PrimsTS identity, open in ``quant`` only.
PRIMS_TS_FAMILY = MoEImplQuery(
    provider=PROVIDER_FLASHINFER, technique=TECHNIQUE_PRIMS_TS, kernel_name=KERNEL_BATCHED_GEMM
)

# Conservative defaults: EPLB, expert bias and MoE LoRA are not validated.
PRIMS_TS_CAPABILITIES = MoEStaticCapability()

# ``moe_sort`` and the finalize take the routing weights as bfloat16.
PRIMS_TS_INPUT_REQUIREMENT = MoEInputRequirement(routing_scales_dtype=torch.bfloat16)


def prims_ts_identity(quant: str) -> MoEImplId:
    return MoEImplId(PROVIDER_FLASHINFER, TECHNIQUE_PRIMS_TS, KERNEL_BATCHED_GEMM, quant)


def prims_ts_descriptor(quant: str, doc: str) -> MoEImplDescriptor:
    """Build one leaf's descriptor; only quant ever differs."""
    return MoEImplDescriptor(
        identity=prims_ts_identity(quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=PRIMS_TS_CAPABILITIES,
        input_requirement=PRIMS_TS_INPUT_REQUIREMENT,
        doc=doc,
    )
