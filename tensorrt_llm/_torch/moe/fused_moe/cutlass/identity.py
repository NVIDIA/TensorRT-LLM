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
"""Cutlass's values of the identity types ``..impl_identity`` defines.

Three ``(provider, technique, kernel_name)`` triples, because not every leaf in
this family runs a CUTLASS kernel:

* ``trtllm.cutlass.grouped_gemm`` -- the CUTLASS grouped GEMM behind
  ``torch.ops.trtllm.fused_moe``.
* ``deepgemm.cuda.hopper_grouped_gemm`` -- FP8 block scales on SM90: the same
  op, but ``CutlassMoeFCRunner`` hands FC1/FC2 to TRT-LLM's in-tree port of
  DeepGEMM (``cpp/include/tensorrt_llm/deep_gemm``). Only ``kernel_name``
  separates its identity from the SM100/103/107 leaf in
  ``..fused_moe_deepgemm``, which calls the ``tensorrt_llm.deep_gemm`` Python
  package.
* ``trtllm.triton.grouped_gemm`` -- FP8 block scales on SM120, an internal
  Triton grouped GEMM that never enters C++.
"""

import dataclasses

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId
from ..interface import MoESchedulerKind

PROVIDER_TRTLLM = "trtllm"
PROVIDER_DEEPGEMM = "deepgemm"

TECHNIQUE_CUTLASS = "cutlass"
TECHNIQUE_CUDA = "cuda"
TECHNIQUE_TRITON = "triton"

KERNEL_GROUPED_GEMM = "grouped_gemm"
KERNEL_HOPPER_GROUPED_GEMM = "hopper_grouped_gemm"

CUTLASS_CAPABILITIES = MoEStaticCapability(
    supports_expert_bias=True,
    supports_eplb=True,
    supports_apply_router_weight_on_input=True,
)

CUTLASS_INPUT_REQUIREMENT = MoEInputRequirement(routing_scales_dtype=torch.float32)


def cutlass_descriptor(
    quant: str,
    doc: str,
    *,
    provider: str = PROVIDER_TRTLLM,
    technique: str = TECHNIQUE_CUTLASS,
    kernel_name: str = KERNEL_GROUPED_GEMM,
    supports_moe_lora: bool = False,
) -> MoEImplDescriptor:
    """Build one leaf's descriptor.

    Args:
        quant: The one quantization format the leaf implements.
        doc: One line naming the kernel and the SMs it runs on.
        provider: Identity segment. The defaults name the CUTLASS grouped
            GEMM; only the two FP8 block-scale leaves, which run another
            kernel, override the segments.
        technique: Identity segment; see ``provider``.
        kernel_name: Identity segment; see ``provider``.
        supports_moe_lora: Whether ``torch.ops.trtllm.fused_moe`` fuses
            routed-expert LoRA for this format. The scheduler reads the flag
            before it reaches for the LoRA helpers, so only a leaf that mixes
            in ``CutlassMoELoraMixin`` may set it.
    """
    return MoEImplDescriptor(
        identity=MoEImplId(provider, technique, kernel_name, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=dataclasses.replace(CUTLASS_CAPABILITIES, supports_moe_lora=supports_moe_lora),
        input_requirement=CUTLASS_INPUT_REQUIREMENT,
        doc=doc,
    )
