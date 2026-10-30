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
"""CUTLASS-family identity segments, shared contracts and the descriptor factory.

Eleven leaves are ``trtllm.cutlass.grouped_gemm.<quant>``. Each ``fp8_block_scales``
leaf names the kernel it runs instead: ``deepgemm.cuda.hopper_grouped_gemm``
(:mod:`.fp8_block_scales_hopper`) and ``trtllm.triton.grouped_gemm``
(:mod:`.fp8_block_scales_triton`).
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
    supports_expert_bias: bool = True,
) -> MoEImplDescriptor:
    """Build one leaf's descriptor.

    Args:
        quant: The one quantization format the leaf implements.
        doc: One line naming the kernel.
        provider: Identity segment. The defaults name the CUTLASS grouped
            GEMM; only the two FP8 block-scale leaves, which run another
            kernel, override the segments.
        technique: Identity segment; see ``provider``.
        kernel_name: Identity segment; see ``provider``.
        supports_moe_lora: Whether ``torch.ops.trtllm.fused_moe`` fuses
            routed-expert LoRA for this format. The scheduler reads the flag
            before it reaches for the LoRA helpers, so only a leaf that mixes
            in ``CutlassMoELoraMixin`` may set it.
        supports_expert_bias: Whether the leaf's kernel applies an expert FC
            bias.
    """
    return MoEImplDescriptor(
        identity=MoEImplId(provider, technique, kernel_name, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=dataclasses.replace(
            CUTLASS_CAPABILITIES,
            supports_moe_lora=supports_moe_lora,
            supports_expert_bias=supports_expert_bias,
        ),
        input_requirement=CUTLASS_INPUT_REQUIREMENT,
        doc=doc,
    )
