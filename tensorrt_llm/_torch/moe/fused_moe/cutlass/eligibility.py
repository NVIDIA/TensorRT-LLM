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
"""The checks each Cutlass-family ``can_implement`` composes.

A leaf's support is what it declares -- ``sm_support``, ``supported_dtypes``,
the gpt-oss and OSS-build flags and ``capabilities`` -- plus the gates its own
``can_implement`` adds.
"""

from dataclasses import dataclass

import torch

from ....utils import ActivationType
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    check_quant_matches_identity,
)
from ..interface import _reject


@dataclass(frozen=True)
class SmSupport:
    """SM versions a leaf's kernel is built and validated for.

    ``minimum`` is the oldest architecture (Ampere by default, the oldest any
    kernel in this family is built for). ``allowed``, when set, enumerates the
    architectures whose cubins exist, which is the common case on the quantized
    paths; leave it unset for a kernel compiled for every later architecture.
    """

    minimum: int = 80
    allowed: frozenset[int] | None = None

    def accepts(self, sm: int) -> bool:
        return sm >= self.minimum and (self.allowed is None or sm in self.allowed)

    def describe(self) -> str:
        if self.allowed is None:
            return f"SM >= {self.minimum}"
        if len(self.allowed) == 1:
            return f"SM{next(iter(self.allowed))} only"
        return "/".join(f"SM{v}" for v in sorted(self.allowed))


def check_sm(cls: type, d: MoEDeployment) -> MoEEligibility | None:
    """Reject an SM version this leaf's kernel was not built for."""
    sm = d.env.sm
    if not cls.sm_support.accepts(sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} supports {cls.sm_support.describe()}, got SM{sm}",
        )
    return None


def check_dtype(cls: type, p: MoEProblem) -> MoEEligibility | None:
    """Reject an activation dtype this leaf's kernel has no instantiation for."""
    if p.dtype_act not in cls.supported_dtypes:
        supported = ", ".join(str(dtype) for dtype in sorted(cls.supported_dtypes, key=str))
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} requires {supported}, got {p.dtype_act}",
        )
    return None


def check_gptoss_activation(cls: type, p: MoEProblem) -> MoEEligibility | None:
    """Reject a gpt-oss / MiniMax SwiGLU shape this leaf's loader cannot take.

    ``supports_gptoss_style`` is about the weight method, not the kernel: the
    unquantized and MXFP4-family methods can load a 1-D gpt-oss expert bias,
    and MXFP8 inherits the generic ``FusedMoEMethodBase`` path that does the
    same. NVFP4 is a partial case -- it serves MiniMax-style SwigluBias, but
    its weight pad asserts 2-D, so a real expert bias is out.
    """
    if not p.swiglu_gptoss_style:
        return None
    if not cls.supports_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} cannot load a gpt-oss bias for quant={cls.descriptor.identity.quant}",
        )
    if cls.rejects_gptoss_expert_bias and p.bias is True:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} cannot load a 1-D gpt-oss expert bias "
            f"(weight-pad assert is 2-D); MiniMax-style SwigluBias without "
            f"bias is eligible",
        )
    return None


def check_moe_lora(cls: type, d: MoEDeployment) -> MoEEligibility | None:
    """Reject routed-expert MoE LoRA on a leaf whose format the op cannot fuse.

    The resolver already drops candidates whose ``capabilities`` decline LoRA
    before ``can_implement`` runs, so this only fires for a direct call -- unit
    tests and microbenchmarks reach a leaf without going through resolution.
    """
    if d.moe_lora_enabled and not cls.capabilities.supports_moe_lora:
        return _reject(
            MoERejectReason.LORA_UNSUPPORTED,
            f"{cls.__name__} does not fuse routed-expert MoE LoRA; only "
            f"unquantized fp16/bf16 and per-tensor FP8 (qdq) do",
        )
    return None


_OSS_BUILD_REQUIRED = "requires a build with USING_OSS_CUTLASS_MOE_GEMM=ON"


def check_oss_cutlass_build(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
    """Reject what only the open-source CUTLASS MoE kernels implement.

    A build with ``USING_OSS_CUTLASS_MOE_GEMM=OFF`` runs the op on the bundled
    internal kernel archive, which has no SiTu or Relu2, no
    post-SiLU clamp and no routed-expert LoRA. A leaf whose format, or whose
    dynamic FC2 scale, needs the open-source kernels declares it through
    ``requires_oss_cutlass_build`` / ``uses_dynamic_fc2_scale``; a leaf that
    never issues the op (``issues_fused_moe_op``) is not bound by it.
    """
    if d.env.oss_cutlass_moe or not cls.issues_fused_moe_op:
        return None
    if p.activation_type in (ActivationType.SiTu, ActivationType.Relu2):
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} {p.activation} {_OSS_BUILD_REQUIRED}",
        )
    if cls.requires_oss_cutlass_build:
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"{cls.__name__} {p.quant} {_OSS_BUILD_REQUIRED}",
        )
    if cls.uses_dynamic_fc2_scale and d.force_dynamic_quantization:
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"{cls.__name__} dynamic FC2 scaling {_OSS_BUILD_REQUIRED}",
        )
    if p.clamp_after_silu:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} post-SiLU clamping {_OSS_BUILD_REQUIRED}",
        )
    if d.moe_lora_enabled:
        return _reject(
            MoERejectReason.LORA_UNSUPPORTED,
            f"{cls.__name__} routed-expert LoRA {_OSS_BUILD_REQUIRED}",
        )
    return None


def check_cutlass_leaf(
    cls: type, p: MoEProblem, d: MoEDeployment, *leaf_gates: MoEEligibility | None
) -> MoEEligibility:
    """Return the first rejection of the family gates and ``leaf_gates``, else admit.

    Identity runs first, so a foreign format is never reported as an SM or
    dtype miss; the build gate runs last, so a build limit is never blamed for a
    problem the leaf turns down anyway. ``leaf_gates`` are pre-evaluated, so
    each must be pure and abstain (``None``) on formats it does not police.
    """
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_sm(cls, d),
        check_dtype(cls, p),
        check_gptoss_activation(cls, p),
        check_moe_lora(cls, d),
        *leaf_gates,
        check_oss_cutlass_build(cls, p, d),
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()


# The activation dtype sets the leaves declare as ``supported_dtypes``.
HP_DTYPES = frozenset({torch.float16, torch.bfloat16})
HP_DTYPES_WITH_FP32 = frozenset({torch.float16, torch.bfloat16, torch.float32})
HP_DTYPES_WITH_FP8 = frozenset({torch.float16, torch.bfloat16, torch.float8_e4m3fn})
BF16_ONLY = frozenset({torch.bfloat16})
