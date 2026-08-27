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
"""
locality domain Execution Planning: centralized enable/disable decisions.

All scattered checks (CuteDSL availability, locality domain hardware support, effective
quantization, supported weight modes, partition alignment, etc.) are
consolidated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from tensorrt_llm._torch.locality_domain.layout import (
    PartitionedTensorLayout,
    make_bf16_linear_output_layout,
    make_nvfp4_linear_output_layout,
)


@dataclass(frozen=True)
class LocalityDomainPolicy:
    """Top-level locality domain configuration, typically stored in ModelConfig."""

    enabled: bool = False
    num_partitions: int = 2
    allowed_ops: frozenset = field(
        default_factory=lambda: frozenset(
            {
                "nvfp4_linear",
                "bf16_linear",
                "bf16_bmm",
                "nvfp4_moe",
                "bf16_moe",
            }
        )
    )
    allowed_backends: tuple = ("cutlass", "cublaslt", "cuda_core")

    def __post_init__(self):
        # Runtime (streams, mempools, locality_domain_device) only supports 2 partitions.
        if self.num_partitions != 2:
            raise ValueError(
                f"locality domain only supports num_partitions=2, got {self.num_partitions}. "
                f"Runtime resources (streams, mempools, TPC masks) are "
                f"hardcoded for exactly 2 partitions."
            )


@dataclass(frozen=True)
class PartitionPlan:
    """Base class for partition decisions."""

    enabled: bool
    num_partitions: int = 2
    backend: str = "cutedsl"
    merge_kind: Literal["concat", "scatter_add", "none"] = "concat"
    reason_if_disabled: Optional[str] = None


@dataclass(frozen=True)
class LinearPartitionPlan(PartitionPlan):
    """Partition plan specific to Linear layers."""

    partition_axis: int = 0  # partition along output dimension
    layout: Optional[PartitionedTensorLayout] = None
    op_kind: Optional[Literal["nvfp4_linear", "bf16_linear"]] = None


# Singleton disabled plan
DISABLED_PLAN = LinearPartitionPlan(
    enabled=False,
    reason_if_disabled="locality domain not enabled or not applicable",
)


class LocalityDomainExecutionPlanner:
    """Produces PartitionPlans based on hardware capability and module config.

    Centralizes all enable/disable logic that was previously scattered in:
    - Linear.maybe_create_locality_domain_sub_modules() (linear.py:2704-2708)
    - attention.py: is_locality_domain_supported() backend patching
    - fused_moe_cute_dsl.py: CuteDslGroupGemmMLP.__init__
    """

    def __init__(self, policy: LocalityDomainPolicy):
        self.policy = policy

    def plan_linear(
        self,
        in_features: int,
        out_features: int,
        quant_config,
        weight_mode,
        *,
        dtype: Optional[torch.dtype] = None,
        use_cute_dsl_bf16_gemm: bool = False,
        enable_locality_domain_bf16_linear: bool = False,
    ) -> LinearPartitionPlan:
        """Decide whether to partition a Linear layer for locality domain execution.

        Returns LinearPartitionPlan with enabled=True only if ALL conditions are met.
        The planner owns the backend decision (always cutedsl for locality domain) — callers
        do NOT need to pre-add 'cutedsl' to their allowed_backends list.
        """
        from tensorrt_llm._torch.cute_dsl_utils import (
            IS_CUTLASS_DSL_AVAILABLE,
            IS_CUTLASS_DSL_RUBIN_AVAILABLE,
        )
        from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
        from tensorrt_llm._torch.modules.linear import WeightMode
        from tensorrt_llm.quantization.mode import QuantAlgo

        # Policy check
        if not self.policy.enabled:
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="LocalityDomainPolicy is disabled"
            )

        # Hardware support
        if not is_locality_domain_enabled():
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="locality domain not supported on this hardware"
            )

        # Select the kernel family from the final effective weight
        # quantization. KV-cache-only quantization does not quantize Linear
        # weights and is therefore eligible for the BF16 path.
        layer_quant_mode = getattr(quant_config, "layer_quant_mode", None)
        is_nvfp4 = layer_quant_mode is not None and layer_quant_mode.has_nvfp4()
        if is_nvfp4 and getattr(quant_config, "quant_algo", None) == QuantAlgo.NVFP4_ARC:
            return LinearPartitionPlan(
                enabled=False,
                reason_if_disabled=(
                    "NVFP4_ARC scale geometry is not supported by locality domain Linear"
                ),
            )
        has_weight_quant = layer_quant_mode is not None and layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        )
        is_bf16 = (
            not has_weight_quant
            and dtype == torch.bfloat16
            and use_cute_dsl_bf16_gemm
            and enable_locality_domain_bf16_linear
        )
        if is_nvfp4:
            op_kind = "nvfp4_linear"
        elif is_bf16:
            op_kind = "bf16_linear"
        else:
            return LinearPartitionPlan(
                enabled=False,
                reason_if_disabled=(
                    "Linear is neither NVFP4 nor an explicitly enabled CuTeDSL BF16 locality domain operation"
                ),
            )

        if op_kind not in self.policy.allowed_ops:
            return LinearPartitionPlan(
                enabled=False,
                reason_if_disabled=f"{op_kind} not in allowed_ops",
            )

        # locality domain uses the Rubin CuTeDSL kernels.
        if not IS_CUTLASS_DSL_AVAILABLE:
            return LinearPartitionPlan(enabled=False, reason_if_disabled="CuteDSL not available")
        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            return LinearPartitionPlan(
                enabled=False, reason_if_disabled="CuTeDSL Rubin kernels not available"
            )

        supported_weight_modes = (
            WeightMode.VANILLA,
            WeightMode.FUSED_GATE_UP_LINEAR,
        )
        if weight_mode not in supported_weight_modes:
            supported_names = ", ".join(mode.name for mode in supported_weight_modes)
            return LinearPartitionPlan(
                enabled=False,
                reason_if_disabled=(
                    f"Weight mode {weight_mode} not supported for {op_kind}; "
                    f"supported modes: {supported_names}"
                ),
            )

        if op_kind == "nvfp4_linear":
            # NVFP4 packs two K elements per byte, while the transformed
            # weight requires a 16-byte-aligned packed K dimension. Reject
            # shapes that would pad only the weight: the shared activation is
            # not K-padded and the locality domain composite requires exact packed-K
            # equality.
            if in_features % 32 != 0:
                return LinearPartitionPlan(
                    enabled=False,
                    reason_if_disabled=(
                        f"in_features={in_features} not divisible by NVFP4 "
                        "packed-K alignment (32 elements)"
                    ),
                )
            layout = make_nvfp4_linear_output_layout(
                out_features,
                in_features,
                self.policy.num_partitions,
            )
        else:
            if in_features % 8 != 0:
                return LinearPartitionPlan(
                    enabled=False,
                    reason_if_disabled=(
                        f"in_features={in_features} not divisible by BF16 "
                        "16-byte alignment (8 elements)"
                    ),
                )
            layout = make_bf16_linear_output_layout(
                out_features,
                in_features,
                self.policy.num_partitions,
            )
        layout_reason = layout.disabled_reason_for_padding_free_split()
        if layout_reason is not None:
            return LinearPartitionPlan(enabled=False, reason_if_disabled=layout_reason)

        return LinearPartitionPlan(
            enabled=True,
            num_partitions=self.policy.num_partitions,
            backend="cutedsl",
            merge_kind="concat",
            partition_axis=0,
            layout=layout,
            op_kind=op_kind,
        )

    def plan_moe(
        self,
        quant_config,
        *,
        moe_backend: str = "CUTEDSL",
        use_fused_finalize: bool = True,
        dtype_activation: torch.dtype = torch.bfloat16,
    ) -> PartitionPlan:
        """Decide whether to partition a MoE GroupGemm for locality domain execution.

        MoE locality domain replicates weights on each partition (not partitioned like Linear).
        Each partition runs the full GroupGemm with inplace output into shared buffers.
        """
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
        from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled

        if not self.policy.enabled:
            return PartitionPlan(
                enabled=False, reason_if_disabled="LocalityDomainPolicy is disabled"
            )

        if not is_locality_domain_enabled():
            return PartitionPlan(
                enabled=False, reason_if_disabled="locality domain not supported on this hardware"
            )

        if moe_backend.upper() != "CUTEDSL":
            return PartitionPlan(
                enabled=False,
                reason_if_disabled=f"locality domain MoE requires CuteDSL backend, got {moe_backend}",
            )

        is_nvfp4 = (
            quant_config is not None
            and hasattr(quant_config, "quant_mode")
            and quant_config.quant_mode.has_nvfp4()
        )
        has_any_quant = (
            quant_config is not None
            and hasattr(quant_config, "quant_mode")
            and quant_config.quant_mode.has_any_quant()
        )
        is_bf16 = not has_any_quant and dtype_activation == torch.bfloat16

        if is_nvfp4:
            op_name = "nvfp4_moe"
        elif is_bf16:
            op_name = "bf16_moe"
        elif not has_any_quant:
            return PartitionPlan(
                enabled=False,
                reason_if_disabled=f"BF16 locality domain MoE requires bfloat16 activation, got {dtype_activation}",
            )
        else:
            return PartitionPlan(
                enabled=False, reason_if_disabled="locality domain MoE only supports NVFP4 or BF16"
            )

        if op_name not in self.policy.allowed_ops:
            return PartitionPlan(enabled=False, reason_if_disabled=f"{op_name} not in allowed_ops")

        if not use_fused_finalize:
            return PartitionPlan(
                enabled=False, reason_if_disabled="locality domain MoE requires fused finalize"
            )

        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            return PartitionPlan(
                enabled=False, reason_if_disabled="CuTeDSL Rubin kernels not available"
            )

        return PartitionPlan(
            enabled=True,
            num_partitions=self.policy.num_partitions,
            backend="cutedsl",
            merge_kind="none",
        )

    def plan_bf16_bmm(
        self,
        *,
        dtype: torch.dtype,
        use_cute_dsl_bf16_bmm: bool,
    ) -> PartitionPlan:
        """Decide whether Rubin BF16 BMM may use two locality domain partitions."""
        from tensorrt_llm._torch.cute_dsl_utils import (
            IS_CUTLASS_DSL_AVAILABLE,
            IS_CUTLASS_DSL_RUBIN_AVAILABLE,
        )
        from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled

        if not self.policy.enabled:
            return PartitionPlan(
                enabled=False, reason_if_disabled="LocalityDomainPolicy is disabled"
            )
        if not is_locality_domain_enabled():
            return PartitionPlan(
                enabled=False,
                reason_if_disabled="locality domain not supported on this hardware",
            )
        if "bf16_bmm" not in self.policy.allowed_ops:
            return PartitionPlan(enabled=False, reason_if_disabled="bf16_bmm not in allowed_ops")
        if not use_cute_dsl_bf16_bmm:
            return PartitionPlan(
                enabled=False,
                reason_if_disabled="CuTeDSL BF16 BMM is disabled",
            )
        if dtype != torch.bfloat16:
            return PartitionPlan(
                enabled=False,
                reason_if_disabled=f"BF16 locality domain BMM requires bfloat16, got {dtype}",
            )
        if not IS_CUTLASS_DSL_AVAILABLE:
            return PartitionPlan(enabled=False, reason_if_disabled="CuteDSL not available")
        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            return PartitionPlan(
                enabled=False,
                reason_if_disabled="CuTeDSL Rubin kernels not available",
            )
        return PartitionPlan(
            enabled=True,
            num_partitions=self.policy.num_partitions,
            backend="cutedsl",
            merge_kind="concat",
        )
