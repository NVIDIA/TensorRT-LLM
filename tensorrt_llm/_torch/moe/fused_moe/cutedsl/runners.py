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
"""Outer autotune runners for the CuteDSL MoE kernels.

The NVFP4 runner serves the grouped-GEMM leaf and, with a narrower tile set, the
fused-FC12 one.
"""

from collections.abc import Callable
from typing import Any

import torch

from tensorrt_llm._utils import get_sm_version

from ....autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ....custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_RUBIN_AVAILABLE
from ....locality_domain.autotune import (
    LocalityDomainConcurrentTunableRunner as _LocalityDomainConcurrentTunableRunner,
)
from ....utils import get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2

# cute_dsl_custom_ops defines these runners only when IS_CUTLASS_DSL_AVAILABLE;
# without the DSL nothing is tuned, so the tuple stays empty.
_TILE_SIZE_CHECKED_RUNNERS: tuple[type, ...] = ()
if IS_CUTLASS_DSL_AVAILABLE:
    from ....custom_ops.cute_dsl_custom_ops import (
        Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmRunner,
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
    )

    _TILE_SIZE_CHECKED_RUNNERS = (
        Sm100BlockScaledContiguousGroupedGemmRunner,
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
        Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
    )


def _unwrap_locality_domain_runner(runner: TunableRunner) -> TunableRunner:
    """Return the kernel runner wrapped by the shared locality domain tuning adapter."""
    if isinstance(runner, _LocalityDomainConcurrentTunableRunner):
        return runner.op_runner
    return runner


def _runner_tactics_match_tile_size(
    comb: list[tuple[TunableRunner, Any]],
    outer_runner_type: type,
    checked_runner_types: tuple[type, ...],
) -> bool:
    """Check that nested kernel tactics use the outer MoE tile size."""
    tile_size = None
    for runner, tactic in comb:
        runner = _unwrap_locality_domain_runner(runner)
        if isinstance(runner, outer_runner_type):
            tile_size = tactic
    if tile_size is None:
        return True

    for runner, tactic in comb:
        runner = _unwrap_locality_domain_runner(runner)
        if isinstance(runner, checked_runner_types):
            mma_tiler_mn, *_ = tactic
            if mma_tiler_mn[0] != tile_size:
                return False
    return True


class CuteDslFusedMoENvfp4InputsHelper(GroupedGemmInputsHelper):
    def __init__(
        self, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: list[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True
        )

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert, start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts)
        )
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device,
        )
        new_token_selected_experts = (
            new_token_selected_experts.view(self.top_k, num_tokens).transpose(0, 1).contiguous()
        )
        return x, new_token_selected_experts, *others


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    tuning_config_cache = dict()

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        enable_finalize_fusion: bool = True,
        enable_alltoall: bool = False,
        output_dtype: torch.dtype = torch.bfloat16,
        scaling_vector_size: int = 16,
        use_direct_expert_metadata: bool = False,
        use_locality_domain: bool = False,
        workload_identity: tuple | None = None,
        prime_inner_tactics: bool = False,
    ) -> None:
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_finalize_fusion = enable_finalize_fusion
        self.enable_alltoall = enable_alltoall
        self.use_direct_expert_metadata = use_direct_expert_metadata
        self.use_locality_domain = use_locality_domain

        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size
        # Extra terms that distinguish two otherwise identically-shaped
        # workloads (locality-domain topology, activation type, ...). Cache key
        # only -- it must not imply anything about ``forward_impl``'s signature.
        self.workload_identity = workload_identity
        # Locality domain only: ``forward_impl`` nests its own AutoTuner calls
        # and accepts ``overlap_moe_output_memset``. Single-op backends must
        # leave this off.
        self.prime_inner_tactics = prime_inner_tactics

    def unique_id(self) -> tuple:
        identity = (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.enable_finalize_fusion,
            self.enable_alltoall,
            self.output_dtype,
            self.scaling_vector_size,
        )
        if self.workload_identity is not None:
            identity += (self.workload_identity,)
        if self.use_direct_expert_metadata:
            identity += ("direct_expert_metadata",)
        return identity

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[int]:
        return self._tile_sizes()

    @staticmethod
    def _tile_sizes() -> list[int]:
        # tile_size=512 is only supported on Rubin (SM107).
        if get_sm_version() == 107:
            return [128, 256, 512]
        return [128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            if self.use_direct_expert_metadata:
                # One tuning profile is shared by the 128- and 256-row
                # tactics. Use the minimum tile size so its capacity buckets
                # cover the union needed by both tactics; each kernel still
                # rounds capacity to its selected tile size.
                helper = GroupedGemmInputsHelper(
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    128,
                )
                tuning_config = TuningConfig(
                    dynamic_tensor_specs=(
                        DynamicTensorSpec(
                            0,
                            0,
                            helper.gen_expert_capacity_tuning_buckets,
                            helper.map_to_expert_capacity_tuning_bucket,
                        ),
                    ),
                    constraint_specs=(
                        ConstraintSpec(1, 0, helper.infer_shape_input0_rows),
                        ConstraintSpec(2, 0, helper.infer_shape_input0_rows),
                        ConstraintSpec(3, 0, helper.infer_shape_input0_rows),
                        ConstraintSpec(4, 0, helper.infer_shape_input0_rows),
                    ),
                    use_cold_l2_cache=True,
                )
            else:
                helper = CuteDslFusedMoENvfp4InputsHelper(
                    self.num_experts, self.top_k, self.num_local_experts, self.local_expert_offset
                )
                tuning_config = TuningConfig(
                    dynamic_tensor_specs=(
                        DynamicTensorSpec(
                            0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                        ),
                    ),
                    constraint_specs=(
                        ConstraintSpec(1, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(3, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(4, 0, helper.infer_shape_num_tokens),
                    ),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            self.__class__.tuning_config_cache[key] = tuning_config
        return self.__class__.tuning_config_cache[key]

    def forward(
        self, inputs: list[torch.Tensor], tactic: int | None, do_preparation: bool = False
    ) -> torch.Tensor:
        if do_preparation:
            if self.prime_inner_tactics:
                # Inner FC tuning cannot run from inside the CUDA graph used to
                # profile an outer tile. Prime every tile's FC1/FC2 cache for
                # this optimization profile before outer profiling starts.
                for tile_size in self._tile_sizes():
                    self.forward_impl(
                        *inputs,
                        enable_alltoall=self.enable_alltoall,
                        tile_size=tile_size,
                        overlap_moe_output_memset=False,
                    )
            return inputs[4]

        if isinstance(tactic, int) and tactic > 0:
            tile_size = tactic
        else:
            tile_size = 128
        if not self.use_direct_expert_metadata:
            return self.forward_impl(
                *inputs, enable_alltoall=self.enable_alltoall, tile_size=tile_size
            )

        recv_expert_count = inputs[-1]
        forward_inputs = inputs[:-1]
        num_rows = forward_inputs[0].size(0)
        if num_rows % self.num_local_experts != 0:
            raise ValueError(
                "Expert-major input rows must be divisible by the number of local experts"
            )
        deep_ep_expert_capacity = num_rows // self.num_local_experts
        # The locality-domain impl does not take the count-native arguments.
        count_native_kwargs = (
            {}
            if self.use_locality_domain
            else dict(
                recv_expert_count=recv_expert_count,
                deep_ep_expert_capacity=deep_ep_expert_capacity,
                use_count_native_expert_metadata=True,
            )
        )
        return self.forward_impl(
            *forward_inputs,
            enable_alltoall=self.enable_alltoall,
            tile_size=tile_size,
            **count_native_kwargs,
        )

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(comb: list[tuple[TunableRunner, Any]]) -> bool:
        checked_runner_types = list(_TILE_SIZE_CHECKED_RUNNERS)
        if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            from ....custom_ops.cute_dsl_custom_ops import (
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner,
            )

            checked_runner_types.extend(
                [
                    Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                    Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                    # The fused FC12 runner also requires mma_tiler_m == tile_size;
                    # without it the Cartesian replay pairs inner tactics captured
                    # under one routing tile with a different outer tile.
                    Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner,
                ]
            )

        return _runner_tactics_match_tile_size(
            comb,
            CuteDslFusedMoENvfp4Runner,
            tuple(checked_runner_types),
        )


class CuteDslFusedMoEBF16InputsHelper(GroupedGemmInputsHelper):
    """Helper for CuteDSL BF16 MoE input preprocessing and autotuning."""

    def __init__(
        self, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: list[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True
        )

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert, start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts)
        )
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device,
        )
        new_token_selected_experts = (
            new_token_selected_experts.view(self.top_k, num_tokens).transpose(0, 1).contiguous()
        )
        return x, new_token_selected_experts, *others


class CuteDslFusedMoEBF16Runner(TunableRunner):
    """Autotuner runner for BF16 MoE on Rubin (SM107).

    Selects tile_size from {64, 128, 256} and delegates to run_moe_bf16_impl.
    """

    tuning_config_cache = dict()

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        enable_alltoall: bool = False,
        output_dtype: torch.dtype = torch.bfloat16,
        workload_identity: tuple | None = None,
        prime_inner_tactics: bool = False,
    ) -> None:
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_alltoall = enable_alltoall
        self.output_dtype = output_dtype
        # See CuteDslFusedMoENvfp4Runner: cache key only, and locality-domain
        # only.
        self.workload_identity = workload_identity
        self.prime_inner_tactics = prime_inner_tactics

    def unique_id(self) -> tuple:
        identity = (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.enable_alltoall,
            self.output_dtype,
        )
        if self.workload_identity is not None:
            identity += (self.workload_identity,)
        return identity

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[int]:
        return self._tile_sizes()

    @staticmethod
    def _tile_sizes() -> list[int]:
        return [64, 128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            helper = CuteDslFusedMoEBF16InputsHelper(
                self.num_experts, self.top_k, self.num_local_experts, self.local_expert_offset
            )
            # BF16 inputs: [x, token_selected_experts, token_final_scales,
            #               moe_output]
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(1, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(3, 0, helper.infer_shape_num_tokens),
                ),
                inputs_pre_hook=helper.inputs_pre_hook,
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(
        self, inputs: list[torch.Tensor], tactic: int | None, do_preparation: bool = False
    ) -> torch.Tensor:
        if do_preparation:
            if self.prime_inner_tactics:
                # See the NVFP4 runner: nested FC tuning must complete before
                # the outer tile is profiled under CUDA graph capture.
                for tile_size in self._tile_sizes():
                    self.forward_impl(
                        *inputs,
                        enable_alltoall=self.enable_alltoall,
                        tile_size=tile_size,
                        overlap_moe_output_memset=False,
                    )
            return inputs[3]

        if isinstance(tactic, int) and tactic > 0:
            tile_size = tactic
        else:
            tile_size = 128
        return self.forward_impl(*inputs, enable_alltoall=self.enable_alltoall, tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(comb: list[tuple[TunableRunner, Any]]) -> bool:
        # BF16 GEMM runners that need CTA_M == tile_size.
        checked_runner_types = []
        if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            from ....custom_ops.cute_dsl_custom_ops import (
                Sm107ContiguousGatherGroupedGemmSwigluFusionRunner,
                Sm107ContiguousGroupedGemmFinalizeFusionRunner,
            )

            checked_runner_types.extend(
                [
                    Sm107ContiguousGatherGroupedGemmSwigluFusionRunner,
                    Sm107ContiguousGroupedGemmFinalizeFusionRunner,
                ]
            )

        return _runner_tactics_match_tile_size(
            comb,
            CuteDslFusedMoEBF16Runner,
            tuple(checked_runner_types),
        )
