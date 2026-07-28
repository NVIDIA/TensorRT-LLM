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

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                          OptimizationProfile, TunableRunner, TuningConfig)
from ...custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from ...cute_dsl_utils import (IS_CUTLASS_DSL_AVAILABLE,
                               IS_CUTLASS_DSL_RUBIN_AVAILABLE)
from ...locality_domain.autotune import \
    LocalityDomainConcurrentTunableRunner as \
    _LocalityDomainConcurrentTunableRunner
from ...locality_domain.policy import LocalityDomainExecutionPlanner
from ...locality_domain.runtime import LocalityDomainRuntime
from ...locality_domain_utils import (_copy_to_new_cuda_allocation,
                                      get_reserved_remainder_stream)
from ...model_config import ModelConfig
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor,
                      get_last_power_of_2_num_tokens_buckets,
                      last_positive_power_of_2)
from .activation import (DEFAULT_MOE_ACTIVATION, ActivationParamShape,
                         MoEActivation, MoEActivationSupport)
from .impl_base import MoEImplBase, apply_moe_impl_construction_state
from .impl_contract import (MoEDeployment, MoEEligibility, MoEInputRequirement,
                            MoEProblem, MoERejectReason, MoERunContext,
                            MoEStaticCapability,
                            nvfp4_fc1_row_alignment_rejection,
                            require_comm_plan)
from .impl_environment import MoEDep
from .interface import _reject
from .quantization import (BF16CuteDslFusedMoEMethod, MoEWeightLoadingMode,
                           NVFP4CuteDslFusedMoEMethod)
from .routing import BaseMoeRoutingMethod

# These runners are defined inside cute_dsl_custom_ops' ``if
# IS_CUTLASS_DSL_AVAILABLE:`` block, which has no else-branch, so importing them
# unconditionally would break every importer of this file -- and create_moe
# imports it eagerly under _torch.models, so that reaches all model startup
# rather than just this backend. Guard the same way the sibling custom_ops
# modules do, and leave the tuple empty when the DSL is absent: no CuteDSL
# runner can be tuned in that case, so nothing can match it.
_TILE_SIZE_CHECKED_RUNNERS: Tuple[type, ...] = ()
if IS_CUTLASS_DSL_AVAILABLE:
    from ...custom_ops.cute_dsl_custom_ops import (
        Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmRunner,
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner)
    _TILE_SIZE_CHECKED_RUNNERS = (
        Sm100BlockScaledContiguousGroupedGemmRunner,
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
        Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner)

_DISABLE_DIRECT_DEEP_EP_METADATA_ENV = "TRTLLM_DISABLE_CUTEDSL_DEEP_EP_DIRECT_METADATA"


def _unwrap_locality_domain_runner(runner: TunableRunner) -> TunableRunner:
    """Return the kernel runner wrapped by the shared locality domain tuning adapter."""
    if isinstance(runner, _LocalityDomainConcurrentTunableRunner):
        return runner.op_runner
    return runner


def _runner_tactics_match_tile_size(
    comb: List[Tuple[TunableRunner, Any]],
    outer_runner_type: type,
    checked_runner_types: Tuple[type, ...],
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


def _expert_count_tile_plan(
    expert_counts: List[int],
    capacity: int,
    tile_size: int,
) -> List[Tuple[int, int, int]]:
    """Return the reference tile plan for count-native scheduling.

    Each tuple is ``(expert_idx, permuted_mn_limit, expanded_row_start)`` for
    one MMA M tile. Source-contract tests use this helper to cover empty
    experts and boundary counts without requiring a GPU build.
    """
    if capacity <= 0 or tile_size <= 0:
        raise ValueError("capacity and tile_size must be positive")

    plan = []
    tile_idx = 0
    for expert_idx, raw_count in enumerate(expert_counts):
        count = min(max(raw_count, 0), capacity)
        for row_in_expert in range(0, count, tile_size):
            rows_in_tile = min(tile_size, count - row_in_expert)
            plan.append((
                expert_idx,
                tile_idx * tile_size + rows_in_tile,
                expert_idx * capacity + row_in_expert,
            ))
            tile_idx += 1
    return plan


@dataclass
class NvFp4WeightView:
    """Bundles all NVFP4 weight tensors for MoE computation.

    Under the VA-based DWDP pipeline ``param.data`` is swapped to a
    composite [num_experts, ...] tensor before the kernel call, so every
    field is a single tensor — the bundle is just a convenient grouping
    that lets the runner forward a single object instead of six.
    """
    w3_w1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc1_global_scale: torch.Tensor
    w2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    fc2_global_scale: torch.Tensor
    expert_size_per_partition: int
    slot_start: int


@torch.compile(options={"max-autotune": True})
def swiglu_fused_moe(x, swiglu_limit_scalar: float = float("inf")):
    x, gate = x.chunk(2, dim=-1)
    if swiglu_limit_scalar != float("inf"):
        gate = gate.clamp(max=swiglu_limit_scalar)
        x = x.clamp(min=-swiglu_limit_scalar, max=swiglu_limit_scalar)
    return F.silu(gate) * x


def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    offset_array: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]

    # Note: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)

    # Note: we have different output scale shape for fp8_quantize_1x128, so we need to handle it differently for sm100 and other archs.
    if is_sm_100f():
        input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                        (1, m, m * w_k))
    else:
        m_padded = (m + 3) // 4 * 4
        input_scale_tmp = a_sf[0:m_padded * w_k]
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
        input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1),
                                                     (1, m, m * w_k))

    weight_scale_tmp = b_sf.permute(1, 2, 0)

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    len_offset_array = offset_array.shape[0]
    for i in range(len_offset_array - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
        ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
                                                                0],
                                         updated_b[:, :, i])
    ref = ref.to(torch.bfloat16)
    return ref


def cute_dsl_nvfp4_grouped_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
):
    assert a.dtype == torch.float4_e2m1fn_x2
    assert a.dim() == 2
    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.dim() == 3
    assert a_sf.dtype == torch.uint8
    assert a_sf.dim() == 1
    assert b_sf.dtype == torch.uint8
    assert b_sf.dim() == 3
    assert alpha.dtype == torch.float32
    assert alpha.dim() == 1

    m, k = a.size(0), a.size(1) * 2
    l, n = b.size(0), b.size(1)
    scale_k = k // scaling_vector_size
    assert m % tile_size == 0
    assert k % (scaling_vector_size * 4) == 0
    assert b.size(2) * 2 == k
    assert a_sf.size(0) == m * scale_k
    assert b_sf.size(0) == l
    assert b_sf.size(1) == n
    assert b_sf.size(2) == scale_k
    assert alpha.size(0) == l

    num_tiles = m // tile_size
    assert tile_idx_to_group_idx.dtype == torch.int32
    assert tile_idx_to_group_idx.size() == (num_tiles, )
    assert num_non_exiting_tiles.dtype == torch.int32
    assert num_non_exiting_tiles.size() == (1, )

    num_tiles_per_expert = torch.bincount(
        tile_idx_to_group_idx[:num_non_exiting_tiles[0].item()], minlength=l)
    offsets = [0] + num_tiles_per_expert.cumsum(dim=0).tolist()

    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end <= start:
            continue
        a_sliced = a[start * tile_size:end * tile_size]
        a_sf_sliced = a_sf[start * tile_size * k // scaling_vector_size:end *
                           tile_size * k // scaling_vector_size]
        ref[start * tile_size:end * tile_size] = torch.ops.trtllm.nvfp4_gemm(
            a_sliced.view(torch.uint8), b[i].view(torch.uint8), a_sf_sliced,
            b_sf[i], alpha[i], output_dtype)

    return ref


class CuteDslFusedMoENvfp4InputsHelper(GroupedGemmInputsHelper):

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                 local_expert_offset: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert,
                                            start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts))
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device)
        new_token_selected_experts = new_token_selected_experts.view(
            self.top_k, num_tokens).transpose(0, 1).contiguous()
        return x, new_token_selected_experts, *others


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    tuning_config_cache = dict()

    def __init__(self,
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
                 use_count_native_expert_metadata: bool = False,
                 deep_ep_expert_capacity: Optional[int] = None,
                 workload_identity: Optional[Tuple] = None):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_finalize_fusion = enable_finalize_fusion
        self.enable_alltoall = enable_alltoall
        self.use_direct_expert_metadata = use_direct_expert_metadata
        self.use_count_native_expert_metadata = use_count_native_expert_metadata
        if self.use_count_native_expert_metadata and not self.use_direct_expert_metadata:
            raise ValueError(
                "Count-native expert metadata requires direct expert metadata")
        self.deep_ep_expert_capacity = deep_ep_expert_capacity

        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size
        self.workload_identity = workload_identity

    def unique_id(self):
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
            identity += (self.workload_identity, )
        if self.use_direct_expert_metadata:
            identity += (
                "direct_expert_metadata",
                self.deep_ep_expert_capacity,
            )
        if self.use_count_native_expert_metadata:
            identity += ("count_native_expert_metadata", )
        return identity

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        return self._tile_sizes()

    @staticmethod
    def _tile_sizes() -> List[int]:
        # tile_size=512 is only supported on Rubin (SM107).
        if get_sm_version() == 107:
            return [128, 256, 512]
        return [128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            if self.use_direct_expert_metadata:
                tuning_config = TuningConfig(use_cold_l2_cache=True)
            else:
                helper = CuteDslFusedMoENvfp4InputsHelper(
                    self.num_experts, self.top_k, self.num_local_experts,
                    self.local_expert_offset)
                tuning_config = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets,
                        last_positive_power_of_2), ),
                    constraint_specs=(ConstraintSpec(
                        1, 0, helper.infer_shape_num_tokens),
                                      ConstraintSpec(
                                          2, 0, helper.infer_shape_num_tokens),
                                      ConstraintSpec(
                                          3, 0, helper.infer_shape_num_tokens),
                                      ConstraintSpec(
                                          4, 0, helper.infer_shape_num_tokens)),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            self.__class__.tuning_config_cache[key] = tuning_config
        return self.__class__.tuning_config_cache[key]

    def forward(self,
                inputs: List[torch.Tensor],
                tactic: Optional[int],
                do_preparation: bool = False) -> torch.Tensor:
        if do_preparation:
            if self.workload_identity is not None:
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
        recv_expert_count = None
        forward_inputs = inputs
        if self.use_direct_expert_metadata:
            recv_expert_count = inputs[-1]
            forward_inputs = inputs[:-1]
        return self.forward_impl(
            *forward_inputs,
            enable_alltoall=self.enable_alltoall,
            tile_size=tile_size,
            recv_expert_count=recv_expert_count,
            deep_ep_expert_capacity=self.deep_ep_expert_capacity,
            use_count_native_expert_metadata=self.
            use_count_native_expert_metadata)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        checked_runner_types = list(_TILE_SIZE_CHECKED_RUNNERS)
        if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            from ...custom_ops.cute_dsl_custom_ops import (
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner)
            checked_runner_types.extend([
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
            ])

        return _runner_tactics_match_tile_size(
            comb,
            CuteDslFusedMoENvfp4Runner,
            tuple(checked_runner_types),
        )


class CuteDslFusedMoEBF16InputsHelper(GroupedGemmInputsHelper):
    """Helper for CuteDSL BF16 MoE input preprocessing and autotuning."""

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                 local_expert_offset: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert,
                                            start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts))
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device)
        new_token_selected_experts = new_token_selected_experts.view(
            self.top_k, num_tokens).transpose(0, 1).contiguous()
        return x, new_token_selected_experts, *others


class CuteDslFusedMoEBF16Runner(TunableRunner):
    """Autotuner runner for BF16/FP16 MoE on Rubin (SM107).

    Selects tile_size from {64, 128, 256} and delegates to run_moe_bf16_impl.
    """
    tuning_config_cache = dict()

    def __init__(self,
                 forward_impl: Callable,
                 num_experts: int,
                 top_k: int,
                 num_local_experts: int,
                 local_expert_offset: int,
                 enable_alltoall: bool = False,
                 output_dtype: torch.dtype = torch.bfloat16,
                 workload_identity: Optional[Tuple] = None):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_alltoall = enable_alltoall
        self.output_dtype = output_dtype
        self.workload_identity = workload_identity

    def unique_id(self):
        identity = (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.enable_alltoall,
            self.output_dtype,
        )
        if self.workload_identity is not None:
            identity += (self.workload_identity, )
        return identity

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        return self._tile_sizes()

    @staticmethod
    def _tile_sizes() -> List[int]:
        return [64, 128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            helper = CuteDslFusedMoEBF16InputsHelper(self.num_experts,
                                                     self.top_k,
                                                     self.num_local_experts,
                                                     self.local_expert_offset)
            # BF16 inputs: [x, token_selected_experts, token_final_scales,
            #               moe_output]
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(DynamicTensorSpec(
                    0, 0, get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2), ),
                constraint_specs=(
                    ConstraintSpec(1, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(3, 0, helper.infer_shape_num_tokens),
                ),
                inputs_pre_hook=helper.inputs_pre_hook,
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(self,
                inputs: List[torch.Tensor],
                tactic: Optional[int],
                do_preparation: bool = False) -> torch.Tensor:
        if do_preparation:
            if self.workload_identity is not None:
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
        return self.forward_impl(*inputs,
                                 enable_alltoall=self.enable_alltoall,
                                 tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        # BF16 GEMM runners that need CTA_M == tile_size.
        checked_runner_types = []
        if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            from ...custom_ops.cute_dsl_custom_ops import (
                Sm107ContiguousGatherGroupedGemmSwigluFusionRunner,
                Sm107ContiguousGroupedGemmFinalizeFusionRunner)
            checked_runner_types.extend([
                Sm107ContiguousGatherGroupedGemmSwigluFusionRunner,
                Sm107ContiguousGroupedGemmFinalizeFusionRunner,
            ])

        return _runner_tactics_match_tile_size(
            comb,
            CuteDslFusedMoEBF16Runner,
            tuple(checked_runner_types),
        )


class CuteDslFusedMoE(MoEImplBase):
    # CuteDSL dispatch/combine path exercises the ceil/floor partition
    # (NVLinkOneSided alltoall with kernel-level remainder handling), so this
    # backend is the only opt-in for non-divisible EP today.
    _supports_non_divisible_ep: bool = True
    """CuteDSL flow of fused mixture of experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    capabilities = MoEStaticCapability(
        supports_dwdp=True,
        supports_eplb=True,
        supports_apply_router_weight_on_input=True)

    input_requirement = MoEInputRequirement(routing_scales_dtype=torch.float32)

    # Kinds mirror the kernel's own SUPPORTED_ACTIVATION_TYPES in
    # cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py.
    # The clamp is a kernel-cache-key scalar and the epilogue has no
    # "clamp absent" branch, so an absent clamp is +inf, not None.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2}),
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    def _has_moe_output_memset_aux_stream(self) -> bool:
        event_dict = getattr(self, 'event_dict', None)
        aux_stream_dict = getattr(self, 'aux_stream_dict', None)
        return (event_dict is not None and aux_stream_dict is not None
                and EventType.Main in event_dict
                and EventType.MoeOutputMemset in event_dict
                and AuxStreamType.MoeOutputMemset in aux_stream_dict)

    def _get_reserved_moe_output_memset_stream(
            self) -> Optional[torch.cuda.Stream]:
        """Resolve and cache the strict split's remainder stream before capture."""
        if hasattr(self, "_cached_reserved_moe_output_memset_stream"):
            return self._cached_reserved_moe_output_memset_stream

        stream = None
        if self._locality_domain_runtime is not None:
            stream = get_reserved_remainder_stream()
        self._cached_reserved_moe_output_memset_stream = stream
        return stream

    def _moe_output_memset_run_stream(self) -> torch.cuda.Stream:
        """Select the remainder stream when present, otherwise the aux stream."""
        remainder_stream = self._get_reserved_moe_output_memset_stream()
        if remainder_stream is not None:
            return remainder_stream
        return self.aux_stream_dict[AuxStreamType.MoeOutputMemset]

    def _locality_domain_workload_identity(
            self, input_dtype: torch.dtype) -> Tuple[Any, ...]:
        """Describe the sharded MoE workload and its compute topology."""
        if self._locality_domain_runtime is None or self._locality_domain_weight_shards is None:
            raise RuntimeError(
                "locality domain workload identity requires initialized shards")
        shard_identity = tuple((
            tuple(shard['w3_w1_weight'].shape),
            str(shard['w3_w1_weight'].dtype),
            tuple(shard['w2_weight'].shape),
            str(shard['w2_weight'].dtype),
        ) for shard in self._locality_domain_weight_shards)
        return (
            self._locality_domain_plan.num_partitions,
            self._locality_domain_runtime.topology_identity(),
            str(input_dtype),
            shard_identity,
        )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """CuteDSL grouped GEMM: NVFP4 on SM100/SM103, bfloat16 activations."""
        sm_version = d.env.sm
        quant_algo = p.quant_algo

        # CuteDslFusedMoE requires at least SM90
        if sm_version < 90:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CuteDslFusedMoE requires SM >= 90, got SM{sm_version}")

        # Output is hardcoded to bfloat16, so input must also be bfloat16 to
        # maintain input/output dtype consistency.
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslFusedMoE only supports bfloat16 activation (output is hardcoded to bfloat16), "
                f"got {p.dtype_act}")

        # CuteDslFusedMoE does NOT support swiglu_gptoss_style
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslFusedMoE does not support swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit)"
            )

        # Localized weight shards are built once from the loaded weights, so
        # they cannot follow EPLB expert migration.
        if (d.locality_domain_requested
                and d.env.has_dep(MoEDep.LOCALITY_DOMAIN) and d.eplb_enabled):
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                "locality domain MoE cannot follow EPLB expert migration")

        # SM107 has no unfused FC2: NVFP4 has no plain grouped GEMM there, and
        # the BF16 op always fuses finalize.
        if sm_version == 107 and not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                "CuteDslFusedMoE on SM107 only has a fused-finalize FC2")

        if quant_algo is None:
            if sm_version != 107:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"Unquantized CuteDSL MoE requires SM107, got SM{sm_version}"
                )
            if not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    "Unquantized CuteDSL MoE on SM107 requires Rubin support in CuTe DSL"
                )
            # The BF16 FC1 op fuses SwiGLU by name and takes no activation
            # argument, unlike its NVFP4 counterpart.
            if p.activation != "Swiglu":
                return _reject(
                    MoERejectReason.ACTIVATION_UNSUPPORTED,
                    f"Unquantized CuteDSL MoE fuses SwiGLU only, got {p.activation}"
                )
            return MoEEligibility.ok()

        # NVFP4 - SM in {100, 103, 107}
        if quant_algo == QuantAlgo.NVFP4:
            if sm_version not in {100, 103, 107}:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"NVFP4 requires SM100, SM103, or SM107, got SM{sm_version}"
                )
            if sm_version == 107 and not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    "NVFP4 CuteDSL MoE on SM107 requires Rubin support in CuTe DSL"
                )
            # process_weights_after_loading() unswizzles the FC1 block scales,
            # which asserts 128-row tiles; without this gate an unaligned shard
            # dies mid weight load with a bare swizzle error.
            rejection = nvfp4_fc1_row_alignment_rejection(p, d)
            if rejection is not None:
                return rejection
            return MoEEligibility.ok()

        # FP8_BLOCK_SCALES lands here on purpose. ``run_moe_fp8_block_scales``
        # exists, but its GEMM is ``cute_dsl_fp8_group_blockwise_gemm_ref`` --
        # an fp32 einsum-per-expert reference, not a CuteDSL kernel -- so
        # claiming the algorithm here would advertise a reference path as a
        # backend. DeepGemm / TRTLLMGen own it on SM100/103, Cutlass on
        # SM90/SM120. See the FP8-block note in MOE_DEVELOPER_GUIDE.md.
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"CuteDslFusedMoE does not support quant_algo={quant_algo}")

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # The scheduler enables the full fast path only for compatible
        # DeepEPLowLatency NVFP4 post-quant dispatch.
        self.disable_deep_ep_direct_metadata = os.environ.get(
            _DISABLE_DIRECT_DEEP_EP_METADATA_ENV, "0") == "1"

        # Read by run_moe_nvfp4* to pick the fused-finalize epilogue, which
        # leaves no seam for a LoRA GEMM.
        self.use_fused_finalize = (not model_config.moe_disable_finalize_fusion
                                   and model_config.lora_config is None)

        # Output-memset overlap is independent of MoE chunking, so ensure its
        # stream and events exist even if the parent creates no chunking event.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[
                AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeOutputMemset]
        }

        self._weights_created = False

        self.scaling_vector_size = 16
        # locality domain: fork/join with _locality_domain kernel variants + shared output buffers.
        # Weight splitting happens in post_load_weights after normal loading.
        self._locality_domain_runtime = None
        self._locality_domain_weight_shards = None  # set in post_load_weights
        planner = LocalityDomainExecutionPlanner(
            model_config.locality_domain_policy)
        self._locality_domain_plan = planner.plan_moe(
            self.quant_config,
            moe_backend=model_config.moe_backend,
            use_fused_finalize=self.use_fused_finalize,
            dtype_activation=self.dtype,
            activation=ActivationType(self.activation_type).name,
        )
        if self._locality_domain_plan.enabled:
            self._locality_domain_runtime = LocalityDomainRuntime(
                self._locality_domain_plan.num_partitions)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        if self._weights_created:
            return
        super().create_weights()

    def _build_local_weight_view(self) -> NvFp4WeightView:
        """Build the weight view from this backend's per-layer weights."""
        return NvFp4WeightView(
            w3_w1_weight=self.w3_w1_weight,
            fc1_weight_scale=self.quant_scales.fc1_weight_block,
            fc1_global_scale=self.quant_scales.fc1_global,
            w2_weight=self.w2_weight,
            fc2_weight_scale=self.quant_scales.fc2_weight_block,
            fc2_global_scale=self.quant_scales.fc2_global,
            expert_size_per_partition=self.expert_size_per_partition,
            slot_start=self.slot_start,
        )

    @property
    def uses_locality_domain(self) -> bool:
        return self._locality_domain_plan.enabled

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
        elif get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            # Unquantized on SM107: the BF16 method interleaves FC1 weights for
            # the fused gather + grouped GEMM + SwiGLU kernel, which serves no
            # other activation.
            if self.activation_type != ActivationType.Swiglu:
                raise ValueError(
                    "Unquantized CuteDslFusedMoE fuses SwiGLU only, got "
                    f"{ActivationType(self.activation_type).name}")
            return BF16CuteDslFusedMoEMethod()
        # ``can_implement`` admits NVFP4, plus unquantized BF16 on SM107, so
        # selection never lands here. Raise rather than fall back: any other
        # method owns a weight layout these kernels cannot read.
        raise ValueError(
            f"CuteDslFusedMoE only supports NVFP4, got {self.quant_config}")

    def _supports_load_balancer(self) -> bool:
        return True

    def _check_configs(self):
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def supports_moe_output_in_alltoall_workspace(self):
        return self.has_nvfp4 or (not self.has_any_quant
                                  and get_sm_version() == 107)

    def quantize_input(self,
                       x: Union[torch.Tensor, Fp4QuantizedTensor],
                       post_quant_comm: bool = True):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path

        Returns: (x, x_sf) where x_sf is already reshaped to 2D if needed

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        if self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)
        elif self.has_deepseek_fp8_block_scales:
            # FP8 block scales doesn't support permutation of quantized inputs.
            # WAR: The quantization is in run_moe_fp8_block_scales.
            pass
        elif not self.has_any_quant:
            # Unquantized BF16/FP16: no quantization needed
            pass
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )

        if x_sf is not None:
            # ``view(0, -1)`` is ambiguous for an empty micro-batch. The
            # scale width is fixed by the logical hidden size, so spell it
            # out for both empty and non-empty inputs.
            scale_cols = (self.hidden_size + self.scaling_vector_size -
                          1) // self.scaling_vector_size
            x_sf = x_sf.view(x_row, scale_cols)
        return x, x_sf

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        weight_view: Optional[NvFp4WeightView] = None,
        recv_expert_count: Optional[torch.Tensor] = None,
        deep_ep_expert_capacity: Optional[int] = None,
        use_deep_ep_direct_metadata: bool = False,
    ) -> torch.Tensor:
        """NVFP4 MoE computation.

        Uses the single-tensor ``run_moe_nvfp4_impl`` path. (The former
        multi-B DWDP path was removed once DWDP switched to VA: VA swaps
        ``param.data`` to a full [num_experts, ...] tensor, so the single-
        tensor kernel is sufficient.)

        Args:
            weight_view: Bundled weight tensors. Must not be None.
            use_deep_ep_direct_metadata: Use adapter-free, count-native DeepEP
                metadata. The scheduler sets this only for the supported path.
        """
        assert self.has_nvfp4
        assert weight_view is not None
        if self.activation_type not in (ActivationType.Swiglu,
                                        ActivationType.Relu2):
            raise NotImplementedError(
                "CuteDSL NVFP4 FC1 supports only SwiGLU and Relu2; "
                f"got {self.activation_type.name}")
        output_dtype = torch.bfloat16

        use_locality_domain = self._locality_domain_runtime is not None
        if use_locality_domain:
            if self.activation_type != ActivationType.Swiglu:
                raise NotImplementedError(
                    "Rubin locality domain NVFP4 MoE currently supports SwiGLU only"
                )

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_final_scales.size(0),
                                         self.hidden_size)
            assert moe_output.dtype == output_dtype

        # Empty micro-batches are valid at the backend boundary. Avoid
        # entering autotuning because its synthetic grouped-GEMM inputs
        # require at least one output row.
        if token_selected_experts.size(0) == 0:
            return moe_output

        effective_top_k = token_selected_experts.size(-1)
        if (recv_expert_count is None) != (deep_ep_expert_capacity is None):
            raise ValueError(
                "recv_expert_count and deep_ep_expert_capacity must be provided together"
            )
        use_direct_expert_metadata = (
            use_deep_ep_direct_metadata
            and recv_expert_count is not None
            and not use_locality_domain
            and is_sm_100f())

        if use_locality_domain:
            forward_impl = self._run_moe_nvfp4_locality_domain
            workload_identity = self._locality_domain_workload_identity(x.dtype)
            tuner_key = (
                "CuteDslFusedMoE::run_moe_nvfp4::locality_domain_end_to_end")
            inputs = [
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                moe_output,
            ]
        else:
            forward_impl = self.run_moe_nvfp4_impl
            workload_identity = None
            tuner_key = "CuteDslFusedMoE::run_moe_nvfp4"
            inputs = [
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                moe_output,
                weight_view,
            ]

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=forward_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
            workload_identity=workload_identity,
            use_direct_expert_metadata=use_direct_expert_metadata,
            use_count_native_expert_metadata=use_direct_expert_metadata,
            deep_ep_expert_capacity=(deep_ep_expert_capacity
                                     if use_direct_expert_metadata else None),
        )

        if use_direct_expert_metadata:
            inputs.append(recv_expert_count)
        _, best_tactic = tuner.choose_one(
            tuner_key,
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        recv_expert_count: Optional[torch.Tensor] = None,
        deep_ep_expert_capacity: Optional[int] = None,
        use_count_native_expert_metadata: bool = False,
    ) -> torch.Tensor:
        """Non-DWDP NVFP4 MoE implementation using single-tensor ops."""
        output_dtype = torch.bfloat16
        sm_version = get_sm_version()
        use_rubin = (sm_version == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE)

        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start

        if recv_expert_count is not None:
            assert deep_ep_expert_capacity is not None
            assert effective_top_k == 1
            assert recv_expert_count.dim() == 1
            assert recv_expert_count.numel() == esp
            assert x.size(0) == esp * deep_ep_expert_capacity
            if use_count_native_expert_metadata:
                # The custom-op schema is shared with the direct-metadata path.
                # Count-native runners interpret each metadata argument as the
                # same expert-count tensor and do not materialize adapter arrays.
                tile_idx_to_expert_idx = recv_expert_count
                tile_idx_to_mn_limit = recv_expert_count
                expanded_idx_to_permuted_idx = recv_expert_count
                permuted_idx_to_expanded_idx = recv_expert_count
                total_num_padded_tokens = None
                num_non_exiting_tiles = recv_expert_count
            else:
                tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_metadata_from_expert_counts(
                    expert_counts=recv_expert_count,
                    capacity=deep_ep_expert_capacity,
                    tile_size=tile_size,
                )
        else:
            tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                local_expert_offset=slot_start,
                local_num_experts=esp,
                tile_tokens_dim=tile_size,
            )

        has_aux_streams = self._has_moe_output_memset_aux_stream()
        if self.use_fused_finalize and has_aux_streams:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                if use_count_native_expert_metadata:
                    torch.ops.trtllm.moe_output_memset_from_expert_counts_inplace(
                        input=moe_output,
                        expert_counts=recv_expert_count,
                        expert_capacity=deep_ep_expert_capacity,
                        ep_size=self.mapping.moe_ep_size,
                        enable_alltoall=enable_alltoall,
                    )
                else:
                    torch.ops.trtllm.moe_output_memset_inplace(
                        input=moe_output,
                        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles=num_non_exiting_tiles,
                        tile_tokens_dim=tile_size,
                        top_k=effective_top_k,
                        ep_size=self.mapping.moe_ep_size,
                        enable_alltoall=enable_alltoall,
                    )
                self.event_dict[EventType.MoeOutputMemset].record()

        # Fused gather + GEMM + activation + quantize for FC1.
        # For gated (SwiGLU): weights are interleaved [up, gate], output is N/2.
        # For non-gated (Relu2): weights are plain, output is N.
        gather_act_op = (
            torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin
            if use_rubin else torch.ops.trtllm.
            cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell)

        gather_act_kwargs = dict(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=weight_view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
            alpha=weight_view.fc1_global_scale,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
        )
        if use_count_native_expert_metadata:
            gather_act_kwargs["expert_counts"] = recv_expert_count
            gather_act_kwargs["expert_capacity"] = deep_ep_expert_capacity
        if use_rubin:
            gather_act_kwargs["output_tensor"] = None
            gather_act_kwargs["output_sf_tensor"] = None
        else:
            gather_act_kwargs["activation_type"] = self.activation_type
            gather_act_kwargs["swiglu_limit_scalar"] = self.act_clamp
        gather_act_kwargs["activation_type"] = self.activation_type

        x, x_sf = gather_act_op(**gather_act_kwargs)

        if self.use_fused_finalize:
            if has_aux_streams:
                self.event_dict[EventType.MoeOutputMemset].wait()
            elif use_count_native_expert_metadata:
                torch.ops.trtllm.moe_output_memset_from_expert_counts_inplace(
                    input=moe_output,
                    expert_counts=recv_expert_count,
                    expert_capacity=deep_ep_expert_capacity,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
            else:
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=moe_output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=effective_top_k,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )

            # FC2: Grouped GEMM + Finalize (scatter-add) fusion
            finalize_inplace_op = (
                torch.ops.trtllm.
                cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin
                if use_rubin else torch.ops.trtllm.
                cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell)

            finalize_inplace_op(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
                alpha=weight_view.fc2_global_scale,
                output=moe_output,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                token_final_scales=token_final_scales,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                num_local_experts=esp,
                local_expert_offset=slot_start,
                tile_size=tile_size,
                output_dtype=output_dtype,
                expert_counts=(recv_expert_count
                               if use_count_native_expert_metadata else None),
                expert_capacity=(deep_ep_expert_capacity
                                 if use_count_native_expert_metadata else 0),
            )
        else:
            if use_rubin:
                # Rubin does not have a basic grouped GEMM kernel (without
                # fused finalize) yet. Force use_fused_finalize=True for Rubin.
                raise NotImplementedError(
                    "Rubin (SM107) MOE requires use_fused_finalize=True. "
                    "Basic grouped GEMM without finalize fusion is not yet "
                    "supported on Rubin.")
            x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
                alpha=weight_view.fc2_global_scale,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                num_local_experts=esp,
                local_expert_offset=slot_start,
                tile_size=tile_size,
                output_dtype=output_dtype,
            )
            torch.ops.trtllm.moe_unpermute_inplace(
                permuted_input=x,
                output=moe_output,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
            )
        return moe_output

    def run_moe_bf16(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        """Autotuner wrapper for BF16/FP16 MoE on Rubin (SM107)."""
        assert not self.has_any_quant
        # The FC2 op below always fuses finalize; ``can_implement`` declines
        # SM107 when the caller disabled it, so honor that rather than ignore it.
        assert self.use_fused_finalize, (
            "BF16 CuteDSL MoE has no unfused FC2 path")
        output_dtype = x.dtype
        effective_top_k = token_selected_experts.size(-1)

        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_selected_experts.size(0),
                                         self.hidden_size)
            assert moe_output.dtype == output_dtype

        if token_selected_experts.size(0) == 0:
            return moe_output

        self._ensure_bf16_alpha(x.device)

        use_locality_domain = self._locality_domain_runtime is not None
        if use_locality_domain:
            forward_impl = self._run_moe_bf16_locality_domain
            workload_identity = self._locality_domain_workload_identity(x.dtype)
            tuner_key = (
                "CuteDslFusedMoE::run_moe_bf16::locality_domain_end_to_end")
        else:
            forward_impl = self.run_moe_bf16_impl
            workload_identity = None
            tuner_key = "CuteDslFusedMoE::run_moe_bf16"

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoEBF16Runner(
            forward_impl=forward_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            enable_alltoall=enable_alltoall,
            output_dtype=output_dtype,
            workload_identity=workload_identity,
        )

        inputs = [x, token_selected_experts, token_final_scales, moe_output]
        _, best_tactic = tuner.choose_one(
            tuner_key,
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def _ensure_bf16_alpha(self, device: torch.device) -> torch.Tensor:
        if not hasattr(self, '_bf16_alpha') or self._bf16_alpha is None \
                or self._bf16_alpha.device != device \
                or self._bf16_alpha.size(0) != self.expert_size_per_partition:
            self._bf16_alpha = torch.ones(self.expert_size_per_partition,
                                          dtype=torch.float32,
                                          device=device)
        return self._bf16_alpha

    def run_moe_bf16_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        moe_output: torch.Tensor,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """BF16/FP16 MoE implementation using CuTE DSL Rubin kernels.

        FC1: gather + grouped GEMM + SwiGLU fusion
        FC2: grouped GEMM + finalize (scatter-add) fusion
        """
        output_dtype = x.dtype
        effective_top_k = token_selected_experts.size(-1)

        # Step 1: moe_sort — identical to NVFP4 path
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=tile_size,
        )

        # Step 2: Memset overlap for fused finalize
        has_aux = self._has_moe_output_memset_aux_stream()
        if has_aux:
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(self._moe_output_memset_run_stream())

        # Step 3: Alpha = 1.0 for all local experts (no quantization scaling)
        # The wrapper initializes this before autotuner profiling so allocation
        # does not happen inside CUDA graph capture.
        self._ensure_bf16_alpha(x.device)

        # Step 4: FC1 — BF16 gather + grouped GEMM + SwiGLU
        fc1_out = torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
            input=x,
            weight=self.w3_w1_weight,
            alpha=self._bf16_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=None,
            partition_id=-1,
        )

        # Step 5: Memset overlap — zero out moe_output rows not touched by
        # the finalize kernel (same pattern as NVFP4 path).
        if has_aux:
            with torch.cuda.stream(self._moe_output_memset_run_stream()):
                self.event_dict[EventType.Main].wait()
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=moe_output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=effective_top_k,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
                self.event_dict[EventType.MoeOutputMemset].record()
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
                top_k=effective_top_k,
                ep_size=self.mapping.moe_ep_size,
                enable_alltoall=enable_alltoall,
            )

        # Step 6: FC2 — BF16 grouped GEMM + finalize (scatter-add) inplace
        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
            input=fc1_out,
            weight=self.w2_weight,
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
        )
        return moe_output

    def _run_moe_nvfp4_locality_domain(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """locality domain path: half-weight children, shared output buffers, fork/join.

        Each child holds localized half-N weights. Both partitions use the
        same tuned tactic and write directly into their strided regions of the
        shared FC1/FC2 output buffers.
        """
        output_dtype = torch.bfloat16
        num_partitions = self._locality_domain_plan.num_partitions
        shards = self._locality_domain_weight_shards
        effective_top_k = token_selected_experts.size(-1)

        # --- moe_sort (shared, on main stream) ---
        (tile_idx_to_expert_idx, tile_idx_to_mn_limit,
         expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
         total_num_padded_tokens,
         num_non_exiting_tiles) = torch.ops.trtllm.moe_sort(
             token_selected_experts=token_selected_experts,
             token_final_scales=token_final_scales,
             num_experts=self.num_slots,
             top_k=effective_top_k,
             local_expert_offset=self.slot_start,
             local_num_experts=self.expert_size_per_partition,
             tile_tokens_dim=tile_size,
         )

        # --- Allocate shared output ---
        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)

        # --- FC1: gather + grouped GEMM + SwiGLU, fork/join ---
        # Each child has half-N weight [num_exp, inner_size, hidden_size].
        # Shared output buffers:
        #   c:    [permute_m, inner_size] — each locality domain writes half via strided layout
        #   c_sf: [full_sf_size] — each locality domain writes at K-tile offset via full_c_shape
        #   Kernel uses full_c_shape to compute sfc layout with full M-tile stride,
        #   so no copy-back or interleave is needed.
        m = permuted_idx_to_expanded_idx.size(0)
        shard_weight_n = shards[0]['w3_w1_weight'].size(1)  # half interleaved N
        shard_interm = shard_weight_n // 2  # post-SwiGLU per partition
        full_interm = shard_interm * num_partitions
        fc1_out = torch.empty(m,
                              shard_interm // 2 * 2,
                              dtype=torch.float4_e2m1fn_x2,
                              device=x.device)
        full_sf_size = m * full_interm // self.scaling_vector_size
        fc1_out_sf = torch.empty(full_sf_size,
                                 dtype=torch.uint8,
                                 device=x.device)

        assert self.use_fused_finalize, (
            "locality domain MoE requires use_fused_finalize=True on Rubin")

        # Inner FC tactics are prepared before outer CUDA-graph profiling. Keep
        # that first-use tuning on the main stream; normal execution still
        # overlaps this memset with the already-prepared FC1 composite op.
        memset_overlapped = (overlap_moe_output_memset
                             and self._has_moe_output_memset_aux_stream())
        if memset_overlapped:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=moe_output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=effective_top_k,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
                self.event_dict[EventType.MoeOutputMemset].record()

        torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            weight_0=shards[0]['w3_w1_weight'].view(torch.float4_e2m1fn_x2),
            weight_1=shards[1]['w3_w1_weight'].view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale_0=shards[0]['fc1_weight_block'].view(torch.uint8),
            weight_scale_1=shards[1]['fc1_weight_block'].view(torch.uint8),
            alpha=shards[0]['fc1_global'],
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=shards[0]['fc2_input_scale'],
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=fc1_out,
            output_sf_tensor=fc1_out_sf,
            scaling_vector_size=self.scaling_vector_size,
            activation_type=self.activation_type,
        )

        fc1_out_sf_merged = fc1_out_sf

        if memset_overlapped:
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
                top_k=effective_top_k,
                ep_size=self.mapping.moe_ep_size,
                enable_alltoall=enable_alltoall,
            )

        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input=fc1_out.view(torch.float4_e2m1fn_x2),
            weight_0=shards[0]['w2_weight'].view(torch.float4_e2m1fn_x2),
            weight_1=shards[1]['w2_weight'].view(torch.float4_e2m1fn_x2),
            input_scale=fc1_out_sf_merged.view(torch.uint8),
            weight_scale_0=shards[0]['fc2_weight_block'].view(torch.uint8),
            weight_scale_1=shards[1]['fc2_weight_block'].view(torch.uint8),
            alpha=shards[0]['fc2_global'],
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
            ep_size=self.mapping.moe_ep_size,
            enable_alltoall=enable_alltoall,
            scaling_vector_size=self.scaling_vector_size,
        )

        return moe_output

    def _run_moe_bf16_locality_domain(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """locality domain path for unquantized BF16 MoE on Rubin."""
        output_dtype = x.dtype
        num_partitions = self._locality_domain_plan.num_partitions
        shards = self._locality_domain_weight_shards
        effective_top_k = token_selected_experts.size(-1)

        (tile_idx_to_expert_idx, tile_idx_to_mn_limit,
         expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
         total_num_padded_tokens,
         num_non_exiting_tiles) = torch.ops.trtllm.moe_sort(
             token_selected_experts=token_selected_experts,
             token_final_scales=token_final_scales,
             num_experts=self.num_slots,
             top_k=effective_top_k,
             local_expert_offset=self.slot_start,
             local_num_experts=self.expert_size_per_partition,
             tile_tokens_dim=tile_size,
         )

        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_selected_experts.size(0),
                                         self.hidden_size)
            assert moe_output.dtype == output_dtype

        self._ensure_bf16_alpha(x.device)

        m = permuted_idx_to_expanded_idx.size(0)
        shard_weight_n = shards[0]['w3_w1_weight'].size(1)
        shard_interm = shard_weight_n // 2
        full_interm = shard_interm * num_partitions
        fc1_out = torch.empty(m,
                              full_interm,
                              dtype=output_dtype,
                              device=x.device)

        assert self.use_fused_finalize, (
            "locality domain MoE requires use_fused_finalize=True on Rubin")

        memset_overlapped = (overlap_moe_output_memset
                             and self._has_moe_output_memset_aux_stream())
        if memset_overlapped:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=moe_output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=effective_top_k,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
                self.event_dict[EventType.MoeOutputMemset].record()

        torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin(
            input=x,
            weight_0=shards[0]['w3_w1_weight'],
            weight_1=shards[1]['w3_w1_weight'],
            alpha=self._bf16_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=fc1_out,
        )

        if memset_overlapped:
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
                top_k=effective_top_k,
                ep_size=self.mapping.moe_ep_size,
                enable_alltoall=enable_alltoall,
            )

        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input=fc1_out,
            weight_0=shards[0]['w2_weight'],
            weight_1=shards[1]['w2_weight'],
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
            ep_size=self.mapping.moe_ep_size,
            enable_alltoall=enable_alltoall,
        )

        return moe_output

    def run_moe_fp8_block_scales(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        assert self.has_deepseek_fp8_block_scales
        assert x_sf is None
        assert self.activation_type == ActivationType.Swiglu, (
            "FP8 block-scales MoE path hardcodes SwiGLU (see swiglu_fused_moe "
            f"below); got activation_type={ActivationType(self.activation_type).name}"
        )
        weight_dtype = self.w3_w1_weight.dtype

        (
            permuted_row_to_unpermuted_row,
            permuted_token_selected_experts,
            x,
            expert_first_token_offset,
            permuted_token_final_scales,
            unpermuted_row_to_permuted_row,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=None,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=True,
        )
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w3_w1_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[0],
            offset_array=expert_first_token_offset,
        )
        x = swiglu_fused_moe(x, self.act_clamp)
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w2_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[1],
            offset_array=expert_first_token_offset,
        )
        top_k = self.routing_method.top_k
        if token_selected_experts is not None:
            top_k = token_selected_experts.shape[-1]

        x = torch.ops.trtllm.moe_finalize_scale_op(
            x,
            None,  # biases
            token_final_scales,
            unpermuted_row_to_permuted_row,
            permuted_row_to_unpermuted_row,
            token_selected_experts,
            expert_first_token_offset,
            enable_alltoall,
            token_final_scales.size(0),  # num_rows
            self.hidden_size,  # (possibly padded) hidden_size
            self.unpadded_hidden_size,  # original hidden size
            top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )
        return x

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run MoE computation with CuteDSL backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes (fp8_block_scales, nvfp4, and unquantized BF16).

        Returns:
            final_hidden_states tensor.
        """
        del workspace  # CuteDSL kernels allocate their own intermediates.
        plan = require_comm_plan(self, ctx)
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        moe_output = plan.moe_output
        enable_alltoall = plan.enable_alltoall

        # Execute MoE computation
        if self.has_nvfp4:
            weight_view = self._build_local_weight_view()
            result = self.run_moe_nvfp4(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
                weight_view=weight_view,
                recv_expert_count=plan.recv_expert_count,
                deep_ep_expert_capacity=plan.deep_ep_expert_capacity,
                use_deep_ep_direct_metadata=plan.use_deep_ep_direct_metadata,
            )
        elif self.has_deepseek_fp8_block_scales:
            result = self.run_moe_fp8_block_scales(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                enable_alltoall=enable_alltoall)
        elif not self.has_any_quant:
            return self.run_moe_bf16(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall)
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )
        return result

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        super().load_weights(weights,
                             allow_partial_loading=allow_partial_loading)
        # Keep DWDP registration after base weight loading. This preserves
        # loaded tensors for collector setup and remains compatible with the
        # later locality domain post_load_weights splitting flow.
        dwdp_handle_collector = getattr(self, "dwdp_handle_collector", None)
        if dwdp_handle_collector is not None:
            dwdp_handle_collector.register_weights(self)

    def post_load_weights(self):
        super().post_load_weights()
        # Split full weights into per-partition halves on localized memory
        if self._locality_domain_runtime is not None:
            self._locality_domain_weight_shards = self._split_weights_for_locality_domain(
            )
            # Weight splitting initializes the process-lifetime locality domain resource.
            # Resolve the borrowed remainder stream now, never during capture.
            self._get_reserved_moe_output_memset_stream()
            self._release_full_weights_after_locality_domain_split()

    def _release_full_weights_after_locality_domain_split(self):
        """Release full tensors that are replaced by localized locality domain shards."""
        for param_name in (
                "w3_w1_weight",
                "w2_weight",
                "w3_w1_weight_scale",
                "w2_weight_scale",
        ):
            param = getattr(self, param_name, None)
            if param is None:
                continue
            setattr(
                self,
                param_name,
                torch.nn.Parameter(param.new_empty(0), requires_grad=False),
            )
        self.quant_method.setup_quant_scales(self)

    def _split_weights_for_locality_domain(self):
        """Split full N-dimension weights into per-partition halves.

        After normal load_weights + post_load_weights, the full weights
        are on self. Split them along dim=1 (N) and allocate halves on
        each locality domain partition's localized memory.
        """
        num_p = self._locality_domain_plan.num_partitions
        shards = []
        for pid in range(num_p):
            with self._locality_domain_runtime.partition_weight_context(pid):
                n1 = self.w3_w1_weight.size(1)
                n2 = self.w2_weight.size(1)
                half_n1 = n1 // num_p
                half_n2 = n2 // num_p
                w3_w1_slice = self.w3_w1_weight[:, pid * half_n1:(pid + 1) *
                                                half_n1]
                w2_slice = self.w2_weight[:, pid * half_n2:(pid + 1) * half_n2]
                shard = {
                    'w3_w1_weight': _copy_to_new_cuda_allocation(w3_w1_slice),
                    'w2_weight': _copy_to_new_cuda_allocation(w2_slice),
                }
                if self.has_nvfp4:
                    fc1_scale_slice = self.quant_scales.fc1_weight_block[:,
                                                                         pid *
                                                                         half_n1:
                                                                         (pid +
                                                                          1) *
                                                                         half_n1]
                    fc2_scale_slice = self.quant_scales.fc2_weight_block[:,
                                                                         pid *
                                                                         half_n2:
                                                                         (pid +
                                                                          1) *
                                                                         half_n2]
                    shard.update({
                        'fc1_weight_block':
                        _copy_to_new_cuda_allocation(fc1_scale_slice),
                        'fc2_weight_block':
                        _copy_to_new_cuda_allocation(fc2_scale_slice),
                        'fc1_global':
                        self.quant_scales.fc1_global,
                        'fc2_global':
                        self.quant_scales.fc2_global,
                        'fc2_input_scale':
                        self.fc2_input_scale,
                    })
                shards.append(shard)
        return shards
