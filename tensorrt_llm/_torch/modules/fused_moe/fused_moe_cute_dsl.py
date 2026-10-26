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
from ...custom_ops.cute_dsl_custom_ops import (
    GroupedGemmInputsHelper,
    Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmRunner,
    Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner)
from ...cute_dsl_utils import IS_CUTLASS_DSL_INTERNAL_AVAILABLE
from ...model_config import ModelConfig, QuantConfig
from ...ugpu.autotune import \
    UgpuConcurrentTunableRunner as _UgpuConcurrentTunableRunner
from ...ugpu.policy import UgpuExecutionPlanner
from ...ugpu.runtime import UgpuRuntime
from ...ugpu_utils import get_reserved_remainder_stream
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor,
                      get_last_power_of_2_num_tokens_buckets,
                      last_positive_power_of_2)
from ..fused_ops.nvfp4_quantize_triton import fp4_quantize
from ..linear import _copy_to_new_cuda_allocation
from .fused_moe_cutlass import CutlassFusedMoE
from .impl_contract import (MoEDeployment, MoEEligibility, MoEProblem,
                            MoERejectReason, MoERunContext, MoEStaticCapability,
                            require_comm_plan)
from .interface import AlltoallMethodType, _reject
from .quantization import (BF16CuteDslFusedMoEMethod, MoEWeightLoadingMode,
                           NVFP4CuteDslFusedMoEMethod)
from .routing import BaseMoeRoutingMethod

# Kill switch for CuteDslFusedMoE.quantize_input_async (fork the NVFP4 input
# quantization onto the MoE aux stream so it overlaps top-k + moe_sort).
ASYNC_INPUT_QUANT = os.environ.get("TLLM_MOE_ASYNC_INPUT_QUANT", "1") == "1"


def _unwrap_ugpu_runner(runner: TunableRunner) -> TunableRunner:
    """Return the kernel runner wrapped by the shared uGPU tuning adapter."""
    if isinstance(runner, _UgpuConcurrentTunableRunner):
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
        runner = _unwrap_ugpu_runner(runner)
        if isinstance(runner, outer_runner_type):
            tile_size = tactic
    if tile_size is None:
        return True

    for runner, tactic in comb:
        runner = _unwrap_ugpu_runner(runner)
        if isinstance(runner, checked_runner_types):
            mma_tiler_mn, *_ = tactic
            if mma_tiler_mn[0] != tile_size:
                return False
    return True


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
                 workload_identity: Optional[Tuple] = None,
                 prime_inner_tactics: bool = False):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_finalize_fusion = enable_finalize_fusion
        self.enable_alltoall = enable_alltoall

        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size
        # Extra terms that distinguish two otherwise identically-shaped
        # workloads (uGPU topology, activation type, ...). Cache key only --
        # it must not imply anything about ``forward_impl``'s signature.
        self.workload_identity = workload_identity
        # uGPU only: ``forward_impl`` nests its own AutoTuner calls and accepts
        # ``overlap_moe_output_memset``. Single-op backends must leave this off.
        self.prime_inner_tactics = prime_inner_tactics

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
            helper = CuteDslFusedMoENvfp4InputsHelper(self.num_experts,
                                                      self.top_k,
                                                      self.num_local_experts,
                                                      self.local_expert_offset)
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(DynamicTensorSpec(
                    0, 0, get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2), ),
                constraint_specs=(ConstraintSpec(1, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(2, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(3, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(
                                      4, 0, helper.infer_shape_num_tokens)),
                inputs_pre_hook=helper.inputs_pre_hook,
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(self,
                inputs: List[torch.Tensor],
                tactic: Optional[int],
                do_preparation: bool = False) -> torch.Tensor:
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
        return self.forward_impl(*inputs,
                                 enable_alltoall=self.enable_alltoall,
                                 tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        # Build the tuple of runner types that need CTA_M == tile_size check.
        # Include both Blackwell (Sm100) and Rubin (Sm107) runners.
        checked_runner_types = [
            Sm100BlockScaledContiguousGroupedGemmRunner,
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
            Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
            Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
        ]
        if IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
            from ...custom_ops.cute_dsl_custom_ops import (
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner)
            checked_runner_types.extend([
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner,
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                # The fused FC12 runner also requires mma_tiler_m == tile_size;
                # without it the Cartesian replay pairs inner tactics captured
                # under one routing tile with a different outer tile.
                Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner,
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
                 workload_identity: Optional[Tuple] = None,
                 prime_inner_tactics: bool = False):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_alltoall = enable_alltoall
        self.output_dtype = output_dtype
        # See CuteDslFusedMoENvfp4Runner: cache key only, and uGPU-only.
        self.workload_identity = workload_identity
        self.prime_inner_tactics = prime_inner_tactics

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
        return self.forward_impl(*inputs,
                                 enable_alltoall=self.enable_alltoall,
                                 tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        # BF16 GEMM runners that need CTA_M == tile_size.
        checked_runner_types = []
        if IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
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


class CuteDslFusedMoE(CutlassFusedMoE):
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

    def _has_moe_output_memset_aux_stream(self) -> bool:
        event_dict = getattr(self, "event_dict", None)
        aux_stream_dict = getattr(self, "aux_stream_dict", None)
        return (event_dict is not None and aux_stream_dict is not None
                and EventType.Main in event_dict
                and EventType.MoeOutputMemset in event_dict
                and AuxStreamType.MoeOutputMemset in aux_stream_dict)

    # ``supports_moe_lora`` is restated because CutlassFusedMoE declares True
    # and the exact-class comparison it replaces answered False here.
    # ``supports_dwdp`` is the capability this backend adds. CuteDslB12xFusedMoE
    # derives from here and needs both, but must spell them out again: setting
    # the attribute replaces the whole object rather than one field.
    capabilities = MoEStaticCapability(supports_moe_lora=False,
                                       supports_dwdp=True)

    def _get_reserved_moe_output_memset_stream(
            self) -> Optional[torch.cuda.Stream]:
        """Resolve and cache the strict split's remainder stream before capture."""
        if hasattr(self, "_cached_reserved_moe_output_memset_stream"):
            return self._cached_reserved_moe_output_memset_stream

        stream = None
        if self._ugpu_runtime is not None:
            stream = get_reserved_remainder_stream()
        self._cached_reserved_moe_output_memset_stream = stream
        return stream

    def _moe_output_memset_run_stream(self) -> torch.cuda.Stream:
        """Select the remainder stream when present, otherwise the aux stream."""
        remainder_stream = self._get_reserved_moe_output_memset_stream()
        if remainder_stream is not None:
            return remainder_stream
        return self.aux_stream_dict[AuxStreamType.MoeOutputMemset]

    def _ugpu_workload_identity(self,
                                input_dtype: torch.dtype) -> Tuple[Any, ...]:
        """Describe the sharded MoE workload and its compute topology."""
        if self._ugpu_runtime is None or self._ugpu_weight_shards is None:
            raise RuntimeError(
                "uGPU workload identity requires initialized shards")
        shard_identity = tuple((
            tuple(shard['w3_w1_weight'].shape),
            str(shard['w3_w1_weight'].dtype),
            tuple(shard['w2_weight'].shape),
            str(shard['w2_weight'].dtype),
        ) for shard in self._ugpu_weight_shards)
        return (
            self._ugpu_plan.num_partitions,
            self._ugpu_runtime.topology_identity(),
            str(input_dtype),
            shard_identity,
        )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """CuteDSL grouped GEMM: NVFP4 on SM100/SM103/SM107, or unquantized BF16 on SM107."""
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

        # Unquantized BF16 on Rubin (SM107).
        if quant_algo is None:
            if sm_version != 107:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"Unquantized CuteDslFusedMoE requires SM107, got SM{sm_version}"
                )
            if not IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    "Unquantized CuteDslFusedMoE on SM107 requires CuTE DSL internal"
                )
            return MoEEligibility.ok()

        # NVFP4 - SM in {100, 103, 107}
        if quant_algo == QuantAlgo.NVFP4:
            if sm_version not in {100, 103, 107}:
                return _reject(
                    MoERejectReason.SM_UNSUPPORTED,
                    f"NVFP4 requires SM100, SM103, or SM107, got SM{sm_version}"
                )
            if sm_version == 107 and not IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    "NVFP4 on SM107 (Rubin) requires CuTE DSL internal")
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
        swiglu_limit_scalar: Optional[float] = None,
        init_load_balancer: bool = True,
        activation_type: ActivationType = ActivationType.Swiglu,
        override_quant_config: Optional[QuantConfig] = None,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            swiglu_limit_scalar=swiglu_limit_scalar,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
            ignore_weights=True,
        )
        self.swiglu_limit_scalar = swiglu_limit_scalar or float("inf")

        # SiTU (Kimi K3) constants. Unlike CUTLASS -- whose SiTuAdaptor reads
        # per-expert ``swiglu_alpha`` / ``swiglu_beta`` tensors -- the CuTe DSL
        # epilogue folds them in as trace-time scalars, so they are resolved
        # here as plain floats. The model layer must pass both; this backend
        # does not read pretrained_config.
        self.situ_beta, self.situ_linear_beta = self._resolve_situ_betas(
            situ_beta, situ_linear_beta)

        # Output-memset overlap is independent of MoE chunking, so ensure its
        # stream and events exist even if the parent creates no chunking event.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[
                AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        if self.event_dict is None:
            self.event_dict = {}
        for key in [EventType.Main, EventType.MoeOutputMemset]:
            if key not in self.event_dict:
                self.event_dict[key] = torch.cuda.Event()
        # Async input quantization (see quantize_input_async / _join_pending_x_ready).
        self._x_quant_event = torch.cuda.Event()
        self._pending_x_ready_event: Optional[torch.cuda.Event] = None

        self.scaling_vector_size = 16
        self.quant_config = override_quant_config if override_quant_config is not None else self.quant_config
        # uGPU: fork/join with _ugpu kernel variants + shared output buffers.
        # Weight splitting happens in post_load_weights_impl after normal loading.
        self._ugpu_runtime = None
        self._ugpu_weight_shards = None  # set in post_load_weights_impl
        planner = UgpuExecutionPlanner(model_config.ugpu_policy)
        self._ugpu_plan = planner.plan_moe(
            self.quant_config,
            moe_backend=model_config.moe_backend,
            use_fused_finalize=self.use_fused_finalize,
            dtype_activation=self.dtype,
        )
        if self._ugpu_plan.enabled:
            if self._using_load_balancer():
                raise ValueError(
                    "uGPU MoE does not support EPLB because localized weight "
                    "shards cannot follow runtime expert migration.")
            self._ugpu_runtime = UgpuRuntime(self._ugpu_plan.num_partitions)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _resolve_situ_betas(
        self,
        situ_beta: Optional[float],
        situ_linear_beta: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Resolve the SiTU constants for ``ActivationType.SiTu``.

        Both betas must be passed explicitly. Returns ``(None, None)`` for
        every non-SiTU activation.
        """
        if self.activation_type != ActivationType.SiTu:
            if situ_beta is not None or situ_linear_beta is not None:
                raise ValueError(
                    "situ_beta / situ_linear_beta require "
                    f"ActivationType.SiTu, got {self.activation_type.name}.")
            return None, None

        if situ_beta is None or situ_linear_beta is None:
            raise ValueError(
                "CuteDslFusedMoE SiTU requires explicit situ_beta and "
                "situ_linear_beta arguments; got "
                f"{situ_beta} and {situ_linear_beta}.")
        if situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError(
                "CuteDslFusedMoE SiTU beta parameters must be positive, got "
                f"{situ_beta} and {situ_linear_beta}.")
        if self.swiglu_limit_scalar != float("inf"):
            # Matches MegaMoE (both backends) and DeepGEMM, which reject
            # activation_clamp together with SiTU.
            raise ValueError(
                "CuteDslFusedMoE SiTU does not support a SwiGLU clamp; "
                "drop swiglu_limit_scalar for SiTU checkpoints.")
        return float(situ_beta), float(situ_linear_beta)

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

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        return AlltoallMethodType.NotEnabled

    def validate_configurable_moe(self, moe) -> None:
        """Reject runtime weight owners that cannot update localized shards."""
        if not self._ugpu_plan.enabled:
            return
        if moe.enable_dwdp:
            raise ValueError(
                "uGPU MoE does not support DWDP because DWDP rebinds the "
                "backend parameters after localized weight shards are created.")
        if moe._using_load_balancer():
            raise ValueError(
                "uGPU MoE does not support EPLB because localized weight "
                "shards cannot follow runtime expert migration.")

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
            return super()._get_quant_method()
        # Unquantized: use BF16 method on SM107 for FC1 weight interleaving
        if get_sm_version() == 107 and IS_CUTLASS_DSL_INTERNAL_AVAILABLE:
            return BF16CuteDslFusedMoEMethod()
        return super()._get_quant_method()

    def supports_moe_output_in_alltoall_workspace(self):
        return self.has_nvfp4 or (not self.has_any_quant
                                  and get_sm_version() == 107)

    def set_entry_nvfp4_input(self, x: torch.Tensor, x_fp4: torch.Tensor,
                              x_sf: torch.Tensor) -> None:
        """Stash the NVFP4 of ``x`` (the bytes ``fp4_quantize(x, fc31_input_scale, 16, False, False)`` returns)
        produced by the MoE-entry add-RMSNorm (TRTLLM_MOE_ENTRY_NORM_NVFP4); quantize_input takes it instead of
        quantizing ``x`` again, for that same tensor object only."""
        self._entry_nvfp4_input = (x, x_fp4, x_sf)

    def _take_entry_nvfp4_input(
            self,
            x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        stash = self.__dict__.pop("_entry_nvfp4_input", None)
        if stash is not None and stash[0] is x:
            return stash[1], stash[2]
        return None

    def clear_entry_nvfp4_input(self) -> bool:
        """Drop an untaken stash; True when the MoE quantized its input itself (e.g. padded or chunked input)."""
        return self.__dict__.pop("_entry_nvfp4_input", None) is not None

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
            entry = (None if isinstance(x, Fp4QuantizedTensor) else
                     self._take_entry_nvfp4_input(x))
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            elif entry is not None:
                # TRTLLM_MOE_ENTRY_NORM_NVFP4: the add-RMSNorm that produced x also wrote these bytes.
                x_row = x.shape[0]
                x, x_sf = entry
            else:
                x_row = x.shape[0]
                # Same bytes as torch.ops.trtllm.fp4_quantize (linear SF); a
                # Triton kernel when TRTLLM_NVFP4_QUANTIZE_TRITON=1.
                x, x_sf = fp4_quantize(x, self.fc31_input_scale,
                                       self.scaling_vector_size, False, False)
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

    def quantize_input_async(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[torch.cuda.Event]]:
        """Quantize ``x`` on the MoE aux stream so it overlaps top-k and ``moe_sort``.

        At small decode batches the three per-layer prologue kernels (top-k,
        input quantization, routing/permutation) are all latency-bound and
        serialize on the main stream (~25 us/layer for 8 tokens on GB300).
        Quantization only depends on ``x``, so the scheduler forks it here
        before routing; the consumer joins via the returned event right before
        the first kernel that reads ``x``/``x_sf`` (``run_moe_nvfp4_impl``,
        after ``moe_sort``). Falls back to the synchronous ``quantize_input``
        (event ``None``) when disabled, not NVFP4, already quantized, or no aux
        stream is available.
        """
        if (not ASYNC_INPUT_QUANT or not self.has_nvfp4
                or isinstance(x, Fp4QuantizedTensor)
                or "_entry_nvfp4_input" in self.__dict__
                or not self._has_moe_output_memset_aux_stream()):
            x_q, x_sf = self.quantize_input(x, post_quant_comm=False)
            return x_q, x_sf, None
        main_stream = torch.cuda.current_stream()
        aux_stream = self.aux_stream_dict[AuxStreamType.MoeOutputMemset]
        self.event_dict[EventType.Main].record()
        x.record_stream(aux_stream)
        with torch.cuda.stream(aux_stream):
            self.event_dict[EventType.Main].wait()
            x_q, x_sf = self.quantize_input(x, post_quant_comm=False)
            self._x_quant_event.record()
        x_q.record_stream(main_stream)
        if x_sf is not None:
            x_sf.record_stream(main_stream)
        return x_q, x_sf, self._x_quant_event

    def _join_pending_x_ready(self) -> None:
        """Wait for the async input quantization (once per forward)."""
        event = getattr(self, "_pending_x_ready_event", None)
        if event is not None:
            event.wait()
            self._pending_x_ready_event = None

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        weight_view: Optional[NvFp4WeightView] = None,
        x_ready_event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        """NVFP4 MoE computation.

        Uses the single-tensor ``run_moe_nvfp4_impl`` path. (The former
        multi-B DWDP path was removed once DWDP switched to VA: VA swaps
        ``param.data`` to a full [num_experts, ...] tensor, so the single-
        tensor kernel is sufficient.)

        Args:
            weight_view: Bundled weight tensors. Must not be None.
            x_ready_event: Set when ``x``/``x_sf`` come from
                ``quantize_input_async``; joined inside ``run_moe_nvfp4_impl``
                right before the FC1 gather GEMM (the uGPU path joins here).
        """
        assert self.has_nvfp4
        assert weight_view is not None
        # SiTu belongs here: everything below it already handles SiTu --
        # ``run_moe_nvfp4_impl`` sets ``gather_act_kwargs["activation_type"]``
        # and forwards ``situ_beta`` / ``situ_linear_beta``, and both act-fusion
        # kernels (blackwell/... and rubin/...) implement the SiTU epilogue.
        # This gate ran BEFORE that dispatch, so it rejected SiTu even though
        # the path underneath supports it -- Kimi K3 NVFP4 on CUTEDSL died here
        # (job 461412).
        #
        # ``self.activation_type`` may be a plain int, so do not format it with
        # ``.name``: that turned the rejection itself into
        # ``AttributeError: 'int' object has no attribute 'name'`` and hid which
        # activation was actually refused.
        if self.activation_type not in (ActivationType.Swiglu,
                                        ActivationType.Relu2,
                                        ActivationType.SiTu):
            raise NotImplementedError(
                "CuteDSL NVFP4 FC1 supports only SwiGLU, Relu2 and SiTu; "
                f"got {getattr(self.activation_type, 'name', self.activation_type)}"
            )
        output_dtype = torch.bfloat16

        use_ugpu = self._ugpu_runtime is not None
        if use_ugpu:
            if self.activation_type != ActivationType.Swiglu:
                raise NotImplementedError(
                    "Rubin uGPU NVFP4 MoE currently supports SwiGLU only")

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

        if use_ugpu:
            # No overlap on the uGPU path yet: join before dispatching.
            if x_ready_event is not None:
                x_ready_event.wait()
            self._pending_x_ready_event = None
        else:
            self._pending_x_ready_event = x_ready_event

        if use_ugpu:
            forward_impl = self._run_moe_nvfp4_ugpu
            workload_identity = self._ugpu_workload_identity(x.dtype)
            tuner_key = ("CuteDslFusedMoE::run_moe_nvfp4::ugpu_end_to_end")
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
            prime_inner_tactics=use_ugpu,
        )

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
    ) -> torch.Tensor:
        """Non-DWDP NVFP4 MoE implementation using single-tensor ops."""
        output_dtype = torch.bfloat16
        sm_version = get_sm_version()
        use_rubin = (sm_version == 107 and IS_CUTLASS_DSL_INTERNAL_AVAILABLE)

        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start

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
        if use_rubin:
            gather_act_kwargs["output_tensor"] = None
            gather_act_kwargs["output_sf_tensor"] = None
        else:
            gather_act_kwargs["swiglu_limit_scalar"] = self.swiglu_limit_scalar
        gather_act_kwargs["activation_type"] = self.activation_type
        if self.activation_type == ActivationType.SiTu:
            gather_act_kwargs["situ_beta"] = self.situ_beta
            gather_act_kwargs["situ_linear_beta"] = self.situ_linear_beta

        # Join the async input quantization (quantize_input_async): x/x_sf were
        # produced on the aux stream and are first read here, after moe_sort
        # ran on the main stream in parallel with the quantization.
        self._join_pending_x_ready()
        x, x_sf = gather_act_op(**gather_act_kwargs)

        if self.use_fused_finalize:
            if has_aux_streams:
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

        use_ugpu = self._ugpu_runtime is not None
        if use_ugpu:
            forward_impl = self._run_moe_bf16_ugpu
            workload_identity = self._ugpu_workload_identity(x.dtype)
            tuner_key = ("CuteDslFusedMoE::run_moe_bf16::ugpu_end_to_end")
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
            prime_inner_tactics=use_ugpu,
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

    def _run_moe_nvfp4_ugpu(
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
        """uGPU path: half-weight children, shared output buffers, fork/join.

        Each child holds localized half-N weights. Both partitions use the
        same tuned tactic and write directly into their strided regions of the
        shared FC1/FC2 output buffers.
        """
        output_dtype = torch.bfloat16
        num_partitions = self._ugpu_plan.num_partitions
        shards = self._ugpu_weight_shards
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
        #   c:    [permute_m, inner_size] — each uGPU writes half via strided layout
        #   c_sf: [full_sf_size] — each uGPU writes at K-tile offset via full_c_shape
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
            "uGPU MoE requires use_fused_finalize=True on Rubin")

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

        torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_ugpu_inplace_rubin(
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
            situ_beta=self.situ_beta
            if self.activation_type == ActivationType.SiTu else -1.0,
            situ_linear_beta=self.situ_linear_beta
            if self.activation_type == ActivationType.SiTu else -1.0,
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

        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_ugpu_inplace_rubin(
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

    def _run_moe_bf16_ugpu(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """uGPU path for unquantized BF16 MoE on Rubin."""
        output_dtype = x.dtype
        num_partitions = self._ugpu_plan.num_partitions
        shards = self._ugpu_weight_shards
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
            "uGPU MoE requires use_fused_finalize=True on Rubin")

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

        torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_ugpu_inplace_rubin(
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

        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_ugpu_inplace_rubin(
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
        x = swiglu_fused_moe(x, self.swiglu_limit_scalar)
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
                x_ready_event=ctx.x_ready_event,
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
        # later uGPU post_load_weights_impl splitting flow.
        dwdp_handle_collector = getattr(self, "dwdp_handle_collector", None)
        if dwdp_handle_collector is not None:
            dwdp_handle_collector.register_weights(self)

    def post_load_weights_impl(self):
        self.quant_method.post_load_weights(self)
        # Split full weights into per-partition halves on localized memory
        if self._ugpu_runtime is not None:
            self._ugpu_weight_shards = self._split_weights_for_ugpu()
            # Weight splitting initializes the process-lifetime uGPU resource.
            # Resolve the borrowed remainder stream now, never during capture.
            self._get_reserved_moe_output_memset_stream()
            self._release_full_weights_after_ugpu_split()

    def _release_full_weights_after_ugpu_split(self):
        """Release full tensors that are replaced by localized uGPU shards."""
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

    def _split_weights_for_ugpu(self):
        """Split full N-dimension weights into per-partition halves.

        After normal load_weights + post_load_weights, the full weights
        are on self. Split them along dim=1 (N) and allocate halves on
        each uGPU partition's localized memory.
        """
        num_p = self._ugpu_plan.num_partitions
        shards = []
        for pid in range(num_p):
            with self._ugpu_runtime.partition_weight_context(pid):
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
