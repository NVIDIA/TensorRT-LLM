# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import functools
import itertools
import math
from typing import List, Optional, Tuple, Type

import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.logger import logger

from ..._utils import get_sm_version, is_sm_100f
from ...math_utils import ceil_div, pad_up
from ..autotuner import (AutoTuner, ConstraintSpec, DistributedTuningStrategy,
                         DynamicTensorSpec, OptimizationProfile, TunableRunner,
                         TuningConfig)
from ..cute_dsl_utils import (IS_CUTLASS_DSL_AVAILABLE,
                              IS_CUTLASS_DSL_RUBIN_AVAILABLE)
from ..locality_domain.autotune import tune_locality_domain_concurrent
from ..locality_domain.runtime import LocalityDomainRuntime
from ..locality_domain_utils import (get_current_locality_domain,
                                     node_local_max_active_clusters)
from ..utils import (ActivationType, deep_gemm_gen_tuning_buckets,
                     fp4_scale_infer_shape, fp8_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     is_gated_activation, last_positive_power_of_2,
                     next_positive_power_of_2)
from .cutedsl_matmul_heuristics import (NVFP4_PRECISION,
                                        nvmmh_enabled_for_nvfp4, nvmmh_fields,
                                        nvmmh_max_tactics, rank_configs)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

# Torch schema parsing rejects ``inf`` as a default value.
SWIGLU_LIMIT_SCALAR_DISABLED = -1.0


def _with_input_cuda_device(function):
    """Run a custom-op implementation under its input tensor's CUDA device."""

    @functools.wraps(function)
    def wrapped(input, *args, **kwargs):
        with torch.cuda.device(input.device):
            return function(input, *args, **kwargs)

    return wrapped


def _validate_16_byte_aligned_dense_tensor(tensor: torch.Tensor,
                                           tensor_name: str) -> None:
    """Validate the pointer/strides required by dense CuTe TMA operands."""
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(
            f"{tensor_name} data pointer must be 16-byte aligned, got "
            f"data_ptr={tensor.data_ptr()}.")
    if tensor.shape[-1] > 1 and tensor.stride(-1) != 1:
        raise ValueError(
            f"{tensor_name} must have a contiguous innermost dimension, got "
            f"shape={tuple(tensor.shape)} and stride={tuple(tensor.stride())}.")
    for dim in range(tensor.dim() - 1):
        if (tensor.shape[dim] > 1
                and tensor.stride(dim) * tensor.element_size() % 16 != 0):
            raise ValueError(
                f"{tensor_name} stride in dimension {dim} must preserve "
                "16-byte alignment, got "
                f"shape={tuple(tensor.shape)} and stride={tuple(tensor.stride())}."
            )


def _canonicalize_swiglu_limit_scalar(swiglu_limit_scalar: float) -> float:
    return float("inf") if swiglu_limit_scalar < 0 else swiglu_limit_scalar


def _get_cute_dsl_swap_ab_candidates(
    m: int,
    output_aligned: bool,
    include_alternative: bool = False,
) -> List[bool]:
    """Return swap candidates in autotuning preference order.

    Both orientations write the same physical row-major [M, N] output. When
    swapping A and B, the kernel sees an [N, M] column-major view, so its
    contiguous C dimension is still the original N dimension. Therefore the
    16-byte output alignment requirement does not depend on logical M.

    Base kernels retain the existing M-based performance preference to bound
    autotuning cost. Mixed-cluster callers request the alternative orientation
    because cluster-grid feasibility depends on which logical axis becomes the
    kernel M dimension.
    """
    if not output_aligned:
        return []
    if m <= 128:
        swap_ab_candidates = [True]
    elif m >= 256:
        swap_ab_candidates = [False]
    else:
        swap_ab_candidates = [False, True]
    if include_alternative and len(swap_ab_candidates) == 1:
        swap_ab_candidates.append(not swap_ab_candidates[0])
    return swap_ab_candidates


def _get_sm107_nvfp4_default_mma_config(
    tile_size: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int]]:
    """Return the valid fallback MMA and cluster shapes for one routing tile."""
    mma_inst_m = min(tile_size, 256)
    return (
        (tile_size, 128, 256),
        (mma_inst_m, 128, 128),
        (mma_inst_m // 128, 1),
    )


class GroupedGemmInputsHelper:
    """Base helper class for grouped GEMM input preparation and tuning.

    Subclasses should override IDX_SHAPE_INFER to specify which input tensor
    to use for shape inference in tuning.
    """
    # Input tensor index for shape inference - subclass can override
    IDX_A = 0
    IDX_SHAPE_INFER = IDX_A  # Default: use a tensor for shape inference

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 num_local_experts: int,
                 local_expert_offset: int,
                 tile_size: int,
                 seed: int = 515):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.tile_size = tile_size
        self.seed = seed
        # Padding values should never be accessed.
        # Intentionally use a large padding value to expose issues early.
        self.pad_val = int(2e9)

    def get_max_num_tiles(self, num_tokens: int) -> int:
        num_expanded_tokens = num_tokens * self.top_k
        if num_expanded_tokens <= self.num_local_experts:
            return num_expanded_tokens
        return (num_expanded_tokens +
                (self.tile_size - 1) * self.num_local_experts) // self.tile_size

    def get_max_num_permuted_tokens(self, num_tokens: int) -> int:
        return self.get_max_num_tiles(num_tokens) * self.tile_size

    def infer_num_tokens(self, max_num_permuted_tokens: int) -> int:
        """Infer the maximum possible number of tokens given the max_num_permuted_tokens.
        """
        max_num_tiles = max_num_permuted_tokens // self.tile_size
        if max_num_tiles >= self.num_local_experts:
            return (max_num_permuted_tokens - (self.tile_size - 1) *
                    (self.num_local_experts - 1)) // self.top_k
        return max_num_tiles // self.top_k

    def gen_tuning_buckets(self, max_num_tokens: int) -> List[int]:
        buckets = get_last_power_of_2_num_tokens_buckets(
            self.infer_num_tokens(max_num_tokens))
        return sorted(
            list(set(self.get_max_num_permuted_tokens(x) for x in buckets)))

    def map_to_tuning_buckets(self, x: int) -> int:
        return self.get_max_num_permuted_tokens(
            last_positive_power_of_2(self.infer_num_tokens(x)))

    def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
        return self.infer_num_tokens(input_shapes[self.IDX_SHAPE_INFER][0])

    def infer_shape_max_num_tiles(self, input_shapes: List[torch.Size]) -> int:
        """Infer max_num_tiles from the shape inference tensor (IDX_SHAPE_INFER)."""
        return input_shapes[self.IDX_SHAPE_INFER][0] // self.tile_size

    def infer_shape_max_num_permuted_tokens(
            self, input_shapes: List[torch.Size]) -> int:
        return self.infer_shape_max_num_tiles(input_shapes) * self.tile_size

    def generate_num_tokens_per_expert(self,
                                       num_tokens: int,
                                       approx_max_load: bool = False
                                       ) -> List[int]:
        ep_size = self.num_experts // self.num_local_experts
        average_num_tokens_per_rank = num_tokens * self.top_k / ep_size

        if approx_max_load:
            # https://en.wikipedia.org/wiki/Balls_into_bins_problem
            # The constant c can be measured empirically, we choose 1.0 for simplicity.
            c = 1.0
            extra_num_tokens_on_curr_rank = c * math.sqrt(
                average_num_tokens_per_rank * math.log(ep_size))
            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank +
                                                extra_num_tokens_on_curr_rank)
        else:
            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank)

        num_tokens_on_curr_rank = min(num_tokens * self.top_k,
                                      num_tokens_on_curr_rank)

        base, remainder = divmod(num_tokens_on_curr_rank,
                                 self.num_local_experts)
        num_tokens_per_expert = [base + 1] * remainder + [base] * (
            self.num_local_experts - remainder)
        assert len(num_tokens_per_expert) == self.num_local_experts
        assert sum(num_tokens_per_expert) == num_tokens_on_curr_rank
        return num_tokens_per_expert

    def generate_token_selected_experts(
            self, num_tokens: int,
            num_tokens_per_expert: List[int]) -> torch.Tensor:
        """Balanced random based on rejection sampling.
        """
        token_selected_experts = -torch.ones(
            num_tokens, self.top_k, dtype=torch.int32)
        num_selected_experts = torch.zeros(num_tokens, dtype=torch.int32)

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(self.seed)
            selection_orders = [
                torch.randperm(num_tokens)
                for _ in range(self.num_local_experts)
            ]

        for j, num_tokens_j in enumerate(num_tokens_per_expert):
            selection_order_j = selection_orders[j].tolist()
            prioritized = torch.nonzero(num_selected_experts <= (
                self.top_k - (self.num_experts - j))).squeeze(-1).tolist()
            if len(prioritized) > 0:
                selection_order_j = prioritized + [
                    i for i in selection_order_j if i not in prioritized
                ]
            for i in selection_order_j:
                if num_selected_experts[i] < self.top_k:
                    token_selected_experts[
                        i,
                        num_selected_experts[i]] = j + self.local_expert_offset
                    num_selected_experts[i] += 1
                    num_tokens_j -= 1
                    if num_tokens_j <= 0:
                        break

        assert ((token_selected_experts
                 >= 0).sum(dim=-1) == num_selected_experts).all().item()
        if self.num_local_experts == self.num_experts:
            assert (num_selected_experts == self.top_k).all().item()
        else:
            assert (num_selected_experts <= self.top_k).all().item()
        return token_selected_experts

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles, *others = inputs
        num_tokens = self.infer_num_tokens(a.size(0))
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)
        token_selected_experts = self.generate_token_selected_experts(
            num_tokens, num_tokens_per_expert)

        token_selected_experts = token_selected_experts.cuda()
        token_final_scales = torch.ones_like(token_selected_experts,
                                             dtype=torch.float32)
        (
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_experts,
            top_k=self.top_k,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.num_local_experts,
            tile_tokens_dim=self.tile_size,
        )
        return a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles, *others

    def inputs_pre_hook_finalize_fusion(
            self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        a, b, a_sf, b_sf, alpha, output, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales = inputs
        num_tokens = self.infer_num_tokens(a.size(0))
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)
        token_selected_experts = self.generate_token_selected_experts(
            num_tokens, num_tokens_per_expert)

        token_selected_experts = token_selected_experts.cuda()
        token_final_scales = torch.ones_like(token_selected_experts,
                                             dtype=torch.float32)
        (
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_experts,
            top_k=self.top_k,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.num_local_experts,
            tile_tokens_dim=self.tile_size,
        )
        return a, b, a_sf, b_sf, alpha, output, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales


class GatherGroupedGemmInputsHelper(GroupedGemmInputsHelper):
    """Helper class for gather-based grouped GEMM input preparation.

    This subclass handles inputs where:
    - a tensor contains original (non-permuted) activations
    - permuted_idx_to_expanded_idx specifies the gather pattern
    - Shape inference uses permuted_idx_to_expanded_idx size instead of a size

    Input tensor layout:
        0: a                       - Original input activation (not permuted)
        1: b                       - Weight tensor
        2: a_sf                    - Scale factor for a
        3: b_sf                    - Scale factor for b
        4: alpha                   - Per-expert scaling factor
        5: tile_idx_to_group_idx   - Tile to expert mapping
        6: tile_idx_to_mn_limit    - Tile M/N limits
        7: permuted_idx_to_expanded_idx        - Token permutation mapping
        8: num_non_exiting_tiles   - Number of valid tiles
        9: global_sf               - Global scale factor
        10+: optional output tensors for inplace variants
    """
    # Override: use permuted_idx_to_expanded_idx for shape inference
    IDX_PERMUTED_IDX_TO_EXPANDED_IDX = 7
    IDX_SHAPE_INFER = IDX_PERMUTED_IDX_TO_EXPANDED_IDX

    @staticmethod
    def _resize_locality_domain_outputs(
        m: int,
        output_tensor: torch.Tensor,
        output_sf_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_m = output_tensor.size(0)
        assert output_m > 0
        assert output_sf_tensor.numel() % output_m == 0
        sf_size_per_m = output_sf_tensor.numel() // output_m
        if output_m == m:
            return output_tensor, output_sf_tensor
        return (output_tensor.new_empty((m, output_tensor.size(1))),
                output_sf_tensor.new_empty((m * sf_size_per_m, )))

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pre-hook for gather-based activation fusion kernel.

        Generates:
            - tile_idx_to_group_idx
            - tile_idx_to_mn_limit
            - permuted_idx_to_expanded_idx (for gather operation)
            - num_non_exiting_tiles

        Input layout:
            0: a                       - Original input activation (not permuted)
            1: b                       - Weight tensor
            2: a_sf                    - Scale factor for a
            3: b_sf                    - Scale factor for b
            4: alpha                   - Per-expert scaling factor
            5: tile_idx_to_group_idx   - Tile to expert mapping
            6: tile_idx_to_mn_limit    - Tile M/N limits
            7: permuted_idx_to_expanded_idx        - Token permutation mapping
            8: num_non_exiting_tiles   - Number of valid tiles
            9: global_sf               - Global scale factor
            10+: optional output tensors for inplace variants
        """
        a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, \
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf, *others = inputs
        # Verify permuted_idx_to_expanded_idx index matches the class constant
        assert inputs[
            self.
            IDX_PERMUTED_IDX_TO_EXPANDED_IDX] is permuted_idx_to_expanded_idx

        max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)
        num_tokens = self.infer_num_tokens(max_num_permuted_tokens)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)
        token_selected_experts = self.generate_token_selected_experts(
            num_tokens, num_tokens_per_expert)

        token_selected_experts = token_selected_experts.cuda()
        token_final_scales = torch.ones_like(token_selected_experts,
                                             dtype=torch.float32)
        (
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_experts,
            top_k=self.top_k,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.num_local_experts,
            tile_tokens_dim=self.tile_size,
        )
        if len(others) >= 2 and others[0] is not None and others[1] is not None:
            others = list(others)
            others[0], others[1] = self._resize_locality_domain_outputs(
                permuted_idx_to_expanded_idx.size(0), others[0], others[1])
        return (a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx,
                tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                num_non_exiting_tiles, global_sf, *others)


def get_dense_gemm_approximate_cta_nums(
        M: int, N: int, tile_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int]) -> int:
    tile_m, tile_n = tile_mn
    cluster_m, cluster_n = cluster_shape_mn
    clustered_ctas_m = pad_up(ceil_div(M, tile_m), cluster_m)
    clustered_ctas_n = pad_up(ceil_div(N, tile_n), cluster_n)
    return clustered_ctas_m * clustered_ctas_n


if IS_CUTLASS_DSL_AVAILABLE:

    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        BlockScaledContiguousGatherGroupedGemmKernel, validate_activation_type)
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm import \
        Sm100BlockScaledContiguousGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import \
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_swiglu_fusion import \
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
    from ..cute_dsl_kernels.blackwell.blockwise_gemm.blockwise_gemm import \
        Sm100BlockwiseGemmKernel
    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_act_fusion import \
        Sm100BlockScaledPersistentDenseGemmActFusionKernel
    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import \
        Sm100BlockScaledPersistentDenseGemmKernel
    from ..cute_dsl_kernels.blackwell.dense_gemm_persistent import \
        PersistentDenseGemmKernel
    from ..cute_dsl_kernels.blackwell.moe_as_dense_gemm.fc1 import \
        Sm100BlockScaledPersistentDenseGemmKernel as DenseGemmSwigluKernel
    from ..cute_dsl_kernels.blackwell.top_k.filtered_top_k_decode_varlen import \
        FilteredTopKKernelVarlenDecode
    from ..cute_dsl_kernels.blackwell.top_k.filtered_top_k_prefill_varlen import \
        FilteredTopKKernelVarlenPrefill
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import \
        STATE_SIZE as DISTRIBUTED_TOPK_STATE_SIZE
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import \
        SinglePassMultiCTARadixTopKKernel
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import \
        STATE_SIZE as CLUSTER_TOPK_STATE_SIZE
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import (
        SinglePassMultiCTARadixTopKClusterKernel, _query_max_cluster_size)
    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    @functools.cache
    def _get_full_device_max_active_clusters(device_id: int,
                                             cluster_size: int) -> int:
        """Return the cached full-device occupancy for a cluster shape."""
        hardware_info = cutlass.utils.HardwareInfo(device_id=device_id)
        return hardware_info.get_max_active_clusters(cluster_size)

    def get_max_activate_clusters(cluster_size):
        max_active = _get_full_device_max_active_clusters(
            torch.cuda.current_device(), cluster_size)
        if get_current_locality_domain() is not None:
            node_local = node_local_max_active_clusters(max_active)
            max_active = node_local if node_local is not None else max(
                1, max_active // 2)
        return max_active

    class CuteDSLNVFP4BlackwellRunner(TunableRunner):
        kernel_class = Sm100BlockScaledPersistentDenseGemmKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def __init__(self,
                     output_dtype: torch.dtype,
                     output_buffer_kind: int = int(BufferKind.DEFAULT),
                     group: Optional[List[int]] = None,
                     use_tvm_ffi: bool = True):
            super().__init__()

            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL NVFP4 only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.output_buffer_kind = int(output_buffer_kind)
            self.group = group
            self.use_tvm_ffi = use_tvm_ffi

        def unique_id(self):
            return (
                self.output_dtype,
                self.output_buffer_kind,
                tuple(self.group) if self.group is not None else None,
                self.use_tvm_ffi,
            )

        def __hash__(self):
            return hash(self.unique_id())

        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.unique_id() == other.unique_id()

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            # Early exit: Check SM version - CuteDSL NVFP4 only supports SM 100 and SM 103
            if (sm_version := get_sm_version()) not in (100, 103):
                logger.debug(
                    f"CuteDSL: SM version {sm_version} is not supported. "
                    f"CuteDSL NVFP4 only supports SM 100 (B200) and SM 103 (B300). Skipping all tactics."
                )
                return []

            assert inputs[0].dim() == 2
            assert inputs[1].dim() == 2

            m = inputs[0].shape[0]
            n = inputs[1].shape[0]
            k = inputs[0].shape[1]
            # Note: the input tensor use uint8 to store fp4, so the real_k is k * 2
            real_k = k * 2
            batch_size = 1
            sf_vec_size = 16

            # Fixed layout for FP4: A and B are always K-major
            a_major = "k"
            b_major = "k"

            # Early exit: Check K dimension alignment
            # For K-major layout (A and B tensors), K is the major mode (contiguous dimension).
            # 16-byte alignment requirement: K must be divisible by 32 for FP4 (128 bits / 4 bits = 32)
            if real_k % 32 != 0:
                logger.debug(
                    f"CuteDSL: K={real_k} does not meet 16-byte alignment requirement "
                    f"(K%32={real_k%32}, expected 0). Skipping all tactics.")
                return []

            # Both swap orientations use the original N as C's physical
            # contiguous dimension and require 16-byte BF16 alignment.
            output_aligned = (n % 8 == 0)
            if not output_aligned:
                logger.debug(
                    f"CuteDSL: Output N={n} does not meet the 16-byte "
                    f"alignment requirement (N%8={n%8}). Skipping all tactics.")
                return []

            swap_ab_candidates = _get_cute_dsl_swap_ab_candidates(
                m, output_aligned)
            if not swap_ab_candidates:
                logger.debug(f"CuteDSL: No valid C layout for M={m}, N={n}. "
                             "Skipping all tactics.")
                return []

            logger.debug(
                f"CuteDSL: M={m}, N={n}(aligned={output_aligned}), K={real_k}(aligned=True). "
                f"Using swap_ab={swap_ab_candidates}")

            # full shamoo
            mma_tiler_mn_candidates = [
                (128, 64),
                (256, 64),
                (128, 128),
                (256, 128),
                (128, 192),
                (256, 192),
                (128, 256),
                (256, 256),
            ]
            cluster_shape_mn_candidates = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            # prune: prefetch is beneficial only when K is large enough
            use_prefetch_candidates = [True, False]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates,
                    swap_ab_candidates, use_prefetch_candidates):
                if swap_ab:
                    c_major = "m"
                    kernel_m = n
                    kernel_n = m
                else:
                    c_major = "n"
                    kernel_m = m
                    kernel_n = n

                if self.__class__.kernel_class.can_implement(
                        cutlass.Float4E2M1FN,  # ab_dtype,
                        cutlass.Float8E4M3FN,  # sf_dtype
                        sf_vec_size,  # sf_vec_size,
                        cutlass.BFloat16,  # c_dtype,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        kernel_m,
                        kernel_n,
                        real_k,
                        batch_size,
                        a_major,
                        b_major,
                        c_major,
                ):
                    # Prefetch pruning to save tuning time
                    cta_nums = get_dense_gemm_approximate_cta_nums(
                        m, n, mma_tiler_mn, cluster_shape_mn)
                    cta_wave_ratio = cta_nums / torch.cuda.get_device_properties(
                    ).multi_processor_count
                    if use_prefetch and not any((
                            # CTA waves ratio between 0.5 and 1.0
                            0.5 < cta_wave_ratio < 1.0,
                            # K is large enough
                            real_k >= 8192,
                    )):
                        continue

                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch))

            logger.debug(
                f"CuteDSL: Found {len(valid_tactics)} valid tactics for M={m}, N={n}, K={real_k}"
            )
            # Optionally rank/prune the sweep with nvMatmulHeuristics (opt-in via
            # TRTLLM_CUTEDSL_NVMMH_ENABLE). Returns the full sweep unchanged when
            # disabled, unconfigured, or on any failure.
            return self._rank_prune_tactics(valid_tactics, m, n, real_k)

        def _swap_ab_candidates(self, m, n):
            """Deterministic swap_ab choice (not swept), constrained by C-layout
            alignment.

            swap_ab=True maps the kernel M to the GEMM N (and vice versa), which
            is preferred when M is small (<=128) so the kernel works on the
            larger dimension; large M does not swap. The chosen value must still
            satisfy the output (C) 16-byte alignment: swap_ab=False needs
            N%8==0, swap_ab=True needs M%8==0. Falls back to the feasible value
            if the preferred one violates alignment; returns [] if neither works.
            """
            m_aligned = m % 8 == 0
            n_aligned = n % 8 == 0
            prefer_swap = m <= 128
            if prefer_swap and m_aligned:
                return [True]
            if not prefer_swap and n_aligned:
                return [False]
            if n_aligned:
                return [False]
            if m_aligned:
                return [True]
            return []

        @staticmethod
        def _heuristic_to_tactic_tile(cta, cluster):
            """Translate a nvMatmulHeuristics (cta, cluster) config to this
            kernel's (mma_tiler_mn, cluster_shape_mn).

            nvMatmulHeuristics caps the per-CTA tile M at 128 on Blackwell and
            encodes the 2-SM (2-CTA) MMA as cluster_m == 2 (in the queried
            kernel frame), so its effective M tile is cluster_m * cta_m. This
            kernel instead encodes the same 2-SM op as mma_tiler_m == 256
            (use_2cta_instrs). So a 2-CTA config (cluster_m == 2) doubles the M
            tile while keeping the cluster; everything else passes through.

            The 2-SM op additionally requires the N tile to be aligned (the
            ``mma_n_align_requirement_2cta`` in libheuristics; 16 for NVFP4/bf16
            output, 32 when the N tile exceeds the 256 UTCMMA max). If N is not
            aligned it is not a valid 2-SM config, so leave it single-CTA.
            """
            cta_m, cta_n = int(cta[0]), int(cta[1])
            cluster_m, cluster_n = int(cluster[0]), int(cluster[1])
            n_align = 32 if cta_n > 256 else 16
            if cluster_m == 2 and cta_n % n_align == 0:
                # 2-SM MMA along the kernel M dimension.
                return (2 * cta_m, cta_n), (cluster_m, cluster_n)
            return (cta_m, cta_n), (cluster_m, cluster_n)

        @staticmethod
        def _unpack_tactic(tactic):
            """Unpack a tactic into the full knob set with back-compat defaults.

            The base tactic is (mma_tiler_mn, cluster_shape_mn, swap_ab,
            use_prefetch). The heuristic path may append two tile-scheduler
            knobs (swizzle_size, raster_along_m); older 4-tuples (and the full
            sweep) default to the kernel's neutral values (no swizzle, M-major
            raster). Non-tuple tactics use the default kernel tactic.
            """
            if not isinstance(tactic, (tuple, list)):
                return (128, 128), (1, 1), False, False, 1, True
            mma_tiler_mn = tactic[0]
            cluster_shape_mn = tactic[1]
            swap_ab = tactic[2]
            use_prefetch = tactic[3]
            swizzle_size = int(tactic[4]) if len(tactic) > 4 else 1
            raster_along_m = bool(tactic[5]) if len(tactic) > 5 else True
            return (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
                    swizzle_size, raster_along_m)

        def _tactic_is_supported(self, mma_tiler_mn, cluster_shape_mn, swap_ab,
                                 use_prefetch, m, n, real_k) -> bool:
            """Whether the kernel can run this (tile, cluster, swap, prefetch).

            Validity gate used by get_valid_tactics() (the full enumeration).
            The nvMatmulHeuristics path does NOT call this -- it trusts the
            model's mapped tiles and emits them directly.
            """
            sf_vec_size = 16
            if swap_ab:
                c_major, kernel_m, kernel_n = "m", n, m
            else:
                c_major, kernel_m, kernel_n = "n", m, n

            if not self.__class__.kernel_class.can_implement(
                    cutlass.Float4E2M1FN,  # ab_dtype
                    cutlass.Float8E4M3FN,  # sf_dtype
                    sf_vec_size,
                    cutlass.BFloat16,  # c_dtype
                    mma_tiler_mn,
                    cluster_shape_mn,
                    kernel_m,
                    kernel_n,
                    real_k,
                    1,  # batch_size
                    "k",  # a_major
                    "k",  # b_major
                    c_major,
            ):
                return False

            # Prefetch pruning: only worthwhile for a CTA-wave ratio in (0.5, 1.0)
            # or large K.
            cta_nums = get_dense_gemm_approximate_cta_nums(
                m, n, mma_tiler_mn, cluster_shape_mn)
            cta_wave_ratio = cta_nums / torch.cuda.get_device_properties(
            ).multi_processor_count
            if use_prefetch and not any(
                (0.5 < cta_wave_ratio < 1.0, real_k >= 8192)):
                return False
            return True

        def _rank_prune_tactics(self, tactics, m, n, real_k):
            """Rank/prune the full-sweep tactics with nvMatmulHeuristics (opt-in).

            Called at the end of get_valid_tactics. A strict, re-validated subset
            of ``tactics`` -- it never introduces a (tile, cluster, swap) the
            kernel validator rejected. Gated by TRTLLM_CUTEDSL_NVMMH_ENABLE;
            TRTLLM_CUTEDSL_NVMMH_FIELDS selects the model-driven knobs. When
            "tile"/"cluster" is selected we keep only the top
            TRTLLM_CUTEDSL_NVMMH_MAX_TACTICS ranked (mma_tiler, cluster, swap)
            keys (all matching prefetch variants retained); when only scheduler
            knobs (swizzle / cta_order) are selected we sweep every tile/cluster
            and just annotate it with the model's swizzle / raster (a safe
            6-tuple extension -- they do not affect can_implement). Deterministic
            swap_ab selection (small M swaps) is applied here, in the opt-in path
            only.

            Purely additive: on any failure, an empty match, or when heuristics
            are disabled / unconfigured, returns ``tactics`` unchanged so
            profiling never loses a valid candidate.
            """
            if not nvmmh_enabled_for_nvfp4() or not tactics:
                return tactics
            fields = nvmmh_fields()
            if not fields:
                return tactics
            try:
                max_tactics = nvmmh_max_tactics()
                # Only prune the tile/cluster set when the model is asked to
                # drive it; scheduler-only fields keep the full sweep.
                prune_tile_cluster = bool(fields & {"tile", "cluster"})
                use_swizzle = "swizzle" in fields
                use_cta_order = "cta_order" in fields
                emit_extended = use_swizzle or use_cta_order

                # Deterministic swap orientation (opt-in only), intersected with
                # the orientations actually present in the valid tactics.
                valid_swaps = {t[2] for t in tactics}
                swap_pref = [
                    s
                    for s in self._swap_ab_candidates(m, n) if s in valid_swaps
                ] or list(valid_swaps)

                # Build the model's ranked preference over (mma_tiler, cluster,
                # swap) keys, plus the per-key scheduler knobs. Query enough
                # configs to find matches within the valid candidate list.
                query_count = max(max_tactics * 4, 16)
                pref_rank = {}
                pref_sched = {}
                for swap_ab in swap_pref:
                    kernel_m, kernel_n = (n, m) if swap_ab else (m, n)
                    for cfg in rank_configs(kernel_m, kernel_n, real_k,
                                            NVFP4_PRECISION, query_count):
                        # Map per-CTA tile + cluster to this kernel's mma_tiler /
                        # cluster (cluster_m==2 encodes an mma_tiler_m==256 2-SM
                        # op). Off-grid maps simply won't match the valid list.
                        mma_tiler_mn, cluster_shape_mn = \
                            self._heuristic_to_tactic_tile(cfg.cta, cfg.cluster)
                        key = (mma_tiler_mn, cluster_shape_mn, swap_ab)
                        if key in pref_rank:
                            continue
                        pref_rank[key] = len(pref_rank)
                        swizzle_size = cfg.swizzle_factor if use_swizzle else 1
                        # nvMatmulHeuristics cta_order==0 is row-major (N-major)
                        # raster -> raster_along_m=False; !=0 -> M-major -> True.
                        raster_along_m = ((cfg.cta_order != 0)
                                          if use_cta_order else True)
                        pref_sched[key] = (max(1, int(swizzle_size)),
                                           bool(raster_along_m))

                # Restrict to the top-K ranked tile/cluster keys ONLY when the
                # model drives tile/cluster. For scheduler-only fields keep every
                # valid tile/cluster key and just annotate it below.
                valid_keys = {(t[0], t[1], t[2]) for t in tactics}
                if prune_tile_cluster:
                    matched_keys = valid_keys & pref_rank.keys()
                    if not matched_keys:
                        return tactics
                    kept_keys = set(
                        sorted(matched_keys,
                               key=lambda kk: pref_rank[kk])[:max_tactics])
                else:
                    kept_keys = valid_keys

                # Emit every supplied (already-valid) tactic whose key is kept,
                # re-validating as a safety net and optionally annotating the
                # model's swizzle / raster (neutral defaults for unranked keys).
                selected = []
                for t in tactics:
                    key = (t[0], t[1], t[2])
                    if key not in kept_keys:
                        continue
                    mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = t[:
                                                                              4]
                    if not self._tactic_is_supported(
                            mma_tiler_mn, cluster_shape_mn, swap_ab,
                            use_prefetch, m, n, real_k):
                        continue
                    if emit_extended:
                        swizzle_size, raster_along_m = pref_sched.get(
                            key, (1, True))
                        selected.append(
                            (mma_tiler_mn, cluster_shape_mn, swap_ab,
                             use_prefetch, swizzle_size, raster_along_m))
                    else:
                        selected.append((mma_tiler_mn, cluster_shape_mn,
                                         swap_ab, use_prefetch))

                logger.debug(
                    f"CuteDSL nvMatmulHeuristics: {len(tactics)} valid -> "
                    f"{len(selected)} tactics ({len(kept_keys)} keys) for "
                    f"M={m}, N={n}, K={real_k}; fields={sorted(fields)}")
                return selected if selected else tactics
            except Exception as e:  # noqa: BLE001 - must never break tuning
                logger.warning_once(
                    f"[nvMatmulHeuristics] NVFP4 tactic filtering failed: {e}. "
                    f"Falling back to full tactic list.",
                    key="nvmmh_nvfp4_filter_failure",
                )
                return tactics

        def should_profile_tactic_in_subprocess(
            self,
            custom_op: str,
            inputs: List[torch.Tensor],
            tactic,
            tuning_config: TuningConfig,
            **kwargs,
        ) -> bool:
            # get_valid_tactics emits 4 fields:
            # (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch).
            return isinstance(tactic, tuple) and len(tactic) == 4

        def make_cute_dsl_global_pointer(self, tensor: torch.Tensor, dtype,
                                         assumed_align: int):
            return make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
            bias: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            """
            Performs fp4 blockwise gemm operation using CuTe DSL.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (m, k), dtype: fp4.
                    inputs[1]: Weight tensor of shape (n, k), dtype: fp4.
                    inputs[2]: Input scale tensor of shape (k//16, m), dtype: fp8.
                    inputs[3]: Weight scale tensor of shape (n, k//16), dtype: fp8.
                    inputs[4]: Alpha scaling factor. dtype: float32.
                tactic: Tiling and cluster strategy, typically a tuple (mma_tiler_mn, cluster_shape_mn).
                bias: Optional per-N bias [N]. Added post-GEMM inside the
                    custom op (native CuTeDSL epilogue fusion is a follow-up).

            Returns:
                torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
            """
            sf_vec_size = 16

            (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
             swizzle_size, raster_along_m) = self._unpack_tactic(tactic)

            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]

            # Allocate output tensor based on output_buffer_kind.
            # allocate_output returns the actual BufferKind used (may fall back
            # to Default if NcclWindow allocation fails); we discard it here.
            c_tensor, _ = torch.ops.trtllm.allocate_output(
                a_tensor, self.output_buffer_kind, self.group, [m, n],
                self.output_dtype)

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)

            real_k = k * 2
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // sf_vec_size, 4)
            sf_n = pad_up(n, 128)

            # Reshape scale factors to CuteDSL's expected format
            # Input format (from CUTLASS/cuBLASLt): (m*k//16,) and (n*k//16,)
            # CuteDSL format: (sf_m*sf_k,) and (sf_n*sf_k,)
            # Note: This is just a view change, no memory copy
            expected_a_sf_size = sf_m * sf_k
            expected_b_sf_size = sf_n * sf_k

            if a_sf_tensor.numel() != expected_a_sf_size:
                raise ValueError(
                    f"CuteDSL: act scale factor size mismatch. "
                    f"Expected {expected_a_sf_size} (sf_m={sf_m} * sf_k={sf_k}), "
                    f"got {a_sf_tensor.numel()} for shape M={m}, K={real_k}")
            if b_sf_tensor.numel() != expected_b_sf_size:
                raise ValueError(
                    f"CuteDSL: weight scale factor size mismatch. "
                    f"Expected {expected_b_sf_size} (sf_n={sf_n} * sf_k={sf_k}), "
                    f"got {b_sf_tensor.numel()} for shape N={n}, K={real_k}")
            if alpha_tensor.numel() != 1:
                raise ValueError(f"CuteDSL: alpha size mismatch. "
                                 f"Expected 1, got {alpha_tensor.numel()}")

            # Reshape to CuteDSL's expected format (just a view, no copy)
            a_sf_tensor = a_sf_tensor.reshape(sf_m * sf_k)
            b_sf_tensor = b_sf_tensor.reshape(sf_n * sf_k)

            if not self.use_tvm_ffi:
                a_ptr = self.make_cute_dsl_global_pointer(
                    a_tensor, cutlass.Float4E2M1FN, 32)
                b_ptr = self.make_cute_dsl_global_pointer(
                    b_tensor, cutlass.Float4E2M1FN, 32)
                a_sf_ptr = self.make_cute_dsl_global_pointer(
                    a_sf_tensor, cutlass.Float8E4M3FN, 16)
                b_sf_ptr = self.make_cute_dsl_global_pointer(
                    b_sf_tensor, cutlass.Float8E4M3FN, 16)
                c_ptr = self.make_cute_dsl_global_pointer(
                    c_tensor, cutlass.BFloat16, 16)
                alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)

                # get stream
                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

            cache_key = (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab,
                         use_prefetch, swizzle_size, raster_along_m,
                         self.use_tvm_ffi)
            if swap_ab:
                kernel_m = n
                kernel_n = m
                kernel_sf_m = sf_n
                kernel_sf_n = sf_m

                kernel_a_tensor = b_tensor
                kernel_a_sf_tensor = b_sf_tensor
                kernel_b_tensor = a_tensor
                kernel_b_sf_tensor = a_sf_tensor

                if not self.use_tvm_ffi:
                    kernel_a_ptr = b_ptr
                    kernel_a_sf_ptr = b_sf_ptr
                    kernel_b_ptr = a_ptr
                    kernel_b_sf_ptr = a_sf_ptr
            else:
                kernel_m = m
                kernel_n = n
                kernel_sf_m = sf_m
                kernel_sf_n = sf_n

                kernel_a_tensor = a_tensor
                kernel_a_sf_tensor = a_sf_tensor
                kernel_b_tensor = b_tensor
                kernel_b_sf_tensor = b_sf_tensor

                if not self.use_tvm_ffi:
                    kernel_a_ptr = a_ptr
                    kernel_a_sf_ptr = a_sf_ptr
                    kernel_b_ptr = b_ptr
                    kernel_b_sf_ptr = b_sf_ptr

            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = self.make_cute_dsl_global_pointer(
                        a_tensor, cutlass.Float4E2M1FN, 32)
                    b_ptr = self.make_cute_dsl_global_pointer(
                        b_tensor, cutlass.Float4E2M1FN, 32)
                    a_sf_ptr = self.make_cute_dsl_global_pointer(
                        a_sf_tensor, cutlass.Float8E4M3FN, 16)
                    b_sf_ptr = self.make_cute_dsl_global_pointer(
                        b_sf_tensor, cutlass.Float8E4M3FN, 16)
                    c_ptr = self.make_cute_dsl_global_pointer(
                        c_tensor, cutlass.BFloat16, 16)
                    alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                    # make faked stream
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                    if swap_ab:
                        kernel_a_ptr = b_ptr
                        kernel_a_sf_ptr = b_sf_ptr
                        kernel_b_ptr = a_ptr
                        kernel_b_sf_ptr = a_sf_ptr
                    else:
                        kernel_a_ptr = a_ptr
                        kernel_a_sf_ptr = a_sf_ptr
                        kernel_b_ptr = b_ptr
                        kernel_b_sf_ptr = b_sf_ptr

                gemm = self.__class__.kernel_class(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_prefetch,
                    swizzle_size,
                    raster_along_m,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                # Note: when tvm_ffi fake stream is used, at least one parameter shoube be tensor type,
                # so we make alpha as the cute.Tensor type in the jit func.
                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    1,  # batch
                    kernel_a_ptr,
                    kernel_b_ptr,
                    kernel_a_sf_ptr,
                    kernel_b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                    max_active_clusters,
                    stream,
                    swap_ab,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # launch gemm kernel
            if self.use_tvm_ffi:
                # call with torch pointer types and no need to pass stream.
                compiled_gemm(
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    kernel_a_tensor.data_ptr(),
                    kernel_b_tensor.data_ptr(),
                    kernel_a_sf_tensor.data_ptr(),
                    kernel_b_sf_tensor.data_ptr(),
                    c_tensor.data_ptr(),
                    alpha_tensor,
                )
            else:
                # call with cute types and need to pass torch stream.
                compiled_gemm(
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    kernel_a_ptr,
                    kernel_b_ptr,
                    kernel_a_sf_ptr,
                    kernel_b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                    stream,
                )

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)
            if bias is not None:
                if bias.ndim != 1 or bias.shape[0] != c_tensor.shape[-1]:
                    raise ValueError(
                        f"bias must be a 1-D tensor of shape [N]={c_tensor.shape[-1]}, "
                        f"got shape {tuple(bias.shape)}")
                c_tensor = c_tensor + bias
            return c_tensor

    # a/b: fp4, scale: fp8, output: bf16
    @torch.library.custom_op("trtllm::cute_dsl_nvfp4_gemm_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_nvfp4_gemm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        output_buffer_kind: int = int(BufferKind.DEFAULT),
        group: Optional[List[int]] = None,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """CuteDSL-based NVFP4 GEMM optimized for Blackwell.

        Args:
            input: Activation tensor [m, k] in FP4 format (packed in uint8)
            weight: Weight tensor [n, k] in FP4 format (packed in uint8)
            input_scale: Activation scale factors
            weight_scale: Weight scale factors
            alpha: Scaling factor
            output_dtype: Output data type (must be bfloat16)
            output_buffer_kind: Output buffer allocation strategy (DEFAULT, USERBUFFERS, or NCCL_WINDOW)
            group: NCCL process group ranks (required when output_buffer_kind=NCCL_WINDOW)
            use_tvm_ffi: Whether to use TVM-FFI to call the kernel. Enable this option could help reduce the kernel host launch overhead.

        Note:
            This function is primarily used internally by nvfp4_gemm.
            Direct usage is discouraged. Consider using nvfp4_gemm instead
            for automatic backend selection with better performance.
        """
        # Validate SM version before attempting to use CuteDSL
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 backend requires SM 100 (B200) or SM 103 (B300), but got SM {sm_version}. "
                f"Please use nvfp4_gemm with backend='auto' for automatic backend selection."
            )

        tuner = AutoTuner.get()

        runner = CuteDSLNVFP4BlackwellRunner(output_dtype, output_buffer_kind,
                                             group, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_gemm_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )

        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake("trtllm::cute_dsl_nvfp4_gemm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,  # Match custom op signature
        output_dtype: torch.dtype,
        output_buffer_kind: int = int(BufferKind.DEFAULT),
        group: Optional[List[int]] = None,
        use_tvm_ffi: bool = True,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k]
        shape[-1] = mat_b.shape[-2]
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

    class CuteDSLNVFP4SwigluBlackwellRunner(TunableRunner):
        """Runner for dense GEMM + SwiGLU fusion on Blackwell GPUs using CuteDSL.

        Fuses the FC1 (gate_up projection) GEMM and SwiGLU activation into a
        single kernel for shared experts. The weight tensor has N columns
        (gate + up interleaved), and the output has N/2 columns after SwiGLU.
        """
        kernel_class = Sm100BlockScaledPersistentDenseGemmActFusionKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def __init__(self,
                     output_dtype: torch.dtype,
                     use_tvm_ffi: bool = True,
                     activation_type: ActivationType = ActivationType.Swiglu):
            super().__init__()

            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL NVFP4 SwiGLU only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.use_tvm_ffi = use_tvm_ffi
            self.activation_type = activation_type
            self.is_gated = is_gated_activation(activation_type)

        def unique_id(self):
            return (self.output_dtype, self.use_tvm_ffi, self.activation_type)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            # Early exit: Check SM version
            if (sm_version := get_sm_version()) not in (100, 103):
                logger.debug(
                    f"CuteDSL SwiGLU: SM version {sm_version} is not supported. "
                    f"CuteDSL NVFP4 SwiGLU only supports SM 100 (B200) and SM 103 (B300). Skipping all tactics."
                )
                return []

            assert inputs[0].dim() == 2
            assert inputs[1].dim() == 2

            m = inputs[0].shape[0]
            n = inputs[1].shape[0]  # Full B width (gate + up)
            k = inputs[0].shape[1]
            real_k = k * 2  # FP4 packed in uint8
            batch_size = 1
            sf_vec_size = 16

            # Fixed layout for FP4: A and B are always K-major
            a_major = "k"
            b_major = "k"

            # Early exit: Check K dimension alignment
            if real_k % 32 != 0:
                logger.debug(
                    f"CuteDSL SwiGLU: K={real_k} does not meet 16-byte alignment requirement "
                    f"(K%32={real_k%32}, expected 0). Skipping all tactics.")
                return []

            # Gated (SwiGLU) halves output N; non-gated (e.g. GELU) keeps full N.
            n_out = n // 2 if self.is_gated else n
            if n_out % 8 != 0:
                logger.debug(
                    f"CuteDSL SwiGLU: N_out={n_out} (N/2) does not meet 16-byte alignment "
                    f"(N_out%8={n_out%8}, expected 0 for BF16). Skipping all tactics."
                )
                return []

            # SwiGLU: swap_ab is not supported (SwiGLU operates on the N dimension)
            # C is always N-major
            c_major = "n"

            mma_tiler_mn_candidates = [
                (128, 128),
                (256, 128),
                (128, 256),
                (256, 256),
            ]
            cluster_shape_mn_candidates = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            use_prefetch_candidates = [True, False]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn, use_prefetch in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates,
                    use_prefetch_candidates):
                kernel_m = m
                kernel_n = n  # Full B width for can_implement

                if self.__class__.kernel_class.can_implement(
                        cutlass.Float4E2M1FN,  # ab_dtype
                        cutlass.Float8E4M3FN,  # sf_dtype
                        sf_vec_size,
                        cutlass.BFloat16,  # c_dtype
                        mma_tiler_mn,
                        cluster_shape_mn,
                        kernel_m,
                        kernel_n,
                        real_k,
                        batch_size,
                        a_major,
                        b_major,
                        c_major,
                ):
                    # Prefetch pruning
                    cta_nums = get_dense_gemm_approximate_cta_nums(
                        m, n, mma_tiler_mn, cluster_shape_mn)
                    cta_wave_ratio = cta_nums / torch.cuda.get_device_properties(
                    ).multi_processor_count
                    if use_prefetch and not any((
                            0.5 < cta_wave_ratio < 1.0,
                            real_k >= 8192,
                    )):
                        continue

                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, use_prefetch))

            logger.debug(
                f"CuteDSL SwiGLU: Found {len(valid_tactics)} valid tactics for M={m}, N={n}, K={real_k}"
            )
            return valid_tactics

        def make_cute_dsl_global_pointer(self, tensor: torch.Tensor, dtype,
                                         assumed_align: int):
            return make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
            **kwargs,
        ) -> torch.Tensor:
            """Performs fused FP4 dense GEMM + SwiGLU using CuTe DSL.

            The weight tensor has N columns (gate + up interleaved).
            The output has N/2 columns after SwiGLU fusion.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (m, k), dtype: fp4.
                    inputs[1]: Weight tensor of shape (n, k), dtype: fp4. n = 2 * intermediate_size.
                    inputs[2]: Input scale tensor, dtype: fp8.
                    inputs[3]: Weight scale tensor, dtype: fp8.
                    inputs[4]: Alpha scaling factor, dtype: float32.
                tactic: Tiling and cluster strategy tuple (mma_tiler_mn, cluster_shape_mn, use_prefetch).

            Returns:
                torch.Tensor: Output tensor of shape (m, n//2), dtype: bf16.
            """
            sf_vec_size = 16

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, use_prefetch = tactic
            else:
                mma_tiler_mn, cluster_shape_mn, use_prefetch = [
                    (128, 128),
                    (1, 1),
                    False,
                ]

            # Optional trailing per-N bias (non-gated GELU only). Default off.
            bias_tensor = inputs[5] if len(inputs) > 5 else None
            (a_tensor, b_tensor, a_sf_tensor, b_sf_tensor,
             alpha_tensor) = inputs[:5]
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
            # Gated (SwiGLU) halves output N; non-gated (e.g. GELU) keeps full N.
            n_out = n // 2 if self.is_gated else n

            # Bias is a per-N vector [n_out]; broadcast over M happens in the
            # kernel via a stride-0 layout. Require contiguous N for that.
            if bias_tensor is not None:
                if bias_tensor.numel() != n_out:
                    raise ValueError(
                        f"CuteDSL GELU: bias must have {n_out} elements "
                        f"(n_out), got {bias_tensor.numel()}")
                bias_tensor = bias_tensor.contiguous()

            # Allocate output tensor with the activation-adjusted N dimension
            c_tensor = torch.empty(*(m, n_out),
                                   dtype=self.output_dtype,
                                   device="cuda")

            real_k = k * 2
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // sf_vec_size, 4)
            sf_n = pad_up(n, 128)  # Scale factor is for full B width

            # Reshape scale factors to CuteDSL's expected format
            expected_a_sf_size = sf_m * sf_k
            expected_b_sf_size = sf_n * sf_k

            if a_sf_tensor.numel() != expected_a_sf_size:
                raise ValueError(
                    f"CuteDSL SwiGLU: act scale factor size mismatch. "
                    f"Expected {expected_a_sf_size} (sf_m={sf_m} * sf_k={sf_k}), "
                    f"got {a_sf_tensor.numel()} for shape M={m}, K={real_k}")
            if b_sf_tensor.numel() != expected_b_sf_size:
                raise ValueError(
                    f"CuteDSL SwiGLU: weight scale factor size mismatch. "
                    f"Expected {expected_b_sf_size} (sf_n={sf_n} * sf_k={sf_k}), "
                    f"got {b_sf_tensor.numel()} for shape N={n}, K={real_k}")
            if alpha_tensor.numel() != 1:
                raise ValueError(f"CuteDSL SwiGLU: alpha size mismatch. "
                                 f"Expected 1, got {alpha_tensor.numel()}")

            a_sf_tensor = a_sf_tensor.reshape(sf_m * sf_k)
            b_sf_tensor = b_sf_tensor.reshape(sf_n * sf_k)

            # Resolve optional bias dtype (bf16/fp32 accepted; consumed in fp32).
            has_bias = bias_tensor is not None
            if has_bias:
                if bias_tensor.dtype == torch.bfloat16:
                    bias_cute_dtype = cutlass.BFloat16
                elif bias_tensor.dtype == torch.float32:
                    bias_cute_dtype = cutlass.Float32
                else:
                    raise ValueError(
                        f"CuteDSL GELU: bias must be bf16 or fp32, "
                        f"got {bias_tensor.dtype}")

            if not self.use_tvm_ffi:
                a_ptr = self.make_cute_dsl_global_pointer(
                    a_tensor, cutlass.Float4E2M1FN, 32)
                b_ptr = self.make_cute_dsl_global_pointer(
                    b_tensor, cutlass.Float4E2M1FN, 32)
                a_sf_ptr = self.make_cute_dsl_global_pointer(
                    a_sf_tensor, cutlass.Float8E4M3FN, 16)
                b_sf_ptr = self.make_cute_dsl_global_pointer(
                    b_sf_tensor, cutlass.Float8E4M3FN, 16)
                c_ptr = self.make_cute_dsl_global_pointer(
                    c_tensor, cutlass.BFloat16, 16)
                bias_ptr = self.make_cute_dsl_global_pointer(
                    bias_tensor, bias_cute_dtype, 4) if has_bias else None
                alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

            # No swap_ab for SwiGLU — always use A as activation, B as weight
            kernel_m = m
            kernel_n = n  # Full B width (passed to wrapper, which creates B with n columns)
            kernel_sf_m = sf_m
            kernel_sf_n = sf_n

            # Cache key includes bias presence + dtype so the bias-free and
            # bias paths compile to distinct kernels (different host signature).
            bias_key = bias_tensor.dtype if has_bias else None
            cache_key = (sf_vec_size, mma_tiler_mn, cluster_shape_mn,
                         use_prefetch, self.use_tvm_ffi, self.activation_type,
                         bias_key)
            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = self.make_cute_dsl_global_pointer(
                        a_tensor, cutlass.Float4E2M1FN, 32)
                    b_ptr = self.make_cute_dsl_global_pointer(
                        b_tensor, cutlass.Float4E2M1FN, 32)
                    a_sf_ptr = self.make_cute_dsl_global_pointer(
                        a_sf_tensor, cutlass.Float8E4M3FN, 16)
                    b_sf_ptr = self.make_cute_dsl_global_pointer(
                        b_sf_tensor, cutlass.Float8E4M3FN, 16)
                    c_ptr = self.make_cute_dsl_global_pointer(
                        c_tensor, cutlass.BFloat16, 16)
                    bias_ptr = self.make_cute_dsl_global_pointer(
                        bias_tensor, bias_cute_dtype, 4) if has_bias else None
                    alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    True,  # vectorized_f32
                    use_prefetch,
                    activation_type=self.activation_type,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                # bias_ptr is the trailing keyword of wrapper; omit it entirely
                # when absent so the bias-free signature is unchanged.
                compile_args = [
                    gemm.wrapper,
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    1,  # batch
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                    max_active_clusters,
                    stream,
                    False,  # swap_ab=False for SwiGLU
                ]
                compile_kwargs = dict(
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2", )
                if has_bias:
                    compile_kwargs["bias_ptr"] = bias_ptr

                compiled_gemm = cute.compile(*compile_args, **compile_kwargs)

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Launch kernel
            if self.use_tvm_ffi:
                # bias data_ptr (when present) is the trailing dynamic arg,
                # mirroring the bias_ptr appended at compile time.
                tvm_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    a_tensor.data_ptr(),
                    b_tensor.data_ptr(),
                    a_sf_tensor.data_ptr(),
                    b_sf_tensor.data_ptr(),
                    c_tensor.data_ptr(),
                    alpha_tensor,
                ]
                if has_bias:
                    tvm_args.append(bias_tensor.data_ptr())
                compiled_gemm(*tvm_args)
            else:
                # bias_ptr is the trailing runtime arg of the compiled wrapper
                # (swap_ab/epilogue_op are constexprs baked in at compile time);
                # omit it entirely when absent.
                call_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                    stream,
                ]
                if has_bias:
                    call_args.append(bias_ptr)
                compiled_gemm(*call_args)

            return c_tensor

    # a/b: fp4, scale: fp8, output: bf16, fused SwiGLU activation
    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """CuteDSL-based NVFP4 dense GEMM with SwiGLU fusion for Blackwell.

        Fuses the FC1 (gate_up projection) GEMM and SwiGLU activation into a
        single kernel. Used for shared expert optimization.

        Args:
            input: Activation tensor [m, k] in FP4 format (packed in uint8)
            weight: Weight tensor [n, k] in FP4 format (packed in uint8).
                    n = 2 * intermediate_size (gate + up interleaved).
            input_scale: Activation scale factors
            weight_scale: Weight scale factors
            alpha: Scaling factor
            output_dtype: Output data type (must be bfloat16)
            use_tvm_ffi: Whether to use TVM-FFI for reduced host launch overhead.

        Returns:
            Output tensor [m, n//2] in bfloat16 after SwiGLU fusion.
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 SwiGLU backend requires SM 100 (B200) or SM 103 (B300), "
                f"but got SM {sm_version}.")

        tuner = AutoTuner.get()

        runner = CuteDSLNVFP4SwigluBlackwellRunner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )

        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        use_tvm_ffi: bool = True,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k] -> output has n//2 columns after SwiGLU
        shape[-1] = mat_b.shape[-2] // 2
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

    class CuteDSLNVFP4GeluBlackwellRunner(CuteDSLNVFP4SwigluBlackwellRunner):
        """Non-gated GELU(tanh) variant of the dense bf16-out runner.

        Reuses the swiglu runner's forward/get_valid_tactics; only the fused
        activation (GELU, non-gated -> output keeps full N) and the kernel
        compile cache differ.
        """
        kernel_cache = dict()

        def __init__(self, output_dtype: torch.dtype, use_tvm_ffi: bool = True):
            super().__init__(output_dtype,
                             use_tvm_ffi,
                             activation_type=ActivationType.Gelu)

        def unique_id(self):
            return (self.output_dtype, self.use_tvm_ffi, 'gelu')

    # a/b: fp4, scale: fp8, output: bf16, fused non-gated GELU(tanh)
    @torch.library.custom_op("trtllm::cute_dsl_nvfp4_dense_gemm_gelu_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_nvfp4_dense_gemm_gelu_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        """CuteDSL NVFP4 dense GEMM + non-gated GELU(tanh) with bf16 output for Blackwell.

        Non-gated counterpart of cute_dsl_nvfp4_dense_gemm_swiglu_blackwell:
        the output keeps the full N dimension (no gate/up halving), and the
        epilogue applies GELU(tanh) before writing bf16. Optionally adds a
        per-N bias (``gelu_tanh(alpha * acc + bias)``).

        Args:
            input: Activation tensor [m, k] in FP4 format (packed in uint8)
            weight: Weight tensor [n, k] in FP4 format (packed in uint8).
                    n = intermediate_size.
            input_scale: Activation scale factors
            weight_scale: Weight scale factors
            alpha: GEMM scaling factor
            output_dtype: Output data type (must be bfloat16)
            bias: Optional per-N bias vector [n] (bf16/fp32, NOT quantized),
                broadcast over M and added before GELU. None (default) -> no bias.
            use_tvm_ffi: Whether to use TVM-FFI for reduced host launch overhead.

        Returns:
            Output tensor [m, n] in bfloat16 after non-gated GELU(tanh).
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 GELU backend requires SM 100 (B200) or SM 103 (B300), "
                f"but got SM {sm_version}.")

        tuner = AutoTuner.get()

        runner = CuteDSLNVFP4GeluBlackwellRunner(output_dtype, use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        if bias is not None:
            inputs.append(bias)
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_gelu_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )

        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_gelu_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        use_tvm_ffi: bool = True,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k] -> non-gated GELU keeps the full N dimension
        shape[-1] = mat_b.shape[-2]
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

    class CuteDSLNVFP4SwigluFP4OutBlackwellRunner(TunableRunner):
        """Runner for dense GEMM + SwiGLU fusion with FP4 output on Blackwell.

        Same as CuteDSLNVFP4SwigluBlackwellRunner but produces Float4E2M1FN
        output with scale factors (SFC quantization), eliminating the bf16→fp4
        requantization between FC1 and FC2.
        """
        kernel_class = Sm100BlockScaledPersistentDenseGemmActFusionKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def __init__(self,
                     use_tvm_ffi: bool = True,
                     activation_type: ActivationType = ActivationType.Swiglu):
            super().__init__()
            self.use_tvm_ffi = use_tvm_ffi
            self.activation_type = activation_type
            self.is_gated = is_gated_activation(activation_type)

        def unique_id(self):
            return (self.use_tvm_ffi, 'fp4out')

        def make_cute_dsl_global_pointer(self, tensor: torch.Tensor, dtype,
                                         assumed_align: int):
            return make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple]:
            # Same tactic search as BF16 runner but with FP4 C dtype.
            # inputs may carry an optional trailing bias (non-gated GELU); only
            # the first 6 are needed for shape inference / tactic validity.
            a, b, a_sf, b_sf, alpha, global_sf = inputs[:6]
            m, k, n = a.shape[0], a.shape[1] * 2, b.shape[0]

            # The fp4out kernel's SFC epilogue does not properly predicate
            # writes when m < CTA tile height, causing OOB memory access.
            # Require m >= 128 (minimum MMA tile M dimension).
            if m < 128:
                return []

            sf_vec_size = 16
            # MMA tiler N restricted to 128/256 for SwiGLU
            mma_tiler_mn_candidates = [(128, 128), (128, 256), (256, 128),
                                       (256, 256)]
            cluster_shape_mn_candidates = [(1, 1), (2, 1), (1, 2), (2, 2)]
            use_prefetch_candidates = [True, False]

            valid_tactics = []
            for mma_tiler_mn in mma_tiler_mn_candidates:
                for cluster_shape_mn in cluster_shape_mn_candidates:
                    for use_prefetch in use_prefetch_candidates:
                        if self.__class__.kernel_class.can_implement(
                                ab_dtype=cutlass.Float4E2M1FN,
                                sf_dtype=cutlass.Float8E4M3FN,
                                sf_vec_size=sf_vec_size,
                                c_dtype=cutlass.Float4E2M1FN,
                                mma_tiler_mn=mma_tiler_mn,
                                cluster_shape_mn=cluster_shape_mn,
                                m=m,
                                n=n,
                                k=k,
                                l=1,
                                a_major="k",
                                b_major="k",
                                c_major="n",
                        ):
                            valid_tactics.append(
                                (mma_tiler_mn, cluster_shape_mn, use_prefetch))

            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Fused FP4 dense GEMM + SwiGLU with FP4 output + SFC.

            Args:
                inputs: [act_fp4, weight_fp4, act_sf, weight_sf, alpha, global_sf]
                    or, for the non-gated GELU path with bias, an extra trailing
                    [bias] (per-N vector, bf16/fp32, NOT quantized). Bias is only
                    consumed by the non-gated path; swiglu callers pass 6 inputs.
                tactic: (mma_tiler_mn, cluster_shape_mn, use_prefetch)

            Returns:
                (fp4_output, output_sf): FP4 output tensor and scale factors.
            """
            sf_vec_size = 16

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, use_prefetch = tactic
            else:
                mma_tiler_mn, cluster_shape_mn, use_prefetch = [
                    (128, 128),
                    (1, 1),
                    False,
                ]

            # Optional trailing per-N bias (non-gated GELU only). Default off.
            bias_tensor = inputs[6] if len(inputs) > 6 else None
            (a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor,
             global_sf_tensor) = inputs[:6]
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
            # Gated (SwiGLU) halves output N; non-gated (e.g. GELU) keeps full N.
            n_out = n // 2 if self.is_gated else n

            # Bias is a per-N vector [n_out]; broadcast over M happens in the
            # kernel via a stride-0 layout. Require contiguous N for that.
            if bias_tensor is not None:
                if bias_tensor.numel() != n_out:
                    raise ValueError(
                        f"CuteDSL GELU FP4Out: bias must have {n_out} elements "
                        f"(n_out), got {bias_tensor.numel()}")
                bias_tensor = bias_tensor.contiguous()

            # Pad m to CTA tile height to prevent OOB writes from
            # the kernel's epilogue on partial tiles.
            cta_m = mma_tiler_mn[0] * cluster_shape_mn[0]
            padded_m = pad_up(m, cta_m)

            # Allocate FP4 output with padded rows (kernel may write full tiles)
            c_tensor = torch.empty(padded_m,
                                   n_out // 2,
                                   dtype=a_tensor.dtype,
                                   device="cuda")

            real_k = k * 2

            # Scale factor dimensions (based on original m)
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // sf_vec_size, 4)
            sf_n = pad_up(n, 128)

            # SFC dimensions (based on original m — the kernel derives its
            # SFC layout from c_tensor.shape which uses m, not padded_m)
            sf_m_c = sf_m // 128
            sf_n_c = pad_up(n_out // sf_vec_size, 4) // 4

            # Allocate output scale factors with extra padding.
            # The kernel's SFC epilogue writes full-tile scale factors
            # including partial tiles that extend beyond m. The SFC layout
            # strides are based on sf_m (original m), but the epilogue may
            # write up to pad_up(padded_m, 128) // 128 blocks. The last
            # such write can land one element past the end of an sf_m-based
            # buffer. Use padded_m-based size to absorb these OOB writes.
            sf_m_sfc = pad_up(padded_m, 128)
            sf_n_cols = pad_up(n_out // sf_vec_size, 4)
            c_sf_tensor = torch.empty(sf_m_sfc * sf_n_cols,
                                      dtype=a_sf_tensor.dtype,
                                      device="cuda")

            # Validate input scale factor sizes
            expected_a_sf_size = sf_m * sf_k
            expected_b_sf_size = sf_n * sf_k

            if a_sf_tensor.numel() != expected_a_sf_size:
                raise ValueError(
                    f"CuteDSL SwiGLU FP4Out: act scale factor size mismatch. "
                    f"Expected {expected_a_sf_size}, got {a_sf_tensor.numel()}")
            if b_sf_tensor.numel() != expected_b_sf_size:
                raise ValueError(
                    f"CuteDSL SwiGLU FP4Out: weight scale factor size mismatch. "
                    f"Expected {expected_b_sf_size}, got {b_sf_tensor.numel()}")

            a_sf_tensor = a_sf_tensor.reshape(sf_m * sf_k)
            b_sf_tensor = b_sf_tensor.reshape(sf_n * sf_k)

            kernel_m = m
            kernel_n = n

            # Resolve optional bias dtype (bf16/fp32 accepted; consumed in fp32).
            has_bias = bias_tensor is not None
            if has_bias:
                if bias_tensor.dtype == torch.bfloat16:
                    bias_cute_dtype = cutlass.BFloat16
                elif bias_tensor.dtype == torch.float32:
                    bias_cute_dtype = cutlass.Float32
                else:
                    raise ValueError(
                        f"CuteDSL GELU FP4Out: bias must be bf16 or fp32, "
                        f"got {bias_tensor.dtype}")

            # Cache key includes bias presence + dtype so the bias-free and
            # bias paths compile to distinct kernels (different host signature).
            bias_key = bias_tensor.dtype if has_bias else None
            cache_key = (sf_vec_size, mma_tiler_mn, cluster_shape_mn,
                         use_prefetch, self.use_tvm_ffi, 'fp4out', bias_key)
            if cache_key not in self.__class__.kernel_cache:
                # Create pointers for compilation
                a_ptr = self.make_cute_dsl_global_pointer(
                    a_tensor, cutlass.Float4E2M1FN, 32)
                b_ptr = self.make_cute_dsl_global_pointer(
                    b_tensor, cutlass.Float4E2M1FN, 32)
                a_sf_ptr = self.make_cute_dsl_global_pointer(
                    a_sf_tensor, cutlass.Float8E4M3FN, 16)
                b_sf_ptr = self.make_cute_dsl_global_pointer(
                    b_sf_tensor, cutlass.Float8E4M3FN, 16)
                c_ptr = self.make_cute_dsl_global_pointer(
                    c_tensor, cutlass.Float4E2M1FN, 32)
                sfc_ptr = self.make_cute_dsl_global_pointer(
                    c_sf_tensor, cutlass.Float8E4M3FN, 16)
                bias_ptr = self.make_cute_dsl_global_pointer(
                    bias_tensor, bias_cute_dtype, 4) if has_bias else None
                alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                norm_const_cute_tensor = cute.runtime.from_dlpack(
                    global_sf_tensor)

                if self.use_tvm_ffi:
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)
                else:
                    torch_stream = torch.cuda.current_stream()
                    stream = cuda.CUstream(torch_stream.cuda_stream)

                gemm = self.__class__.kernel_class(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    True,  # vectorized_f32
                    use_prefetch,
                    activation_type=self.activation_type,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                # bias_ptr is the trailing positional of wrapper_fp4out; omit it
                # entirely when absent so the bias-free signature is unchanged.
                compile_args = [
                    gemm.wrapper_fp4out,
                    kernel_m,
                    kernel_n,
                    real_k,
                    sf_m // 128,
                    sf_n // 128,
                    sf_k // 4,
                    sf_m_c,
                    sf_n_c,
                    1,  # batch
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    sfc_ptr,
                    alpha_cute_tensor,
                    norm_const_cute_tensor,
                    max_active_clusters,
                    stream,
                ]
                if has_bias:
                    compile_args.append(bias_ptr)

                compiled_gemm = cute.compile(
                    *compile_args,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Launch kernel
            if self.use_tvm_ffi:
                # bias data_ptr (when present) is the trailing dynamic arg,
                # mirroring the bias_ptr appended at compile time.
                tvm_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    sf_m // 128,
                    sf_n // 128,
                    sf_k // 4,
                    sf_m_c,
                    sf_n_c,
                    a_tensor.data_ptr(),
                    b_tensor.data_ptr(),
                    a_sf_tensor.data_ptr(),
                    b_sf_tensor.data_ptr(),
                    c_tensor.data_ptr(),
                    c_sf_tensor.data_ptr(),
                    alpha_tensor,
                    global_sf_tensor,
                ]
                if has_bias:
                    tvm_args.append(bias_tensor.data_ptr())
                compiled_gemm(*tvm_args)
            else:
                a_ptr = self.make_cute_dsl_global_pointer(
                    a_tensor, cutlass.Float4E2M1FN, 32)
                b_ptr = self.make_cute_dsl_global_pointer(
                    b_tensor, cutlass.Float4E2M1FN, 32)
                a_sf_ptr = self.make_cute_dsl_global_pointer(
                    a_sf_tensor, cutlass.Float8E4M3FN, 16)
                b_sf_ptr = self.make_cute_dsl_global_pointer(
                    b_sf_tensor, cutlass.Float8E4M3FN, 16)
                c_ptr = self.make_cute_dsl_global_pointer(
                    c_tensor, cutlass.Float4E2M1FN, 32)
                sfc_ptr = self.make_cute_dsl_global_pointer(
                    c_sf_tensor, cutlass.Float8E4M3FN, 16)
                bias_ptr = self.make_cute_dsl_global_pointer(
                    bias_tensor, bias_cute_dtype, 4) if has_bias else None
                alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                norm_const_cute_tensor = cute.runtime.from_dlpack(
                    global_sf_tensor)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                # bias_ptr is the trailing positional of wrapper_fp4out (after
                # stream); omit it entirely when absent.
                call_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    sf_m // 128,
                    sf_n // 128,
                    sf_k // 4,
                    sf_m_c,
                    sf_n_c,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    sfc_ptr,
                    alpha_cute_tensor,
                    norm_const_cute_tensor,
                    stream,
                ]
                if has_bias:
                    call_args.append(bias_ptr)
                compiled_gemm(*call_args)

            # Trim padded C output back to original m rows
            c_tensor = c_tensor[:m]

            # Trim SFC buffer back to the size downstream expects.
            # The kernel wrote using sf_m-based strides; valid data
            # occupies the first sf_m * sf_n_cols elements. The extra
            # padding beyond that absorbed OOB SFC epilogue writes.
            expected_sf_size = sf_m * sf_n_cols
            if c_sf_tensor.numel() > expected_sf_size:
                c_sf_tensor = c_sf_tensor[:expected_sf_size]

            return c_tensor, c_sf_tensor

    # a/b: fp4, scale: fp8, output: fp4 + sfc, fused SwiGLU activation
    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        global_sf: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL-based NVFP4 dense GEMM + SwiGLU with FP4 output for Blackwell.

        Same as cute_dsl_nvfp4_dense_gemm_swiglu_blackwell but produces FP4
        output with scale factors, eliminating bf16→fp4 requantization.

        Args:
            input: Activation tensor [m, k] in FP4 format (packed)
            weight: Weight tensor [n, k] in FP4 format (packed).
                    n = 2 * intermediate_size (gate + up interleaved).
            input_scale: Activation scale factors
            weight_scale: Weight scale factors
            alpha: FC1 scaling factor
            global_sf: FC2 input scale (norm_const for SFC quantization)
            use_tvm_ffi: Whether to use TVM-FFI.

        Returns:
            Tuple of (fp4_output, output_sf):
                fp4_output: [m, n//4] in FP4 packed format
                output_sf: Scale factors for the output (1D)
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 SwiGLU FP4Out requires SM 100 or SM 103, "
                f"but got SM {sm_version}.")

        tuner = AutoTuner.get()

        runner = CuteDSLNVFP4SwigluFP4OutBlackwellRunner(use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha, global_sf]
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )

        return runner(inputs, tactic=best_tactic)

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        global_sf: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = mat_b.shape[-2]
        n_out = n // 2
        sf_vec_size = 16
        # FP4 output packed: [m, n_out // 2]. Use new_empty with the input
        # shape list so the SymInt for the token dim is preserved through the
        # FX graph (matches the BF16 / nvfp4_gemm fake patterns; a positional
        # torch.empty(m, ...) loses the SymInt link required by the piecewise
        # CUDA graph optimizer).
        fp4_shape = list(mat_a.shape)
        fp4_shape[-1] = n_out // 2
        fp4_output = mat_a.new_empty(fp4_shape)
        # Scale factors: 1D
        m = mat_a.shape[0]
        sf_size = pad_up(m, 128) * pad_up(n_out // sf_vec_size, 4)
        output_sf = input_scale.new_empty([sf_size])
        return fp4_output, output_sf

    class CuteDSLNVFP4GeluFP4OutBlackwellRunner(
            CuteDSLNVFP4SwigluFP4OutBlackwellRunner):
        """Non-gated GELU(tanh) variant of the dense FP4-out runner.

        Reuses the swiglu runner's forward/get_valid_tactics; only the fused
        activation (GELU, non-gated -> output keeps full N) and the kernel
        compile cache differ.
        """
        kernel_cache = dict()

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__(use_tvm_ffi, activation_type=ActivationType.Gelu)

        def unique_id(self):
            return (self.use_tvm_ffi, 'gelu_fp4out')

    # a/b: fp4, scale: fp8, output: fp4 + sfc, fused non-gated GELU(tanh)
    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        global_sf: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_tvm_ffi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL NVFP4 dense GEMM + non-gated GELU(tanh) with FP4 output for Blackwell.

        Non-gated counterpart of cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell:
        the output keeps the full N dimension (no gate/up halving), and the
        epilogue applies GELU(tanh) before FP4 quantization. Optionally adds a
        per-N bias (``gelu_tanh(alpha * acc + bias)``).

        Args:
            input: Activation tensor [m, k] in FP4 format (packed)
            weight: Weight tensor [n, k] in FP4 format (packed). n = intermediate_size.
            input_scale: Activation scale factors
            weight_scale: Weight scale factors
            alpha: GEMM scaling factor
            global_sf: Output scale (norm_const for SFC quantization)
            bias: Optional per-N bias vector [n] (bf16/fp32, NOT quantized),
                broadcast over M and added before GELU. None (default) -> no bias.
            use_tvm_ffi: Whether to use TVM-FFI.

        Returns:
            Tuple of (fp4_output, output_sf):
                fp4_output: [m, n//2] in FP4 packed format
                output_sf: Scale factors for the output (1D)
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 GELU FP4Out requires SM 100 or SM 103, "
                f"but got SM {sm_version}.")

        tuner = AutoTuner.get()

        runner = CuteDSLNVFP4GeluFP4OutBlackwellRunner(use_tvm_ffi)
        inputs = [input, weight, input_scale, weight_scale, alpha, global_sf]
        if bias is not None:
            inputs.append(bias)
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )

        return runner(inputs, tactic=best_tactic)

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_gelu_fp4out_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        global_sf: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_tvm_ffi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # bias does not change output shape.
        m = mat_a.shape[0]
        n = mat_b.shape[-2]
        n_out = n  # non-gated: output keeps full N
        sf_vec_size = 16
        # FP4 output packed: [m, n_out // 2]
        fp4_output = torch.empty(m,
                                 n_out // 2,
                                 dtype=mat_a.dtype,
                                 device=mat_a.device)
        # Scale factors: 1D
        sf_size = pad_up(m, 128) * pad_up(n_out // sf_vec_size, 4)
        output_sf = torch.empty(sf_size,
                                dtype=input_scale.dtype,
                                device=input_scale.device)
        return fp4_output, output_sf

    class Sm100BlockScaledContiguousGroupedGemmRunner(TunableRunner):
        kernel_class = Sm100BlockScaledContiguousGroupedGemmKernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     tile_size: int,
                     output_dtype: torch.dtype,
                     scaling_vector_size: int = 16):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size

            assert output_dtype == torch.bfloat16
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size

            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}"
                )

            if self.tile_size not in (128, 256):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}"
                )

        def unique_id(self):
            return (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
                self.output_dtype,
                self.scaling_vector_size,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            a, b, *_ = inputs
            b_list = b if isinstance(b, (list, tuple)) else [b]
            m, k = a.size(0), a.size(1) * 2
            l = sum(bi.size(0) for bi in b_list)  # noqa: E741
            n = b_list[0].size(1)

            mma_tiler_mn_candidates = [(self.tile_size, 128),
                                       (self.tile_size, 256)]
            cluster_shape_mn_candidates = [(self.tile_size // 128, 1),
                                           (self.tile_size // 128, 2)]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
                # Skip tactics where the cluster shape exceeds available
                # tiles. Launching more cluster CTAs than tiles causes
                # out-of-bounds memory access in the CuteDSL kernel.
                if (ceil_div(m, mma_tiler_mn[0]) < cluster_shape_mn[0]
                        or ceil_div(n, mma_tiler_mn[1]) < cluster_shape_mn[1]):
                    continue
                if self.__class__.kernel_class.can_implement(
                        ab_dtype=cutlass.Float4E2M1FN,
                        sf_dtype=cutlass.Float8E4M3FN,
                        sf_vec_size=self.scaling_vector_size,
                        c_dtype=cutlass.BFloat16,
                        mma_tiler_mn=mma_tiler_mn,
                        cluster_shape_mn=cluster_shape_mn,
                        m=m,
                        n=n,
                        k=k,
                        l=l,
                        a_major="k",
                        b_major="k",
                        c_major="n",
                ):
                    valid_tactics.append((mma_tiler_mn, cluster_shape_mn))

            return valid_tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = GroupedGemmInputsHelper(self.num_experts, self.top_k,
                                                 self.num_local_experts,
                                                 self.local_expert_offset,
                                                 self.tile_size)
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, helper.gen_tuning_buckets,
                        helper.map_to_tuning_buckets), ),
                    constraint_specs=(ConstraintSpec(2, 0,
                                                     fp4_scale_infer_shape),
                                      ConstraintSpec(
                                          5, 0,
                                          helper.infer_shape_max_num_tiles)),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(self, inputs: List[torch.Tensor],
                    tactic: Optional[tuple]) -> torch.Tensor:
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles = inputs
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
            l, n = b.size(0), b.size(1)  # noqa: E741
            scale_k = k // self.scaling_vector_size
            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert b.size(2) * 2 == k
            assert a_sf.size(0) == m * scale_k
            assert b_sf.size(0) == l
            assert b_sf.size(1) == n
            assert b_sf.size(2) == scale_k
            assert alpha.size(0) == l

            num_tiles = m // self.tile_size
            assert tile_idx_to_group_idx.dtype == torch.int32
            assert tile_idx_to_group_idx.size() == (num_tiles, )
            assert num_non_exiting_tiles.dtype == torch.int32
            assert num_non_exiting_tiles.numel() == 1

            c = torch.empty(m, n, dtype=self.output_dtype, device=a.device)

            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                 cute.AddressSpace.gmem)
            tile_idx_to_group_idx_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                cute.AddressSpace.gmem)
            num_non_exiting_tiles_ptr = make_ptr(
                cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                cute.AddressSpace.gmem)
            c_ptr = make_ptr(cutlass.BFloat16,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=16)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn = tactic
            else:
                mma_tiler_mn = (self.tile_size, 128)
                cluster_shape_mn = (self.tile_size // 128, 1)
            assert mma_tiler_mn[
                0] == self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

            cache_key = (self.scaling_vector_size, self.tile_size, mma_tiler_mn,
                         cluster_shape_mn)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    m,
                    n,
                    k,
                    l,
                    tile_size=self.tile_size,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            compiled_gemm(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                alpha_ptr,
                tile_idx_to_group_idx_ptr,
                num_non_exiting_tiles_ptr,
                m,
                n,
                k,
                l,
                stream=stream,
            )
            return c

    @torch.library.custom_op("trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_nvfp4_grouped_gemm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        tuner = AutoTuner.get()

        runner = Sm100BlockScaledContiguousGroupedGemmRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, output_dtype, scaling_vector_size)
        inputs = [
            input, weight, input_scale, weight_scale, alpha,
            tile_idx_to_group_idx, num_non_exiting_tiles
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        m = input.size(0)
        n = weight.size(1)
        return torch.empty(m, n, dtype=output_dtype, device=input.device)

    class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
            TunableRunner):
        kernel_class = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     tile_size: int,
                     output_dtype: torch.dtype,
                     scaling_vector_size: int = 16):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size

            assert output_dtype == torch.bfloat16
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size

            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}"
                )

            if self.tile_size not in (128, 256):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}"
                )

        def unique_id(self):
            return (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
                self.output_dtype,
                self.scaling_vector_size,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            a, b, *_ = inputs
            m, k = a.size(0), a.size(1) * 2
            l, n = b.size(0), b.size(1)  # noqa: E741

            mma_tiler_mn_candidates = [(self.tile_size, 128),
                                       (self.tile_size, 256)]
            cluster_shape_mn_candidates = [(self.tile_size // 128, 1),
                                           (self.tile_size // 128, 2)]
            # raster_along_m=False should be theoretically more performant than raster_along_m=True.
            # TODO: Add raster_along_m=True if we find it more performant in some cases.
            raster_along_m_candidates = [False]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates,
                    raster_along_m_candidates):
                # Skip tactics where the cluster shape exceeds available
                # tiles. Launching more cluster CTAs than tiles causes
                # out-of-bounds memory access in the CuteDSL kernel.
                if (ceil_div(m, mma_tiler_mn[0]) < cluster_shape_mn[0]
                        or ceil_div(n, mma_tiler_mn[1]) < cluster_shape_mn[1]):
                    continue
                if self.__class__.kernel_class.can_implement(
                        ab_dtype=cutlass.Float4E2M1FN,
                        sf_dtype=cutlass.Float8E4M3FN,
                        sf_vec_size=self.scaling_vector_size,
                        out_dtype=cutlass.BFloat16,
                        mma_tiler_mn=mma_tiler_mn,
                        cluster_shape_mn=cluster_shape_mn,
                        m=m,
                        n=n,
                        k=k,
                        l=l,
                        a_major="k",
                        b_major="k",
                        out_major="n",
                ):
                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, raster_along_m))

            return valid_tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = GroupedGemmInputsHelper(self.num_experts, self.top_k,
                                                 self.num_local_experts,
                                                 self.local_expert_offset,
                                                 self.tile_size)
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, helper.gen_tuning_buckets,
                        helper.map_to_tuning_buckets), ),
                    constraint_specs=(
                        ConstraintSpec(2, 0, fp4_scale_infer_shape),
                        ConstraintSpec(5, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(6, 0, helper.infer_shape_max_num_tiles),
                        ConstraintSpec(7, 0, helper.infer_shape_max_num_tiles),
                        ConstraintSpec(
                            8, 0, helper.infer_shape_max_num_permuted_tokens),
                        ConstraintSpec(10, 0, helper.infer_shape_num_tokens)),
                    inputs_pre_hook=helper.inputs_pre_hook_finalize_fusion,
                    use_cold_l2_cache=True,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(self, inputs: List[torch.Tensor],
                    tactic: Optional[tuple]) -> torch.Tensor:
            a, b, a_sf, b_sf, alpha, c, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales = inputs
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
            l, n = b.size(0), b.size(1)  # noqa: E741
            scale_k = k // self.scaling_vector_size
            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert b.size(2) * 2 == k
            assert a_sf.size(0) == m * scale_k
            assert b_sf.size(0) == l
            assert b_sf.size(1) == n
            assert b_sf.size(2) == scale_k
            assert alpha.size(0) == l

            assert c.dtype == self.output_dtype
            assert c.dim() == 2
            num_tokens = c.size(0)
            assert c.size(1) == n

            num_tiles = m // self.tile_size
            assert tile_idx_to_group_idx.dtype == torch.int32
            assert tile_idx_to_group_idx.size() == (num_tiles, )
            assert tile_idx_to_mn_limit.dtype == torch.int32
            assert tile_idx_to_mn_limit.size() == (num_tiles, )
            assert permuted_idx_to_expanded_idx.dtype == torch.int32
            assert permuted_idx_to_expanded_idx.size() == (m, )
            assert num_non_exiting_tiles.dtype == torch.int32
            assert num_non_exiting_tiles.numel() == 1
            assert token_final_scales.dtype == torch.float32
            assert token_final_scales.dim() == 2
            assert token_final_scales.size() == (num_tokens, self.top_k)

            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                 cute.AddressSpace.gmem)
            tile_idx_to_group_idx_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                cute.AddressSpace.gmem)
            tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32,
                                                tile_idx_to_mn_limit.data_ptr(),
                                                cute.AddressSpace.gmem)
            permuted_idx_to_expanded_idx_ptr = make_ptr(
                cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                cute.AddressSpace.gmem)
            num_non_exiting_tiles_ptr = make_ptr(
                cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                cute.AddressSpace.gmem)
            token_final_scales_ptr = make_ptr(cutlass.Float32,
                                              token_final_scales.data_ptr(),
                                              cute.AddressSpace.gmem)
            c_ptr = make_ptr(cutlass.BFloat16,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=16)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
            else:
                mma_tiler_mn = (self.tile_size, 128)
                cluster_shape_mn = (self.tile_size // 128, 1)
                raster_along_m = False
            assert mma_tiler_mn[
                0] == self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

            cache_key = (self.scaling_vector_size, self.tile_size, mma_tiler_mn,
                         cluster_shape_mn, raster_along_m)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    raster_along_m=raster_along_m,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compile_args = [
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    token_final_scales_ptr,
                    m,
                    n,
                    k,
                    l,
                    num_tokens,
                    self.top_k,
                ]

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    *compile_args,
                    tile_size=self.tile_size,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            exec_args = [
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                alpha_ptr,
                tile_idx_to_group_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                permuted_idx_to_expanded_idx_ptr,
                num_non_exiting_tiles_ptr,
                token_final_scales_ptr,
                m,
                n,
                k,
                l,
                num_tokens,
                self.top_k,
            ]
            compiled_gemm(*exec_args, stream=stream)
            return c

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
        mutates_args=("output", ),
        device_types="cuda")
    def cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> None:
        tuner = AutoTuner.get()

        runner = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, output_dtype, scaling_vector_size)

        inputs = [
            input, weight, input_scale, weight_scale, alpha, output,
            tile_idx_to_group_idx, tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx, num_non_exiting_tiles,
            token_final_scales
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        runner(inputs, tactic=best_tactic)

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_grouped_gemm_finalize_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        num_tokens = token_final_scales.size(0)
        n = weight.size(1)
        output = torch.zeros(num_tokens,
                             n,
                             dtype=output_dtype,
                             device=input.device)
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            output=output,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            output_dtype=output_dtype,
            scaling_vector_size=scaling_vector_size,
        )
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        num_tokens = token_final_scales.size(0)
        n = weight.size(1)
        return torch.empty(num_tokens,
                           n,
                           dtype=output_dtype,
                           device=input.device)

    class Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner(
            TunableRunner):
        kernel_class = Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     tile_size: int,
                     scaling_vector_size: int = 16,
                     swiglu_limit_scalar: float = float("inf")):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size
            self.scaling_vector_size = scaling_vector_size
            self.swiglu_limit_scalar = swiglu_limit_scalar

            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}"
                )

            if self.tile_size not in (128, 256):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}"
                )

        def unique_id(self):
            return (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
                self.scaling_vector_size,
                self.swiglu_limit_scalar,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            a, b, *_ = inputs
            m, k = a.size(0), a.size(1) * 2
            l, n = b.size(0), b.size(1)  # noqa: E741

            mma_tiler_mn_candidates = [(self.tile_size, 128),
                                       (self.tile_size, 256)]
            cluster_shape_mn_candidates = [(self.tile_size // 128, 1),
                                           (self.tile_size // 128, 2)]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
                # Skip tactics where the cluster shape exceeds available
                # tiles. Launching more cluster CTAs than tiles causes
                # out-of-bounds memory access in the CuteDSL kernel.
                if (ceil_div(m, mma_tiler_mn[0]) < cluster_shape_mn[0]
                        or ceil_div(n, mma_tiler_mn[1]) < cluster_shape_mn[1]):
                    continue
                if self.__class__.kernel_class.can_implement(
                        ab_dtype=cutlass.Float4E2M1FN,
                        sf_dtype=cutlass.Float8E4M3FN,
                        sf_vec_size=self.scaling_vector_size,
                        c_dtype=cutlass.Float4E2M1FN,
                        mma_tiler_mn=mma_tiler_mn,
                        cluster_shape_mn=cluster_shape_mn,
                        m=m,
                        n=n,
                        k=k,
                        l=l,
                        a_major="k",
                        b_major="k",
                        c_major="n",
                ):
                    valid_tactics.append((mma_tiler_mn, cluster_shape_mn))

            return valid_tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = GroupedGemmInputsHelper(self.num_experts, self.top_k,
                                                 self.num_local_experts,
                                                 self.local_expert_offset,
                                                 self.tile_size)
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, helper.gen_tuning_buckets,
                        helper.map_to_tuning_buckets), ),
                    constraint_specs=(ConstraintSpec(2, 0,
                                                     fp4_scale_infer_shape),
                                      ConstraintSpec(
                                          5, 0,
                                          helper.infer_shape_max_num_tiles)),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(self, inputs: List[torch.Tensor],
                    tactic: Optional[tuple]) -> torch.Tensor:
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles, global_sf = inputs
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
            l, n = b.size(0), b.size(1)  # noqa: E741
            scale_k = k // self.scaling_vector_size
            interm_size = n // 2
            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert n % (self.scaling_vector_size * 4 * 2) == 0
            assert b.size(2) * 2 == k
            assert a_sf.size(0) == m * scale_k
            assert b_sf.size(0) == l
            assert b_sf.size(1) == n
            assert b_sf.size(2) == scale_k
            assert alpha.size(0) == l

            num_tiles = m // self.tile_size
            assert tile_idx_to_group_idx.dtype == torch.int32
            assert tile_idx_to_group_idx.size() == (num_tiles, )
            assert num_non_exiting_tiles.dtype == torch.int32
            assert num_non_exiting_tiles.numel() == 1
            assert global_sf.dtype == torch.float32
            assert global_sf.numel() == 1

            c = torch.empty(m, interm_size // 2, dtype=a.dtype, device=a.device)
            c_sf = torch.empty(m * interm_size // self.scaling_vector_size,
                               dtype=a_sf.dtype,
                               device=a_sf.device)

            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                 cute.AddressSpace.gmem)
            tile_idx_to_group_idx_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                cute.AddressSpace.gmem)
            num_non_exiting_tiles_ptr = make_ptr(
                cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                cute.AddressSpace.gmem)
            global_sf_ptr = make_ptr(cutlass.Float32, global_sf.data_ptr(),
                                     cute.AddressSpace.gmem)
            c_ptr = make_ptr(cutlass.Float4E2M1FN,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            c_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                c_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn = tactic
            else:
                mma_tiler_mn = (self.tile_size, 128)
                cluster_shape_mn = (self.tile_size // 128, 1)
            assert mma_tiler_mn[
                0] == self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

            cache_key = (self.scaling_vector_size, self.tile_size, mma_tiler_mn,
                         cluster_shape_mn, self.swiglu_limit_scalar)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(self.scaling_vector_size,
                                                   mma_tiler_mn,
                                                   cluster_shape_mn, True,
                                                   self.swiglu_limit_scalar)
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    c_sf_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    global_sf_ptr,
                    m,
                    n,
                    k,
                    l,
                    tile_size=self.tile_size,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            compiled_gemm(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                c_sf_ptr,
                alpha_ptr,
                tile_idx_to_group_idx_ptr,
                num_non_exiting_tiles_ptr,
                global_sf_ptr,
                m,
                n,
                k,
                l,
                stream=stream,
            )
            return c, c_sf

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_swiglu_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_grouped_gemm_swiglu_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tuner = AutoTuner.get()
        swiglu_limit_scalar = _canonicalize_swiglu_limit_scalar(
            swiglu_limit_scalar)

        runner = Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, scaling_vector_size, swiglu_limit_scalar)
        inputs = [
            input, weight, input_scale, weight_scale, alpha,
            tile_idx_to_group_idx, num_non_exiting_tiles, global_sf
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_swiglu_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_grouped_gemm_swiglu_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = input.size(0)
        n = weight.size(1)
        interm_size = n // 2
        output = torch.empty(m,
                             interm_size // 2,
                             dtype=input.dtype,
                             device=input.device)
        output_scale = torch.empty(m * interm_size // scaling_vector_size,
                                   dtype=input_scale.dtype,
                                   device=input_scale.device)
        return output, output_scale

    class Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner(
            TunableRunner):
        kernel_class = BlockScaledContiguousGatherGroupedGemmKernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     tile_size: int,
                     scaling_vector_size: int = 16,
                     activation_type: ActivationType = ActivationType.Swiglu,
                     swiglu_limit_scalar: float = float("inf")):
            """Initialize the runner.

            Args:
                activation_type: ``ActivationType`` for the fused epilogue. Only
                    ``Swiglu`` (gated) and ``Relu2`` (non-gated) are supported.
                swiglu_limit_scalar: Uniform clamp limit for SwiGLU. ``+inf`` disables clamp.
            """
            super().__init__()
            self.activation_type = validate_activation_type(activation_type)
            self.is_gated = is_gated_activation(self.activation_type)
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            if tile_size not in [128, 256]:
                raise ValueError(
                    f"Tile size {tile_size} is not supported, it only supports 128 and 256."
                )
            self.tile_size = tile_size
            self.scaling_vector_size = scaling_vector_size
            self.swiglu_limit_scalar = swiglu_limit_scalar

            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}"
                )

            if self.tile_size not in (128, 256):
                raise ValueError(
                    f"{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}"
                )

        def unique_id(self):
            return (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
                self.scaling_vector_size,
                self.activation_type,
                self.swiglu_limit_scalar,
            )

        def get_valid_tactics(
            self,
            inputs: List,
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            # Tuning uses layout: a, b, a_sf, b_sf, alpha, ...
            a = inputs[0]
            b = inputs[1]
            permuted_idx_to_expanded_idx = inputs[7]
            # m is the permuted size from permuted_idx_to_expanded_idx, not from a
            m = permuted_idx_to_expanded_idx.size(0)
            k = a.size(1) * 2
            l, n = b.size(0), b.size(1)  # noqa: E741

            mma_tiler_mn_candidates = [(self.tile_size, 128),
                                       (self.tile_size, 256)]
            cluster_shape_mn_candidates = [(self.tile_size // 128, 1)]
            # TODO: Add raster_along_m=True if we find it more performant in some cases.
            raster_along_m_candidates = [False]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates,
                    raster_along_m_candidates):
                if self.__class__.kernel_class.can_implement(
                        ab_dtype=cutlass.Float4E2M1FN,
                        sf_dtype=cutlass.Float8E4M3FN,
                        sf_vec_size=self.scaling_vector_size,
                        c_dtype=cutlass.Float4E2M1FN,
                        mma_tiler_mn=mma_tiler_mn,
                        cluster_shape_mn=cluster_shape_mn,
                        m=m,
                        n=n,
                        k=k,
                        l=l,
                        a_major="k",
                        b_major="k",
                        c_major="n",
                ):
                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, raster_along_m))

            return valid_tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = GatherGroupedGemmInputsHelper(self.num_experts,
                                                       self.top_k,
                                                       self.num_local_experts,
                                                       self.local_expert_offset,
                                                       self.tile_size)
                # Tuning uses layout:
                # a, b, a_sf, b_sf, alpha, tile_idx, tile_mn_limit, permuted_idx, ...
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    # Use permuted_idx_to_expanded_idx (IDX_SHAPE_INFER) for tuning
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        GatherGroupedGemmInputsHelper.IDX_SHAPE_INFER, 0,
                        helper.gen_tuning_buckets,
                        helper.map_to_tuning_buckets), ),
                    constraint_specs=(ConstraintSpec(
                        0, 0, helper.infer_shape_num_tokens),
                                      ConstraintSpec(
                                          2, 0, helper.infer_shape_num_tokens),
                                      ConstraintSpec(
                                          5, 0,
                                          helper.infer_shape_max_num_tiles),
                                      ConstraintSpec(
                                          6, 0,
                                          helper.infer_shape_max_num_tiles)),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(self, inputs: List,
                    tactic: Optional[tuple]) -> torch.Tensor:
            """Forward pass.

            Input layout:
                0: a                               - tensor
                1: b                               - tensor
                2: a_sf                            - tensor
                3: b_sf                            - tensor
                4: alpha                           - tensor
                5: tile_idx_to_group_idx           - tensor
                6: tile_idx_to_mn_limit            - tensor
                7: permuted_idx_to_expanded_idx    - tensor
                8: num_non_exiting_tiles           - tensor
                9: global_sf                       - tensor
            """
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, \
                tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, \
                num_non_exiting_tiles, global_sf = inputs

            # Verify input dtypes and dimensions
            assert a.dtype == torch.float4_e2m1fn_x2
            assert a.dim() == 2
            assert b.dtype == torch.float4_e2m1fn_x2
            assert b.dim() == 3
            assert a_sf.dtype == torch.uint8
            assert a_sf.dim() == 2
            assert b_sf.dtype == torch.uint8
            assert b_sf.dim() == 3
            assert alpha.dtype == torch.float32
            assert alpha.dim() == 1

            # a.size(0) is orig_m (original input size before gather)
            # permuted_idx_to_expanded_idx.size(0) is m (permuted size after gather)
            orig_m, k = a.size(0), a.size(1) * 2
            m = permuted_idx_to_expanded_idx.size(0)
            l, n = b.size(0), b.size(1)  # noqa: E741
            scale_k = k // self.scaling_vector_size
            interm_size = n // 2 if self.is_gated else n

            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            if self.is_gated:
                assert n % (self.scaling_vector_size * 4 * 2) == 0
            else:
                assert n % (self.scaling_vector_size * 4) == 0
            assert b.size(2) * 2 == k
            assert a_sf.size(0) == orig_m
            assert a_sf.size(1) == scale_k

            num_tiles = m // self.tile_size
            assert tile_idx_to_group_idx.dtype == torch.int32
            assert tile_idx_to_group_idx.size() == (num_tiles, )
            assert tile_idx_to_mn_limit.dtype == torch.int32
            assert tile_idx_to_mn_limit.size() == (num_tiles, )
            assert permuted_idx_to_expanded_idx.dtype == torch.int32
            assert permuted_idx_to_expanded_idx.size() == (m, )
            assert num_non_exiting_tiles.dtype == torch.int32
            assert num_non_exiting_tiles.numel() == 1
            assert global_sf.dtype == torch.float32
            assert global_sf.numel() == 1

            # Allocate output tensors
            c = torch.empty(m, interm_size // 2, dtype=a.dtype, device=a.device)
            c_sf = torch.empty(m * interm_size // self.scaling_vector_size,
                               dtype=a_sf.dtype,
                               device=a_sf.device)

            # Create pointers.
            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                 cute.AddressSpace.gmem)
            c_ptr = make_ptr(cutlass.Float4E2M1FN,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            c_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                c_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            tile_idx_to_group_idx_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                cute.AddressSpace.gmem)
            tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32,
                                                tile_idx_to_mn_limit.data_ptr(),
                                                cute.AddressSpace.gmem)
            permuted_idx_to_expanded_idx_ptr = make_ptr(
                cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                cute.AddressSpace.gmem)
            num_non_exiting_tiles_ptr = make_ptr(
                cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                cute.AddressSpace.gmem)
            global_sf_ptr = make_ptr(cutlass.Float32, global_sf.data_ptr(),
                                     cute.AddressSpace.gmem)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
            else:
                mma_tiler_mn = (self.tile_size, 128)
                cluster_shape_mn = (self.tile_size // 128, 1)
                raster_along_m = False
            assert mma_tiler_mn[
                0] == self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

            cache_key = (self.scaling_vector_size, self.tile_size, self.top_k,
                         mma_tiler_mn, cluster_shape_mn, raster_along_m,
                         self.activation_type, self.swiglu_limit_scalar)

            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
                    topk=self.top_k,
                    raster_along_m=raster_along_m,
                    activation_type=self.activation_type,
                    swiglu_limit=self.swiglu_limit_scalar,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compile_args = [
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    c_sf_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    global_sf_ptr,
                    orig_m,
                    m,
                    n,
                    k,
                    l,
                ]

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    *compile_args,
                    tile_size=self.tile_size,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                    activation_type=self.activation_type,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            exec_args = [
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                c_sf_ptr,
                alpha_ptr,
                tile_idx_to_group_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                permuted_idx_to_expanded_idx_ptr,
                num_non_exiting_tiles_ptr,
                global_sf_ptr,
                orig_m,
                m,
                n,
                k,
                l,
            ]

            compiled_gemm(*exec_args, stream=stream)

            return c, c_sf

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        activation_type: int = int(ActivationType.Swiglu),
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL-based NVFP4 gather grouped GEMM with activation fusion.

        Supports ``ActivationType.Swiglu`` (gated) and ``ActivationType.Relu2``
        (non-gated) epilogues; other ``ActivationType`` values raise an
        assertion in the runner.
        """
        tuner = AutoTuner.get()
        swiglu_limit_scalar = _canonicalize_swiglu_limit_scalar(
            swiglu_limit_scalar)

        runner = Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner(
            num_experts,
            top_k,
            num_local_experts,
            local_expert_offset,
            tile_size,
            scaling_vector_size,
            activation_type=ActivationType(activation_type),
            swiglu_limit_scalar=swiglu_limit_scalar)
        inputs = [
            input, weight, input_scale, weight_scale, alpha,
            tile_idx_to_group_idx, tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        output = runner.forward(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell")
    def _fake_single_b(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        activation_type: int = int(ActivationType.Swiglu),
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = permuted_idx_to_expanded_idx.size(0)
        n = weight.size(1)
        is_gated = is_gated_activation(ActivationType(activation_type))
        interm_size = n // 2 if is_gated else n
        output = torch.empty(m,
                             interm_size // 2,
                             dtype=input.dtype,
                             device=input.device)
        output_scale = torch.empty(m * interm_size // scaling_vector_size,
                                   dtype=input_scale.dtype,
                                   device=input_scale.device)
        return output, output_scale

    _INDEXER_Q_CUTEDSL_TUNING_BUCKETS = (
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    )
    _INDEXER_Q_POSITION_IDS_INPUT_INDEX = 3

    def _map_cutedsl_indexer_q_tuning_bucket(num_tokens: int) -> int:
        if num_tokens <= 4:
            return 4
        if num_tokens <= 8:
            return 8
        if num_tokens <= 16:
            return 16
        return max(32, last_positive_power_of_2(num_tokens))

    def _prepare_cutedsl_indexer_q_tuning_inputs(
            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # The autotuner resizes position_ids to the token bucket, leaving newly
        # allocated values uninitialized. Use position zero for every tuning
        # row so the fused RoPE lookup always stays inside cos_sin_cache.
        position_ids = inputs[_INDEXER_Q_POSITION_IDS_INPUT_INDEX]
        inputs[_INDEXER_Q_POSITION_IDS_INPUT_INDEX] = torch.zeros_like(
            position_ids)
        return inputs

    class CuteDSLIndexerQBlackwellRunner(TunableRunner):
        """Native MXF8 GEMM with fused DSv4 indexer-Q RoPE/MXFP4 output."""

        kernel_class = Sm100BlockScaledPersistentDenseGemmActFusionKernel
        small_m_kernel_class = Sm100BlockScaledPersistentDenseGemmKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0,
                0,
                _INDEXER_Q_CUTEDSL_TUNING_BUCKETS,
                _map_cutedsl_indexer_q_tuning_bucket,
            ), ),
            constraint_specs=(ConstraintSpec(3, 0,
                                             lambda shapes: shapes[0][0]), ),
            inputs_pre_hook=_prepare_cutedsl_indexer_q_tuning_inputs,
            use_cold_l2_cache=True,
            # CuTe kernels are JIT compiled into a process-local cache while
            # the autotuner profiles tactics. Never persist only the selected
            # tactic: a new process would otherwise skip that compilation and
            # pay for cute.compile in its first inference forward.
            exclude_from_cache=True,
            # Every rank owns a process-local CuTe module cache. Profiling in
            # parallel across ranks would leave the winning tactic uncompiled
            # on ranks that benchmarked a different subset.
            distributed_tuning_strategy=DistributedTuningStrategy.INDEPENDENT,
        )

        _small_m_tactics = (
            ("swap_ab", (128, 16), (1, 1), False, 4),
            ("swap_ab", (128, 16), (1, 1), False, 8),
        )
        _native_tactics = (
            ("native", (128, 128), (1, 1), False, 0),
            ("native", (128, 128), (1, 2), False, 0),
            ("native", (128, 128), (2, 1), False, 0),
            ("native", (128, 128), (2, 1), True, 0),
            ("native", (256, 128), (2, 1), False, 0),
            ("native", (256, 128), (2, 1), True, 0),
            ("native", (256, 128), (2, 2), True, 0),
        )

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__()
            self.use_tvm_ffi = use_tvm_ffi

        def unique_id(self):
            return (self.use_tvm_ffi, )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple]:
            if not is_sm_100f():
                return []
            m, k = inputs[0].shape
            n = inputs[1].shape[0]
            tactics = []
            if self._small_m_kernel_is_supported(m, n, k):
                tactics.extend(self.__class__._small_m_tactics)

            tactics.extend([
                tactic for tactic in self.__class__._native_tactics
                if self.__class__.kernel_class.can_implement(
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E8M0FNU,
                    32,
                    cutlass.Float4E2M1FN,
                    tactic[1],
                    tactic[2],
                    m,
                    n,
                    k,
                    1,
                    "k",
                    "k",
                    "n",
                )
            ])
            return tactics

        @staticmethod
        def _small_m_kernel_is_supported(m: int, n: int, k: int) -> bool:
            return 0 < m <= 16 and n % 128 == 0 and k % 128 == 0

        @classmethod
        def _fallback_tactic(cls, m: int, n: int, k: int) -> Tuple:
            """Safe eager-mode fallback when TRT-LLM autotuning is disabled."""
            if cls._small_m_kernel_is_supported(m, n, k):
                if m <= 4:
                    return ("swap_ab", (128, 16), (1, 1), False, 4)
                if m <= 8:
                    return ("swap_ab", (128, 16), (1, 1), False, 8)
            return ("native", (256, 128), (2, 1), False, 0)

        @staticmethod
        def _ptr(tensor: torch.Tensor, dtype, align: int = 16):
            return make_ptr(
                dtype,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=align,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            input, weight, weight_scale, position_ids, cos_sin_cache, alpha = inputs
            m, k = input.shape
            n = weight.shape[0]
            if tactic == -1:
                tactic = self._fallback_tactic(m, n, k)
            (kernel_kind, mma_tiler_mn, cluster_shape_mn, use_prefetch,
             transform_warps) = tactic
            if kernel_kind == "swap_ab":
                if not 0 < m <= mma_tiler_mn[1]:
                    raise ValueError(
                        "The small-M indexer-Q kernel requires one non-empty "
                        f"token tile: M={m}, tile N={mma_tiler_mn[1]}")
                if n % 128 != 0 or k % 128 != 0:
                    raise ValueError(
                        "The small-M indexer-Q kernel requires N and K to be "
                        f"divisible by 128, but got N={n}, K={k}")

            a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(input)
            packed = torch.empty((m, n // 2),
                                 dtype=torch.uint8,
                                 device=input.device)
            output_scale = torch.empty((m, n // 32),
                                       dtype=torch.uint8,
                                       device=input.device)

            a_ptr = self._ptr(a, cutlass.Float8E4M3FN)
            b_ptr = self._ptr(weight, cutlass.Float8E4M3FN)
            a_sf_ptr = self._ptr(a_sf, cutlass.Float8E8M0FNU)
            b_sf_ptr = self._ptr(weight_scale, cutlass.Float8E8M0FNU)
            packed_ptr = self._ptr(packed, cutlass.Uint8)
            output_scale_ptr = self._ptr(output_scale, cutlass.Float8E8M0FNU)
            position_ids_ptr = self._ptr(position_ids, cutlass.Int32, 4)
            cos_sin_ptr = self._ptr(cos_sin_cache, cutlass.Float32, 32)
            alpha_cute = cute.runtime.from_dlpack(alpha)

            if self.use_tvm_ffi:
                stream = cute.runtime.make_fake_stream(
                    use_tvm_ffi_env_stream=True)
            else:
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            cache_key = (kernel_kind, mma_tiler_mn, cluster_shape_mn,
                         use_prefetch, transform_warps, self.use_tvm_ffi)
            if cache_key not in self.__class__.kernel_cache:
                if kernel_kind == "swap_ab":
                    gemm = self.__class__.small_m_kernel_class(
                        32,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_prefetch=use_prefetch,
                        indexer_q_fusion=True,
                        indexer_transform_warps=transform_warps,
                    )
                    compile_entry = gemm.wrapper_indexer_q_swap_ab
                else:
                    gemm = self.__class__.kernel_class(
                        32,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        True,
                        use_prefetch,
                        activation_type=ActivationType.Identity,
                        indexer_q_fusion=True,
                    )
                    compile_entry = gemm.wrapper_indexer_q
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])
                compiled = cute.compile(
                    compile_entry,
                    m,
                    n,
                    k,
                    pad_up(m, 128) // 128,
                    pad_up(n, 128) // 128,
                    pad_up(k // 32, 4) // 4,
                    cos_sin_cache.shape[0],
                    1,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    packed_ptr,
                    output_scale_ptr,
                    position_ids_ptr,
                    cos_sin_ptr,
                    alpha_cute,
                    max_active_clusters,
                    stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )
                self.__class__.kernel_cache[cache_key] = compiled
            else:
                compiled = self.__class__.kernel_cache[cache_key]

            dynamic_args = [
                m,
                n,
                k,
                pad_up(m, 128) // 128,
                pad_up(n, 128) // 128,
                pad_up(k // 32, 4) // 4,
                cos_sin_cache.shape[0],
            ]
            if self.use_tvm_ffi:
                compiled(
                    *dynamic_args,
                    a.data_ptr(),
                    weight.data_ptr(),
                    a_sf.data_ptr(),
                    weight_scale.data_ptr(),
                    packed.data_ptr(),
                    output_scale.data_ptr(),
                    position_ids.data_ptr(),
                    cos_sin_cache.data_ptr(),
                    alpha,
                )
            else:
                compiled(
                    *dynamic_args,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    packed_ptr,
                    output_scale_ptr,
                    position_ids_ptr,
                    cos_sin_ptr,
                    alpha_cute,
                    stream,
                )
            return packed.view(torch.int8), output_scale.view(torch.int32)

    @torch.library.custom_op(
        "trtllm::cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell",
        mutates_args=(),
        device_types="cuda",
    )
    def cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        alpha: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        runner = CuteDSLIndexerQBlackwellRunner(use_tvm_ffi)
        inputs = [
            input, weight, weight_scale, position_ids, cos_sin_cache, alpha
        ]
        tuner = AutoTuner.get()
        _, tactic = tuner.choose_one(
            "trtllm::cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        return runner(inputs, tactic=tactic)

    @torch.library.register_fake(
        "trtllm::cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        alpha: torch.Tensor,
        use_tvm_ffi: bool = True,
    ):
        m, n = input.shape[0], weight.shape[0]
        return (
            input.new_empty((m, n // 2), dtype=torch.int8),
            input.new_empty((m, n // 128), dtype=torch.int32),
        )

    class CuteDSLFp8BlackwellRunner(TunableRunner):
        kernel_class = Sm100BlockwiseGemmKernel
        kernel_cache = dict()

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 1, fp8_scale_infer_shape), ),
        )

        def __init__(self,
                     output_dtype: torch.dtype = torch.bfloat16,
                     use_tvm_ffi: bool = True):
            super().__init__()
            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL FP8 GEMM only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.use_tvm_ffi = use_tvm_ffi

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[int]:
            if not is_sm_100f():
                logger.debug(
                    f"CuteDSL: SM version {get_sm_version()} is not supported. "
                    f"CuteDSL FP8 GEMM only supports SM 100 family. Skipping all tactics."
                )
                return []

            m = inputs[0].shape[0]
            n = inputs[1].shape[0]
            k = inputs[0].shape[1]
            batch_size = 1
            # m,k
            a_major = "k"
            # n, k
            b_major = "k"
            # m, n
            c_major = "n"

            use_2cta_instrs_candi = [False, True]
            mma_tiler_mn_candi = [(64, 128), (128, 128), (256, 128)]
            cluster_shape_mn_candi = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            return [
                (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
                for use_2cta_instrs in use_2cta_instrs_candi
                for mma_tiler_mn in mma_tiler_mn_candi
                for cluster_shape_mn in cluster_shape_mn_candi
                if self.__class__.kernel_class.can_implement(
                    cutlass.Float8E4M3FN,  # ab_dtype,
                    cutlass.Float32,  # acc_dtype,
                    cutlass.BFloat16,  # c_dtype,
                    use_2cta_instrs,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    m,
                    n,
                    k,
                    batch_size,
                    a_major,
                    b_major,
                    c_major,
                )
            ]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
        ) -> torch.Tensor:
            """
            Performs fp8 blockwise (deepgemm like) operation using CuTe DSL.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (m, k), dtype: fp8.
                    inputs[1]: Weight tensor of shape (n, k), dtype: fp8.
                    inputs[2]: Input scale factor tensor of shape (k // 128, m), dtype: fp32.
                    inputs[3]: Weight scale factor tensor of shape (n // 128, k // 128), dtype: fp32.
                tactic: Tiling and cluster strategy, typically a tuple (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn).

            Returns:
                torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
            """
            if isinstance(tactic, tuple):
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
            else:
                # fallback to default tactic
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = [
                    False,
                    (128, 128),
                    (1, 1),
                ]
            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor = inputs
            m, n, k = a_tensor.shape[0], b_tensor.shape[0], b_tensor.shape[1]
            sf_m = m
            sf_k = ceil_div(k, 128)
            sf_n = ceil_div(n, 128)
            c_tensor = torch.empty(*(m, n),
                                   dtype=torch.bfloat16,
                                   device=a_tensor.device)
            c_tmp = c_tensor.view((1, m, n))
            c_tmp = c_tmp.permute(1, 2, 0)

            if not self.use_tvm_ffi:
                a_ptr = make_ptr(
                    cutlass.Float8E4M3FN,
                    a_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_ptr = make_ptr(
                    cutlass.Float8E4M3FN,
                    b_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                a_sf_ptr = make_ptr(
                    cutlass.Float32,
                    a_sf_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_sf_ptr = make_ptr(
                    cutlass.Float32,
                    b_sf_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                c_cute_tensor = cute.runtime.from_dlpack(
                    c_tmp).mark_layout_dynamic(leading_dim=1)

                # get stream
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            cache_key = (
                use_2cta_instrs,
                mma_tiler_mn,
                cluster_shape_mn,
                self.use_tvm_ffi,
            )
            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = make_ptr(
                        cutlass.Float8E4M3FN,
                        a_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_ptr = make_ptr(
                        cutlass.Float8E4M3FN,
                        b_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    a_sf_ptr = make_ptr(
                        cutlass.Float32,
                        a_sf_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_sf_ptr = make_ptr(
                        cutlass.Float32,
                        b_sf_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    # Convert c_tensor to cute tensor for TVM FFI for env stream detection
                    c_cute_tensor = cute.runtime.from_dlpack(
                        c_tmp).mark_layout_dynamic(leading_dim=1)
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    cutlass.Float32,  # acc_dtype,
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    1,  # batch
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_cute_tensor,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # launch gemm kernel
            if self.use_tvm_ffi:
                # call with torch pointer types and no need to pass stream.
                compiled_gemm(
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    1,  # batch
                    a_tensor.data_ptr(),
                    b_tensor.data_ptr(),
                    a_sf_tensor.data_ptr(),
                    b_sf_tensor.data_ptr(),
                    c_tmp,
                )
            else:
                # call with cute types and need to pass torch stream.
                compiled_gemm(
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    1,  # batch
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_cute_tensor,
                    stream=stream,
                )
            return c_tensor

    # a/b: fp8, scale: fp32, output: bf16
    @torch.library.custom_op("trtllm::cute_dsl_fp8_gemm_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_fp8_gemm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> torch.Tensor:
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"CuteDSL FP8 GEMM only supports bfloat16 output, got {output_dtype}"
            )
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL FP8 GEMM only supports SM 100 family. Skipping all tactics."
            )
        tuner = AutoTuner.get()

        runner = CuteDSLFp8BlackwellRunner(output_dtype=output_dtype,
                                           use_tvm_ffi=use_tvm_ffi)

        inputs = [input, weight, input_scale, weight_scale]
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_fp8_gemm_blackwell::gemm",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_fp8_gemm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k]
        shape[-1] = mat_b.shape[-2]
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

    class CuteDSLFp8BlackwellBmmRunner(TunableRunner):
        kernel_class = Sm100BlockwiseGemmKernel
        kernel_cache = dict()

        # Keep the output M dimension aligned with input0's bucketed M so
        # profiling uses consistent BMM shapes and runtime cache keys can be
        # shared by inputs that map to the same bucket.
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 1, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 2, fp8_scale_infer_shape),
                              ConstraintSpec(
                                  4, 1,
                                  lambda input_shapes: input_shapes[0][1])),
        )

        def __init__(self,
                     output_dtype: torch.dtype = torch.bfloat16,
                     use_tvm_ffi: bool = True):
            super().__init__()
            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL FP8 BMM only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.use_tvm_ffi = use_tvm_ffi

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[int]:

            if not is_sm_100f():
                logger.debug(
                    f"CuteDSL: SM version {get_sm_version()} is not supported. "
                    f"CuteDSL FP8 BMM only supports SM 100 family. Skipping all tactics."
                )
                return []
            # [b, m, k]
            batch_size, m, k = inputs[0].shape[0], inputs[0].shape[1], inputs[
                0].shape[2]
            # [b, n, k]
            n = inputs[1].shape[1]
            # m,k
            a_major = "k"
            # n, k
            b_major = "k"
            # m, n
            c_major = "n"

            use_2cta_instrs_candi = [False, True]
            mma_tiler_mn_candi = [(64, 128), (128, 128), (256, 128)]
            cluster_shape_mn_candi = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            return [
                (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
                for use_2cta_instrs in use_2cta_instrs_candi
                for mma_tiler_mn in mma_tiler_mn_candi
                for cluster_shape_mn in cluster_shape_mn_candi
                if self.__class__.kernel_class.can_implement(
                    cutlass.Float8E4M3FN,  # ab_dtype,
                    cutlass.Float32,  # acc_dtype,
                    cutlass.BFloat16,  # c_dtype,
                    use_2cta_instrs,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    m,
                    n,
                    k,
                    batch_size,
                    a_major,
                    b_major,
                    c_major,
                )
            ]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
        ) -> None:
            """
            Performs fp8 blockwise (deepgemm like) batched gemm operation using CuTe DSL.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (batch_size, m, k), dtype: fp8.
                    inputs[1]: Weight tensor of shape (batch_size, n, k), dtype: fp8.
                    inputs[2]: Input scale tensor of shape (batch_size, k // 128, pad_up(m, 4)), dtype: fp32.
                    inputs[3]: Weight scale tensor of shape (batch_size, n // 128, k // 128), dtype: fp32.
                tactic: Tiling and cluster strategy, typically a tuple (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, m, n), dtype: bf16.
            """
            if isinstance(tactic, tuple):
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
            else:
                # fallback to default tactic
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = [
                    False,
                    (128, 128),
                    (1, 1),
                ]

            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, c_tensor = inputs
            c_tmp = c_tensor.permute(1, 2, 0)

            batch_size = a_tensor.shape[0]
            m = a_tensor.shape[1]
            k = a_tensor.shape[2]
            n = b_tensor.shape[1]
            sf_m = pad_up(m, 4)
            sf_k = ceil_div(k, 128)
            sf_n = ceil_div(n, 128)

            if not self.use_tvm_ffi:
                a_ptr = make_ptr(
                    cutlass.Float8E4M3FN,
                    a_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_ptr = make_ptr(
                    cutlass.Float8E4M3FN,
                    b_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                a_sf_ptr = make_ptr(
                    cutlass.Float32,
                    a_sf_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_sf_ptr = make_ptr(
                    cutlass.Float32,
                    b_sf_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                c_cute_tensor = cute.runtime.from_dlpack(
                    c_tmp).mark_layout_dynamic(leading_dim=1)

                # get stream
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            cache_key = (
                use_2cta_instrs,
                mma_tiler_mn,
                cluster_shape_mn,
                self.use_tvm_ffi,
            )
            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = make_ptr(
                        cutlass.Float8E4M3FN,
                        a_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_ptr = make_ptr(
                        cutlass.Float8E4M3FN,
                        b_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    a_sf_ptr = make_ptr(
                        cutlass.Float32,
                        a_sf_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_sf_ptr = make_ptr(
                        cutlass.Float32,
                        b_sf_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    # Convert c_tensor to cute tensor for TVM FFI for env stream detection)
                    c_cute_tensor = cute.runtime.from_dlpack(
                        c_tmp).mark_layout_dynamic(leading_dim=1)
                    # make faked stream for TVM FFI
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    cutlass.Float32,  # acc_dtype,
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_cute_tensor,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # launch gemm kernel
            if self.use_tvm_ffi:
                # call with torch pointer types and no need to pass stream.
                compiled_gemm(
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    batch_size,
                    a_tensor.data_ptr(),
                    b_tensor.data_ptr(),
                    a_sf_tensor.data_ptr(),
                    b_sf_tensor.data_ptr(),
                    c_tmp,
                )
            else:
                # call with cute types and need to pass torch stream.
                compiled_gemm(
                    m,
                    n,
                    k,
                    sf_m,
                    sf_n,
                    sf_k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_cute_tensor,
                    stream=stream,
                )

    # a/b: fp8, scale: fp32, output: bf16
    @torch.library.custom_op("trtllm::cute_dsl_fp8_bmm_blackwell",
                             mutates_args=("output", ),
                             device_types="cuda")
    def cute_dsl_fp8_bmm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> None:
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"CuteDSL FP8 BMM only supports bfloat16 output, got {output_dtype}"
            )
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL FP8 BMM only supports SM 100 family. Skipping all tactics."
            )

        tuner = AutoTuner.get()

        runner = CuteDSLFp8BlackwellBmmRunner(output_dtype=output_dtype,
                                              use_tvm_ffi=use_tvm_ffi)

        inputs = [input, weight, input_scale, weight_scale, output]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_fp8_bmm_blackwell::gemm",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_fp8_bmm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ) -> None:
        batch_size, m, k = mat_a.shape[0], mat_a.shape[1], mat_a.shape[2]
        n = mat_b.shape[1]
        assert output.dtype == torch.bfloat16, "CuTe DSL fp8 bmm output dtype must be bf16"
        assert output.shape == (batch_size, m,
                                n), "CuTe DSL fp8 bmm output shape is incorrect"

    # =============================================================================
    # Dense GEMM with SwiGLU Fusion (FC1 Kernel for MoE as Dense GEMM)
    # =============================================================================

    class CuteDSLNVFP4DenseGemmSwigluRunner(TunableRunner):
        """Runner for Dense GEMM with SwiGLU fusion (MoE FC1 layer as dense GEMM).

        This kernel performs: C = SwiGLU(alpha * (SFA * A) @ (SFB * B))
        where SwiGLU(x) = up * silu(gate), with up/gate extracted from interleaved output.

        Input shapes:
        - A: (M, K) - activation tensor
        - B: (N, K, L) - weight tensor (L is typically 1 for dense)
        - alpha: (expert_count,) - per-expert scaling, indexed by weight_per_expert

        Output shape:
        - C: (M, N//2) - N//2 due to SwiGLU fusion
        """

        kernel_class = DenseGemmSwigluKernel
        kernel_cache = dict()
        tuning_config_cache = dict()
        _CUTLASS_DTYPE_MAP = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
            torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
        }

        def __init__(
            self,
            expert_count: int,
            weight_per_expert: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
        ):
            super().__init__()
            self.expert_count = expert_count
            self.weight_per_expert = weight_per_expert
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size

        def unique_id(self):
            return (
                self.expert_count,
                self.weight_per_expert,
                self.output_dtype,
                self.scaling_vector_size,
            )

        def __hash__(self):
            return hash(self.unique_id())

        def __eq__(self, other):
            if not isinstance(other, CuteDSLNVFP4DenseGemmSwigluRunner):
                return False
            return self.unique_id() == other.unique_id()

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
            """Return valid (mma_tiler_mn, cluster_shape_mn) combinations."""
            # Check SM version - only supports SM 100 and SM 103
            major, minor = torch.cuda.get_device_capability()
            if not (major == 10 and minor in [0, 3]):
                return []

            a = inputs[0]
            b = inputs[1]
            # a: [m, k//2] (fp4 packed), b: [num_expert, weight_per_expert, k//2]
            m = a.shape[0]
            k = a.shape[1] * 2  # fp4 packed in k dimension
            n = b.shape[0] * b.shape[1]  # num_expert * weight_per_expert
            l = 1  # dense GEMM  # noqa: E741

            # Define candidates together
            mma_tiler_mn_candidates = [(128, 128), (128, 256), (256, 256)]
            cluster_shape_mn_candidates = [(1, 1), (1, 2), (1, 4), (2, 1)]

            # Map torch dtype to cutlass dtype
            if self.output_dtype not in self._CUTLASS_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported output_dtype {self.output_dtype} for FC1 DenseGEMM runner"
                )
            c_cutlass_dtype = self._CUTLASS_DTYPE_MAP[self.output_dtype]

            tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
                if self.kernel_class.can_implement(
                        cutlass.Float4E2M1FN,  # ab_dtype
                        cutlass.Float8E4M3FN,  # sf_dtype
                        self.scaling_vector_size,
                        c_cutlass_dtype,  # c_dtype
                        mma_tiler_mn,
                        cluster_shape_mn,
                        m,
                        n,
                        k,
                        l,
                        "k",  # a_major
                        "k",  # b_major
                        "n",  # c_major
                        self.expert_count,
                        self.weight_per_expert,
                ):
                    tactics.append((mma_tiler_mn, cluster_shape_mn))

            return tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.tuning_config_cache:
                self.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, deep_gemm_gen_tuning_buckets), ),
                    constraint_specs=(ConstraintSpec(2, 0,
                                                     fp4_scale_infer_shape), ),
                    use_cold_l2_cache=True,
                    tune_max_num_tokens=512,
                    distributed_tuning_strategy=DistributedTuningStrategy.
                    PARALLEL,
                )
            return self.tuning_config_cache[key]

        def forward(
            self,
            inputs: List[Optional[torch.Tensor]],
            tactic: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Execute the dense GEMM with SwiGLU fusion.

            Args:
                inputs: [a, b, a_sf, b_sf, alpha, alpha_post, norm_const]
                    - alpha_post can be None to skip post-SwiGLU scaling
                tactic: ((mma_m, mma_n), (cluster_m, cluster_n))

            Returns:
                Tuple of (output, output_scale_factor)
            """
            a, b, a_sf, b_sf, alpha, alpha_post, norm_const = inputs[:7]

            # Get dimensions
            # a: [m, k//2] (fp4 packed), b: [num_expert, weight_per_expert, k//2]
            m = a.shape[0]
            k = a.shape[1] * 2  # fp4 packed in k dimension
            n = b.shape[0] * b.shape[1]  # num_expert * weight_per_expert
            l = 1  # dense GEMM  # noqa: E741
            n_out = n // 2  # SwiGLU output

            # Default tactic if not provided
            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn = tactic
            else:
                mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 1)

            # Allocate output tensor
            c_dtype = self.output_dtype
            if c_dtype == torch.float4_e2m1fn_x2:
                # FP4 packed: 2 elements per byte, so shape is (m, n_out // 2)
                c = torch.empty((m, n_out // 2), dtype=c_dtype, device=a.device)
            else:
                c = torch.empty((m, n_out), dtype=c_dtype, device=a.device)

            # Allocate output scale factor (for FP4 output quantization)
            # Shape: (32, 4, pad_up(m, 128) // 128, 4, scale_n_out // 4, l)
            scale_n_out = n_out // self.scaling_vector_size
            c_sf_shape = (32, 4, pad_up(m, 128) // 128, 4, scale_n_out // 4, l)
            c_sf = torch.empty(c_sf_shape, dtype=torch.uint8, device=a.device)

            # Get CUDA stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # Map torch dtype to cutlass dtype
            if c_dtype not in self._CUTLASS_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported output_dtype {c_dtype} for FC1 DenseGEMM runner"
                )
            c_cutlass_dtype = self._CUTLASS_DTYPE_MAP[c_dtype]

            # Create pointers for kernel
            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            c_ptr = make_ptr(c_cutlass_dtype,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=16)
            c_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                c_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                 cute.AddressSpace.gmem)
            alpha_post_ptr = None
            if alpha_post is not None:
                alpha_post_ptr = make_ptr(cutlass.Float32,
                                          alpha_post.data_ptr(),
                                          cute.AddressSpace.gmem)
            norm_const_ptr = make_ptr(cutlass.Float32, norm_const.data_ptr(),
                                      cute.AddressSpace.gmem)

            # Cache key for compiled kernel
            cache_key = (
                self.weight_per_expert,
                mma_tiler_mn,
                cluster_shape_mn,
                self.scaling_vector_size,
                self.expert_count,
                alpha_post is not None,  # Whether alpha_post is enabled
                self.
                output_dtype,  # Include output dtype to avoid cache collision
            )

            if cache_key not in self.__class__.kernel_cache:
                # Get max active clusters only when compiling kernel
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                kernel = self.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    weight_per_expert=self.weight_per_expert,
                )

                # Compile the kernel and cache it
                compiled_gemm = cute.compile(
                    kernel.wrapper,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    c_sf_ptr,
                    alpha_ptr,
                    alpha_post_ptr,
                    norm_const_ptr,
                    m,
                    n,
                    k,
                    l,
                    expert_count=self.expert_count,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Call the compiled kernel
            compiled_gemm(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                c_sf_ptr,
                alpha_ptr,
                alpha_post_ptr,
                norm_const_ptr,
                m,
                n,
                k,
                l,
                stream=stream,
            )

            return c, c_sf

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell",
        mutates_args=(),
        device_types="cuda",
    )
    def cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        alpha_post: Optional[torch.Tensor],
        norm_const: torch.Tensor,
        expert_count: int,
        weight_per_expert: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense GEMM with SwiGLU fusion for MoE FC1 layer.

        Computes: C = alpha_post * SwiGLU(alpha * (input @ weight.T))
        When alpha_post is None: C = SwiGLU(alpha * (input @ weight.T))

        Args:
            input: Input activation tensor (M, K//2) in fp4 packed format
            weight: Weight tensor [num_expert, weight_per_expert, k//2] in fp4 packed format
            input_scale: Scale factor for input
            weight_scale: Scale factor for weight
            alpha: Per-expert alpha scale (expert_count,)
            alpha_post: Per-token per-expert alpha scale (M, expert_count) applied after SwiGLU,
                or None to skip post-SwiGLU scaling
            norm_const: Normalization constant for SFC generation
            expert_count: Number of experts
            weight_per_expert: Number of weight columns per expert
            output_dtype: Output data type (bfloat16 or float16)
            scaling_vector_size: Block scaling vector size (default: 16)

        Returns:
            Tuple of (output, output_scale_factor)
        """
        runner = CuteDSLNVFP4DenseGemmSwigluRunner(
            expert_count=expert_count,
            weight_per_expert=weight_per_expert,
            output_dtype=output_dtype,
            scaling_vector_size=scaling_vector_size,
        )

        inputs = [
            input, weight, input_scale, weight_scale, alpha, alpha_post,
            norm_const
        ]

        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )

        output, output_sf = runner(inputs, tactic=best_tactic)
        return output, output_sf

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        alpha_post: Optional[torch.Tensor],
        norm_const: torch.Tensor,
        expert_count: int,
        weight_per_expert: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # weight: [num_expert, weight_per_expert, k//2] (fp4 packed)
        m = input.shape[0]
        n = weight.shape[0] * weight.shape[1]  # num_expert * weight_per_expert
        n_out = n // 2  # SwiGLU output
        l = 1  # dense GEMM  # noqa: E741

        if output_dtype == torch.float4_e2m1fn_x2:
            # FP4 packed: 2 elements per byte
            output = input.new_empty((m, n_out // 2), dtype=output_dtype)
        else:
            output = input.new_empty((m, n_out), dtype=output_dtype)

        # Output scale factor shape
        scale_n_out = n_out // scaling_vector_size
        c_sf_shape = (32, 4, pad_up(m, 128) // 128, 4, scale_n_out // 4, l)
        output_sf = input.new_empty(c_sf_shape, dtype=torch.uint8)

        return output, output_sf

    # Import FC2 kernel
    from ..cute_dsl_kernels.blackwell.moe_as_dense_gemm.fc2 import \
        Sm100BlockScaledPersistentDenseGemmKernel as DenseGemmFC2Kernel

    class CuteDSLNVFP4DenseGemmFC2Runner(TunableRunner):
        """Runner for Dense GEMM FC2 layer (MoE second projection).

        This kernel performs: C = (A * SFA) @ (B * SFB) * alpha_scale
        where alpha_scale has shape (m, expert_count) for per-token-per-expert scaling.

        Input shapes:
        - A: (M, K) - activation tensor, K = weight_per_expert * expert_count
        - B: (N, K) - weight tensor
        - alpha_scale: (M, expert_count) - per-token-per-expert scaling

        Output shape:
        - C: (M, N)
        """

        kernel_class = DenseGemmFC2Kernel
        kernel_cache = dict()
        tuning_config_cache = dict()
        _CUTLASS_DTYPE_MAP = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
        }

        def __init__(
            self,
            expert_count: int,
            weight_per_expert: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
        ):
            super().__init__()
            self.expert_count = expert_count
            self.weight_per_expert = weight_per_expert
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size

        def unique_id(self):
            return (
                self.expert_count,
                self.weight_per_expert,
                self.output_dtype,
                self.scaling_vector_size,
            )

        def __hash__(self):
            return hash(self.unique_id())

        def __eq__(self, other):
            if not isinstance(other, CuteDSLNVFP4DenseGemmFC2Runner):
                return False
            return self.unique_id() == other.unique_id()

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
            """Return valid (mma_tiler_mn, cluster_shape_mn, split_k) combinations."""
            # Check SM version - only supports SM 100 and SM 103
            major, minor = torch.cuda.get_device_capability()
            if not (major == 10 and minor in [0, 3]):
                return []

            a = inputs[0]
            b = inputs[1]
            # a: [m, k//2] (fp4 packed), b: [n, k//2]
            m = a.shape[0]
            k = a.shape[1] * 2  # fp4 packed in k dimension
            n = b.shape[0]
            l = 1  # dense GEMM  # noqa: E741

            # Define candidates
            mma_tiler_mn_candidates = [(128, 64), (128, 128), (128, 256),
                                       (256, 128)]
            cluster_shape_mn_candidates = [(1, 1), (1, 2), (1, 4), (2, 1)]
            split_k_candidates = [1, 2, 4]

            # Map torch dtype to cutlass dtype
            if self.output_dtype not in self._CUTLASS_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported output_dtype {self.output_dtype} for FC2 DenseGEMM runner"
                )
            c_cutlass_dtype = self._CUTLASS_DTYPE_MAP[self.output_dtype]

            # MMA tile K size for split-K divisibility check
            _MMA_TILE_K = 256

            tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
                if self.kernel_class.can_implement(
                        cutlass.Float4E2M1FN,  # ab_dtype
                        cutlass.Float8E4M3FN,  # sf_dtype
                        self.scaling_vector_size,
                        c_cutlass_dtype,  # c_dtype
                        mma_tiler_mn,
                        cluster_shape_mn,
                        m,
                        n,
                        k,
                        l,
                        "k",  # a_major
                        "k",  # b_major
                        "n",  # c_major
                        self.expert_count,
                        self.weight_per_expert,
                ):
                    for split_k in split_k_candidates:
                        # K-tiles must be evenly divisible by split_k,
                        # and each split must contain whole experts.
                        k_tiles = k // _MMA_TILE_K
                        tiles_per_expert = self.weight_per_expert // _MMA_TILE_K
                        if (k_tiles % split_k == 0 and
                            (k_tiles // split_k) % tiles_per_expert == 0):
                            tactics.append(
                                (mma_tiler_mn, cluster_shape_mn, split_k))

            return tactics

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.tuning_config_cache:
                self.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, deep_gemm_gen_tuning_buckets), ),
                    constraint_specs=(
                        ConstraintSpec(2, 0, fp4_scale_infer_shape),
                        ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),
                    ),
                    use_cold_l2_cache=True,
                    tune_max_num_tokens=512,
                    distributed_tuning_strategy=DistributedTuningStrategy.
                    PARALLEL,
                )
            return self.tuning_config_cache[key]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: Optional[Tuple[Tuple[int, int], Tuple[int, int], int]],
        ) -> torch.Tensor:
            """Execute the dense GEMM FC2.

            Args:
                inputs: [a, b, a_sf, b_sf, alpha_scale]
                tactic: ((mma_m, mma_n), (cluster_m, cluster_n), split_k)

            Returns:
                Output tensor
            """
            a, b, a_sf, b_sf, alpha_scale = inputs[:5]

            # Get dimensions
            # a: [m, k//2] (fp4 packed), b: [n, k//2]
            m = a.shape[0]
            k = a.shape[1] * 2  # fp4 packed in k dimension
            n = b.shape[0]
            l = 1  # dense GEMM  # noqa: E741

            # The kernel wrapper expects alpha_scale laid out token-major
            # (token has stride 1, expert has stride m), which gives
            # warp 6 a coalesced load of 32 contiguous M alphas per expert.
            # PyTorch's default contiguous (M, expert_count) is expert-major,
            # so transpose+contiguous to convert.
            alpha_scale = alpha_scale.t().contiguous()

            # Default tactic if not provided
            if isinstance(tactic, tuple) and len(tactic) == 3:
                mma_tiler_mn, cluster_shape_mn, split_k = tactic
            elif isinstance(tactic, tuple) and len(tactic) == 2:
                mma_tiler_mn, cluster_shape_mn = tactic
                split_k = 1
            else:
                mma_tiler_mn, cluster_shape_mn, split_k = (128, 128), (1, 1), 1

            # Allocate output tensor
            c_dtype = self.output_dtype
            if split_k > 1:
                # Atomic reduction accumulates onto C; must be zero-initialized
                c = torch.zeros((m, n), dtype=c_dtype, device=a.device)
            else:
                c = torch.empty((m, n), dtype=c_dtype, device=a.device)

            # Get CUDA stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # Map torch dtype to cutlass dtype
            if c_dtype not in self._CUTLASS_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported output_dtype {c_dtype} for FC2 DenseGEMM runner"
                )
            c_cutlass_dtype = self._CUTLASS_DTYPE_MAP[c_dtype]

            # Create pointers for kernel
            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            b_ptr = make_ptr(cutlass.Float4E2M1FN,
                             b.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                b_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
            alpha_scale_ptr = make_ptr(cutlass.Float32,
                                       alpha_scale.data_ptr(),
                                       cute.AddressSpace.gmem,
                                       assumed_align=4)
            c_ptr = make_ptr(c_cutlass_dtype,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=16)

            # Cache key for compiled kernel
            cache_key = (
                self.expert_count,
                self.weight_per_expert,
                mma_tiler_mn,
                cluster_shape_mn,
                split_k,
                self.scaling_vector_size,
                self.
                output_dtype,  # Include output dtype to avoid cache collision
            )

            if cache_key not in self.__class__.kernel_cache:
                # Get max active clusters only when compiling kernel
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                kernel = self.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    expert_count=self.expert_count,
                    weight_per_expert=self.weight_per_expert,
                    split_k=split_k,
                )

                # Compile the kernel and cache it
                compiled_gemm = cute.compile(
                    kernel.wrapper,
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    alpha_scale_ptr,
                    c_ptr,
                    m,
                    n,
                    k,
                    l,
                    expert_count=self.expert_count,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Call the compiled kernel
            compiled_gemm(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                alpha_scale_ptr,
                c_ptr,
                m,
                n,
                k,
                l,
                stream=stream,
            )

            return c

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_dense_gemm_fc2_blackwell",
        mutates_args=(),
        device_types="cuda",
    )
    def cute_dsl_nvfp4_dense_gemm_fc2_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha_scale: torch.Tensor,
        expert_count: int,
        weight_per_expert: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        """Dense GEMM FC2 for MoE second projection.

        Performs: C = (A * SFA) @ (B * SFB) * alpha_scale

        Args:
            input: Input activation (M, K//2) in fp4 packed format
            weight: Weight tensor (N, K//2) in fp4 packed format
            input_scale: Scale factor for input (swizzled)
            weight_scale: Scale factor for weight (swizzled)
            alpha_scale: Per-token-per-expert scale (M, expert_count)
            expert_count: Number of experts
            weight_per_expert: Number of weights per expert
            output_dtype: Output data type (bfloat16 or float16)
            scaling_vector_size: Block scaling vector size (default: 16)

        Returns:
            Output tensor (M, N)
        """
        # FC2 DenseGEMM kernel tiles K with MMA tile size 256.
        # weight_per_expert must be 256-aligned so expert boundaries
        # align with MMA tile boundaries for correct alpha_scale splitting.
        _MMA_TILE_K = 256
        assert weight_per_expert % _MMA_TILE_K == 0, (
            f"cute_dsl_nvfp4_dense_gemm_fc2_blackwell requires weight_per_expert "
            f"to be a multiple of {_MMA_TILE_K} (got {weight_per_expert})")

        runner = CuteDSLNVFP4DenseGemmFC2Runner(
            expert_count=expert_count,
            weight_per_expert=weight_per_expert,
            output_dtype=output_dtype,
            scaling_vector_size=scaling_vector_size,
        )

        inputs = [input, weight, input_scale, weight_scale, alpha_scale]

        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_dense_gemm_fc2_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )

        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_dense_gemm_fc2_blackwell")
    def _(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha_scale: torch.Tensor,
        expert_count: int,
        weight_per_expert: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        # input: [m, k//2] (fp4 packed), weight: [n, k//2]
        m = input.shape[0]
        n = weight.shape[0]

        output = input.new_empty((m, n), dtype=output_dtype)
        return output

    def _get_num_sms() -> int:
        """Return the number of SMs on the current device (cached)."""
        if not hasattr(_get_num_sms, "_value"):
            _get_num_sms._value = (
                torch.cuda.get_device_properties().multi_processor_count)
        return _get_num_sms._value

    # Module-level dtype mapping (avoid recreating per call)
    _TORCH_TO_CUTLASS_DTYPE = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    class CuteDSLTopKDecodeSingleCTARunner:
        """Runner for CuTE DSL Top-K decode kernel (single CTA version).

        This runner manages compilation and execution of the filtered top-k kernel
        optimized for Blackwell architecture using CuTE DSL. It implements a
        radix-based filtering algorithm for efficient top-k selection.

        The runner caches compiled kernels based on configuration (dtype, shape, top_k)
        to avoid redundant recompilation.

        All methods are class-level — no instantiation needed. Call methods directly
        via ``CuteDSLTopKDecodeSingleCTARunner.forward(...)``.

        Attributes:
            kernel_cache: Class-level dict mapping configuration tuples to compiled kernels.
                         Keys are (dtype, num_cols, top_k, next_n, return_val, num_copy_bits,
                         large_occupancy, overflow_policy).

        Note:
            - Requires Blackwell architecture (SM100+)
            - Maximum tested top_k is 2048 (see kernel documentation for larger values)
            - Supports fp16, bf16, and fp32 dtypes
            - Automatically selects occupancy optimization based on batch size
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()

        @classmethod
        def _compile(cls,
                     dtype,
                     bucketed_num_cols,
                     top_k,
                     next_n,
                     return_val,
                     num_copy_bits,
                     large_occupancy,
                     overflow_policy,
                     cache_smem_values=False):
            """Compile and cache a single-CTA top-k kernel for the given config."""
            key = (
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                large_occupancy,
                overflow_policy,
                cache_smem_values,
            )
            if key in cls.kernel_cache:
                return
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            input_fake = cute.runtime.make_fake_compact_tensor(dtype,
                                                               (n_rows, n_cols),
                                                               stride_order=(1,
                                                                             0),
                                                               assumed_align=32)
            if overflow_policy == "GMEM_SPILL":
                buffer_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32,
                    (n_rows, cute.sym_int(), n_cols),
                    stride_order=(2, 1, 0),
                    assumed_align=32,
                )
            else:
                buffer_fake = None
            seqlen_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_batch, ),
                stride_order=(0, ),
            )
            output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
            if return_val:
                output_values_fake = cute.runtime.make_fake_compact_tensor(
                    dtype,
                    (n_rows, top_k),
                    stride_order=(1, 0),
                )
            else:
                output_values_fake = None
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            filtered_topk_func = FilteredTopKKernelVarlenDecode(
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                large_occupancy=large_occupancy,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
            )
            compiled_kernel = cute.compile(
                filtered_topk_func,
                input_fake,
                None,  # indices_fake
                buffer_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=4 if large_occupancy else 1,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled_kernel

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            seq_lens: torch.Tensor,
            top_k: int,
            next_n: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            overflow_policy: str = "REREAD",
            output_indices: Optional[torch.Tensor] = None,
            cache_smem_values: bool = False,
        ):
            """Execute filtered top-k selection on input logits."""
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape
            bucketed_num_cols = next_positive_power_of_2(num_cols)

            num_sms = _get_num_sms()
            large_occupancy = num_rows > num_sms

            key = (
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                large_occupancy,
                overflow_policy,
                cache_smem_values,
            )
            cls._compile(*key)
            compiled_kernel = cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            # Prepare output tensors
            if output_indices is not None:
                output_indices_torch = output_indices
            else:
                output_indices_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch.int32,
                    buffer_name="single_cta_output_indices",
                    reserve_buffer=reserve)
            if return_val:
                output_values_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch_dtype,
                    buffer_name="single_cta_output_values",
                    reserve_buffer=reserve)
            else:
                output_values_torch = None

            # Prepare buffer (GMEM_SPILL only; other policies use None)
            if overflow_policy == "GMEM_SPILL":
                # extra buffer: num_rows * buffer_numbers * num_cols * 4 bytes
                # fp32: up to 256 MB (256 * 2 * 262144 * 4)
                # fp16/bf16: up to 128 MB (256 * 1 * 262144 * 4)
                buffer_numbers = 2 if dtype == cutlass.Float32 else 1
                buffer_torch = cls.buffers.get_buffer(
                    [num_rows, buffer_numbers, bucketed_num_cols],
                    torch.int32,
                    buffer_name="single_cta_buffer",
                    reserve_buffer=reserve)
                buffer_torch = buffer_torch[:, :, :num_cols]
            else:
                buffer_torch = None

            # Execute kernel (TVM FFI uses env stream automatically)
            compiled_kernel(
                input_values,
                None,  # indices
                buffer_torch,
                seq_lens,
                output_indices_torch,
                output_values_torch,
            )

            return output_indices_torch, output_values_torch

    # TODO: rename,  CuteDSLTopKDecodeRadixFilterSPMultiCTARunner -> CuteDSLRadixFilterTopKSPMultiCTARunner
    class CuteDSLTopKDecodeRadixFilterSPMultiCTARunner:
        """Runner for the radix-FILTER single-pass multi-CTA decode top-k kernel.

        Distinct from the existing radix-SELECT SP multi-CTA runners
        (``CuteDSLTopKDecodeSinglePassMultiCTA[Cluster]Runner``): this one drives
        ``FilteredTopKKernelVarlenDecode`` with ``single_pass_multi_cta=True`` —
        a cluster of ``cluster_size`` CTAs cooperates on one row via DSMEM
        histogram merge + DSMEM prefix-scan collection (no GMEM state).

        ``cluster_size`` is a REQUIRED argument (no auto-config yet — radix-select's
        ``_get_chunk_config`` tuning does not transfer; auto-config is a documented
        TODO). ``chunk_size_per_cta = ceil(bucketed_num_cols / cluster_size)``.
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()

        @classmethod
        def _compile(
            cls,
            dtype,
            bucketed_num_cols,
            top_k,
            next_n,
            return_val,
            num_copy_bits,
            cluster_size,
            chunk_size_per_cta,
            overflow_policy,
            cache_smem_values=False,
        ):
            key = (
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                cluster_size,
                chunk_size_per_cta,
                overflow_policy,
                cache_smem_values,
            )
            if key in cls.kernel_cache:
                return
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            input_fake = cute.runtime.make_fake_compact_tensor(dtype,
                                                               (n_rows, n_cols),
                                                               stride_order=(1,
                                                                             0),
                                                               assumed_align=32)
            if overflow_policy == "GMEM_SPILL":
                # Per-CTA spill buffer: (num_rows * cluster_size, num_buffers,
                # chunk_size_per_cta) — independent dims from the input tensor.
                buffer_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32,
                    (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                    stride_order=(2, 1, 0),
                    assumed_align=32,
                )
            else:
                buffer_fake = None
            seqlen_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, ), stride_order=(0, ))
            output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k), stride_order=(1, 0))
            if return_val:
                output_values_fake = cute.runtime.make_fake_compact_tensor(
                    dtype, (n_rows, top_k), stride_order=(1, 0))
            else:
                output_values_fake = None
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            filtered_topk_func = FilteredTopKKernelVarlenDecode(
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                chunk_size_per_cta=chunk_size_per_cta,
                num_ctas_per_row=cluster_size,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
                single_pass_multi_cta=True,
            )
            compiled_kernel = cute.compile(
                filtered_topk_func,
                input_fake,
                None,  # indices_fake
                buffer_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=1,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled_kernel

        @staticmethod
        def auto_cluster_size(num_tokens: int,
                              num_rows: int,
                              is_fp32: bool,
                              num_sms: Optional[int] = None) -> int:
            """cluster_size for the REREAD overflow policy (has the large_occupancy
            re-scan cliff). Shares peak_cs/occ_cap with auto_cluster_size_truncate
            but floors nr > num_sms at cs=2 for huge N to dodge the single-CTA
            large_occupancy REREAD blowup. Re-tuned on a B200 REREAD sweep
            (~1.2% mean overhead vs oracle). Caveat: tuned on randn, fixed-length
            seqlen inputs; real (concentrated) logits or varlen may shift it.
            """
            num_sms = num_sms or _get_num_sms()
            n = num_tokens
            # peak_cs by N; fp32 stays single up to 16K (4 refine rounds cost more).
            peak = (1 if n <= 8192 or (is_fp32 and n <= 16384) else
                    4 if n <= 32768 else 8 if n <= 131072 else 16)
            # occ_cap by num_rows; grid budget tightens as cs grows. nr > num_sms
            # -> cs=2 for huge N (split the row so single-CTA large_occupancy REREAD
            # blowup is avoided), else single.
            occ = (16 if num_rows <= 4 else 8 if num_rows <= 8 else 4 if
                   num_rows <= 32 else 2 if num_rows <= 64 else 1 if num_rows <=
                   num_sms else 2 if n >= 262144 else 1)
            cs = min(peak, occ, _query_max_cluster_size())
            return cs if cs >= 2 else 1

        @staticmethod
        def auto_cluster_size_truncate(num_tokens: int,
                                       num_rows: int,
                                       is_fp32: bool,
                                       num_sms: Optional[int] = None) -> int:
            """cluster_size for cliff-free overflow policies (TRUNCATE/GMEM_SPILL).
            NOT interchangeable with auto_cluster_size (REREAD-tuned): here
            nr > num_sms uses single-CTA (no REREAD large_occupancy blowup).
            Re-tuned on a B200 TRUNCATE sweep (~0.6% mean overhead vs oracle).
            Caveat: tuned on randn, fixed-length seqlen inputs; real (concentrated)
            logit distributions or varlen seqlens may shift the optimum.
            """
            num_sms = num_sms or _get_num_sms()
            n = num_tokens
            # peak_cs by N; fp32 stays single up to 16K (4 refine rounds cost more).
            peak = (1 if n <= 8192 or (is_fp32 and n <= 16384) else
                    4 if n <= 32768 else 8 if n <= 131072 else 16)
            # occ_cap by num_rows; grid budget tightens as cs grows. nr > num_sms
            # -> single, except a narrow just-over-one-wave SP band at large N.
            occ = (16 if num_rows <= 4 else
                   8 if num_rows <= 8 else 4 if num_rows <= 32 else
                   2 if num_rows <= 64 else 1 if num_rows <= num_sms else 2 if
                   (num_rows <= 200 and n >= 131072) else 1)
            cs = min(peak, occ, _query_max_cluster_size())
            return cs if cs >= 2 else 1

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            seq_lens: torch.Tensor,
            top_k: int,
            next_n: int,
            cluster_size: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            overflow_policy: str = "REREAD",
            output_indices: Optional[torch.Tensor] = None,
            cache_smem_values: bool = False,
        ):
            """Execute radix-filter SP multi-CTA cluster top-k.

            ``cluster_size`` (= ctas_per_group) must be provided by the caller.
            """
            assert cluster_size >= 1, f"cluster_size must be >= 1, got {cluster_size}"
            hw_max_cluster = _query_max_cluster_size()
            assert cluster_size <= hw_max_cluster, (
                f"cluster_size={cluster_size} exceeds hardware max cluster "
                f"size {hw_max_cluster}")
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape
            bucketed_num_cols = next_positive_power_of_2(num_cols)
            chunk_size_per_cta = math.ceil(bucketed_num_cols / cluster_size)

            key = (
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                cluster_size,
                chunk_size_per_cta,
                overflow_policy,
                cache_smem_values,
            )
            cls._compile(*key)
            compiled_kernel = cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            if output_indices is not None:
                output_indices_torch = output_indices
            else:
                output_indices_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch.int32,
                    buffer_name="rf_sp_multi_cta_output_indices",
                    reserve_buffer=reserve,
                )
            if return_val:
                output_values_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch_dtype,
                    buffer_name="rf_sp_multi_cta_output_values",
                    reserve_buffer=reserve,
                )
            else:
                output_values_torch = None

            if overflow_policy == "GMEM_SPILL":
                # Per-CTA extra buffer: (num_rows * cluster_size, buffer_numbers,
                # chunk_size_per_cta).
                buffer_numbers = 2 if dtype == cutlass.Float32 else 1
                buffer_torch = cls.buffers.get_buffer(
                    [
                        num_rows * cluster_size, buffer_numbers,
                        chunk_size_per_cta
                    ],
                    torch.int32,
                    buffer_name="rf_sp_multi_cta_buffer",
                    reserve_buffer=reserve,
                )
            else:
                buffer_torch = None

            compiled_kernel(
                input_values,
                None,  # indices
                buffer_torch,
                seq_lens,
                output_indices_torch,
                output_values_torch,
            )

            return output_indices_torch, output_values_torch

    @torch.library.custom_op("trtllm::cute_dsl_topk_decode_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_topk_decode_blackwell(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
    ) -> torch.Tensor:
        """CuteDSL-based Top-K selection optimized for Blackwell decode phase.

        Args:
            input_values: Input logits tensor [batch_size * next_n, vocab_size]
            seq_lens: Sequence lengths for each batch [batch_size]
            top_k: Number of top elements to select (max 16384)
            next_n: Number of candidates per sequence (for speculative decoding)
            num_copy_bits: Number of bits for vectorized memory copy (128 or 256)

        Returns:
            indices: Top-k indices [batch_size * next_n, top_k]

        Note:
            This function requires Blackwell architecture (SM100+) and CuTE DSL support.
            Maximum supported top_k is 16384.
        """
        # Validate SM version
        sm_version = get_sm_version()
        if sm_version < 100:
            raise ValueError(
                f"CuTE DSL top-k requires Blackwell (SM100+), but got SM {sm_version}. "
                "Use standard top-k implementation for older architectures.")

        # Validate inputs
        if top_k <= 0 or top_k > 16384:
            raise ValueError(
                f"top_k must be in range [1, 16384], got {top_k}. "
                "Maximum supported top_k is 16384 for Blackwell architecture.")

        if next_n <= 0:
            raise ValueError(f"next_n must be positive, got {next_n}")

        if num_copy_bits not in [128, 256]:
            raise ValueError(
                f"num_copy_bits must be 128 or 256, got {num_copy_bits}")

        if input_values.dim() != 2:
            raise ValueError(
                f"input_values must be 2D [num_rows, vocab_size], got shape {input_values.shape}"
            )

        if seq_lens.dim() != 1:
            raise ValueError(
                f"seq_lens must be 1D [batch_size], got shape {seq_lens.shape}")

        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if input_values.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported dtype {input_values.dtype}. "
                             f"Supported dtypes: {supported_dtypes}")

        indices, _ = CuteDSLTopKDecodeSingleCTARunner.forward(
            input_values=input_values,
            seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=False,  # Only return indices
            num_copy_bits=num_copy_bits,
        )
        return indices

    @torch.library.register_fake("trtllm::cute_dsl_topk_decode_blackwell")
    def _(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
    ):
        num_rows = input_values.shape[0]
        input_values.dtype

        # Create output tensors matching the custom op return signature: (values, indices)
        indices = input_values.new_empty((num_rows, top_k), dtype=torch.int32)
        return indices

    class CuteDSLTopKPrefillSingleCTARunner:
        """Runner for CuTE DSL Top-K prefill kernel (single CTA per row).

        Uses FilteredTopKKernelVarlenPrefill with large_occupancy=True (512
        threads/CTA, reduced SMEM). Row extents are supplied as row_starts /
        row_ends tensors; output indices are LOCAL (0-indexed within each row's
        valid range), matching the CUDA indexer_topk_prefill convention.

        All methods are class-level — no instantiation needed.
        """

        kernel_cache: dict = {}
        buffers = get_memory_buffers()

        @classmethod
        def _compile(cls,
                     dtype,
                     bucketed_num_cols,
                     top_k,
                     return_val,
                     num_copy_bits,
                     overflow_policy,
                     cache_smem_values=False):
            """Compile and cache a single-CTA prefill top-k kernel."""
            key = (dtype, bucketed_num_cols, top_k, return_val, num_copy_bits,
                   overflow_policy, cache_smem_values)
            if key in cls.kernel_cache:
                return
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            input_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=32,
            )
            row_starts_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, ),
                stride_order=(0, ),
            )
            row_ends_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, ),
                stride_order=(0, ),
            )
            if overflow_policy == "GMEM_SPILL":
                buffer_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32,
                    (n_rows, cute.sym_int(), n_cols),
                    stride_order=(2, 1, 0),
                    assumed_align=32,
                )
            else:
                buffer_fake = None
            output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
            if return_val:
                output_values_fake = cute.runtime.make_fake_compact_tensor(
                    dtype,
                    (n_rows, top_k),
                    stride_order=(1, 0),
                )
            else:
                output_values_fake = None
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            filtered_topk_func = FilteredTopKKernelVarlenPrefill(
                dtype,
                bucketed_num_cols,
                top_k,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
            )
            compiled_kernel = cute.compile(
                filtered_topk_func,
                input_fake,
                row_starts_fake,
                row_ends_fake,
                buffer_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=4,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled_kernel

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            row_starts: torch.Tensor,
            row_ends: torch.Tensor,
            top_k: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            overflow_policy: str = "REREAD",
            output_indices: Optional[torch.Tensor] = None,
            cache_smem_values: bool = False,
        ):
            """Execute filtered top-k selection for prefill rows."""
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape
            bucketed_num_cols = next_positive_power_of_2(num_cols)

            key = (dtype, bucketed_num_cols, top_k, return_val, num_copy_bits,
                   overflow_policy, cache_smem_values)
            cls._compile(*key)
            compiled_kernel = cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            if output_indices is not None:
                output_indices_torch = output_indices
            else:
                output_indices_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch.int32,
                    buffer_name="prefill_single_cta_output_indices",
                    reserve_buffer=reserve,
                )
            if return_val:
                output_values_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch_dtype,
                    buffer_name="prefill_single_cta_output_values",
                    reserve_buffer=reserve,
                )
            else:
                output_values_torch = None

            if overflow_policy == "GMEM_SPILL":
                buffer_numbers = 2 if dtype == cutlass.Float32 else 1
                buffer_torch = cls.buffers.get_buffer(
                    [num_rows, buffer_numbers, bucketed_num_cols],
                    torch.int32,
                    buffer_name="prefill_single_cta_buffer",
                    reserve_buffer=reserve,
                )
                buffer_torch = buffer_torch[:, :, :num_cols]
            else:
                buffer_torch = None

            compiled_kernel(
                input_values,
                row_starts,
                row_ends,
                buffer_torch,
                output_indices_torch,
                output_values_torch,
            )
            return output_indices_torch, output_values_torch

    @torch.library.custom_op(
        "trtllm::cute_dsl_indexer_topk_prefill_blackwell",
        mutates_args=("output_indices", ),
        device_types="cuda",
    )
    def cute_dsl_indexer_topk_prefill_blackwell(
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        num_copy_bits: int = 256,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ) -> None:
        """CuTE DSL radix-based top-k for prefill.

        Args:
            input_values:   Logits tensor of shape (num_rows, num_cols).
            row_starts:     Per-row start column (inclusive), shape (num_rows,), int32.
            row_ends:       Per-row end column (exclusive), shape (num_rows,), int32.
            top_k:          Number of top-k indices to select per row.
            num_copy_bits:  Vector copy width in bits (default 256).
            overflow_policy: How to handle threshold-bucket SMEM overflow.
                             "GMEM_SPILL" (default, exact) or "TRUNCATE" (non-exact,
                             no extra buffer).
            cache_smem_values: Cache ordered values in SMEM to avoid re-reading from
                               GMEM in refinement rounds (reduces S by 2x).

            output_indices: Pre-allocated Int32 tensor of shape (num_rows, top_k),
                            written in place with LOCAL indices (0-indexed within
                            [row_start, row_end) for each row). Padding positions
                            are -1.
        """
        # Write into the caller-provided output_indices (mutates_args) rather than
        # returning the runner's reusable pool buffer, matching the decode op and
        # the CUDA indexer_topk_prefill contract (write into a caller buffer slice).
        CuteDSLTopKPrefillSingleCTARunner.forward(
            input_values,
            row_starts,
            row_ends,
            top_k,
            return_val=False,
            num_copy_bits=num_copy_bits,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            output_indices=output_indices,
        )

    @torch.library.register_fake(
        "trtllm::cute_dsl_indexer_topk_prefill_blackwell")
    def _(
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        num_copy_bits: int = 256,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ):
        return None

    class CuteDSLTopKDecodeMultiCTARunner:
        """Runner for CuTE DSL Top-K decode kernel (multi CTA version).

        This runner manages compilation and execution of the filtered top-k kernel
        using multiple CTAs per row, optimized for Blackwell architecture using
        CuTE DSL. It splits each row into chunks processed by separate CTAs,
        then merges partial results in a second kernel pass.

        Supports two modes:
        - **Static** (dynamic=False): Fixed grid (num_rows, num_ctas_per_row).
          All rows get the same number of CTAs.
        - **Dynamic** (dynamic=True): 1D grid with binary search task mapping.
          Each row gets only the CTAs it needs. Merge kernel reads per-row
          valid length from an offset table.

        The runner caches compiled kernel pairs (first pass + merge pass) based on
        configuration to avoid redundant recompilation.

        All methods are class-level — no instantiation needed. Call methods directly
        via ``CuteDSLTopKDecodeMultiCTARunner.forward(...)``.

        Attributes:
            kernel_cache: Class-level dict mapping configuration tuples to compiled
                         kernel pairs (first_kernel, second_kernel).

        Note:
            - Requires Blackwell architecture (SM100+)
            - Maximum tested top_k is 2048
            - Supports fp16, bf16, and fp32 dtypes
            - Automatically selects occupancy optimization based on batch size
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()

        @classmethod
        def _compile(cls,
                     dtype,
                     top_k,
                     next_n,
                     return_val,
                     num_copy_bits,
                     large_occupancy,
                     chunk_size_per_cta,
                     num_ctas_per_row,
                     overflow_policy="REREAD",
                     dynamic=False,
                     cache_smem_values=False):
            """Compile and cache multi-CTA top-k kernels for the given config."""
            key = (
                dtype,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                large_occupancy,
                chunk_size_per_cta,
                num_ctas_per_row,
                overflow_policy,
                dynamic,
                cache_smem_values,
            )
            if key in cls.kernel_cache:
                return
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            input_fake = cute.runtime.make_fake_compact_tensor(dtype,
                                                               (n_rows, n_cols),
                                                               stride_order=(1,
                                                                             0),
                                                               assumed_align=32)
            # extra_buffer for GMEM_SPILL: spills threshold-bin candidates that
            # overflow SMEM. Not needed for TRUNCATE/REREAD/REREAD_ALWAYS.
            if overflow_policy == "GMEM_SPILL":
                buffer_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32,
                    (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                    stride_order=(2, 1, 0),
                    assumed_align=32,
                )
            else:
                buffer_fake = None
            seqlen_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_batch, ),
                stride_order=(0, ),
            )
            n_first_kernel_output_cols = cute.sym_int()
            first_kernel_output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, n_first_kernel_output_cols),
                stride_order=(1, 0),
            )
            first_kernel_output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, n_first_kernel_output_cols),
                stride_order=(1, 0),
                assumed_align=32,
            )
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            # First kernel: process each chunk independently
            filtered_topk_func_first = FilteredTopKKernelVarlenDecode(
                dtype,
                chunk_size_per_cta,  # num_cols
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=True,  # first kernel must return values
                large_occupancy=large_occupancy,
                enable_multi_cta=True,
                chunk_size_per_cta=chunk_size_per_cta,
                num_ctas_per_row=num_ctas_per_row,
                merge_blocks=False,
                enable_dynamic_multi_cta=dynamic,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
            )
            compiled_kernel_first = cute.compile(
                filtered_topk_func_first,
                input_fake,
                None,  # indices_fake
                buffer_fake,
                seqlen_fake,
                first_kernel_output_indices_fake,
                first_kernel_output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=1,
                options="--enable-tvm-ffi",
            )

            # Second kernel: merge partial results
            merge_num_cols = num_ctas_per_row * top_k
            indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, n_first_kernel_output_cols),
                stride_order=(1, 0),
            )
            output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
            if return_val:
                output_values_fake = cute.runtime.make_fake_compact_tensor(
                    dtype,
                    (n_rows, top_k),
                    stride_order=(1, 0),
                )
            else:
                output_values_fake = None

            filtered_topk_func_second = FilteredTopKKernelVarlenDecode(
                dtype,
                merge_num_cols,  # num_cols
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                large_occupancy=large_occupancy,
                enable_multi_cta=False,
                merge_blocks=True,
                varlen_merge_input=dynamic,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
            )
            compiled_kernel_second = cute.compile(
                filtered_topk_func_second,
                input_fake,
                indices_fake,
                buffer_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=1,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = (compiled_kernel_first,
                                     compiled_kernel_second)

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            seq_lens: torch.Tensor,
            top_k: int,
            next_n: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            chunk_size_per_cta: int = 16384,
            overflow_policy: str = "REREAD",
            dynamic: bool = True,
            output_indices: Optional[torch.Tensor] = None,
            cache_smem_values: bool = False,
        ):
            """Execute multi-CTA filtered top-k selection on input logits."""
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape

            num_sms = _get_num_sms()
            large_occupancy = num_rows > num_sms

            num_ctas_per_row = math.ceil(num_cols / chunk_size_per_cta)
            merge_cols = num_ctas_per_row * top_k

            key = (
                dtype,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                large_occupancy,
                chunk_size_per_cta,
                num_ctas_per_row,
                overflow_policy,
                dynamic,
                cache_smem_values,
            )
            cls._compile(*key)
            compiled_kernel_first, compiled_kernel_second = \
                cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            # Intermediate buffers for first kernel output
            first_output_indices = cls.buffers.get_buffer(
                [num_rows, merge_cols],
                torch.int32,
                buffer_name="multi_cta_first_output_indices",
                reserve_buffer=reserve)
            first_output_values = cls.buffers.get_buffer(
                [num_rows, merge_cols],
                torch_dtype,
                buffer_name="multi_cta_first_output_values",
                reserve_buffer=reserve)

            # extra_buffer for GMEM_SPILL: spills threshold-bin candidates that
            # overflow SMEM. Not needed for TRUNCATE/REREAD/REREAD_ALWAYS.
            if overflow_policy == "GMEM_SPILL":
                buffer_numbers = 2 if dtype == cutlass.Float32 else 1
                buffer_dim2 = max(chunk_size_per_cta, merge_cols)
                buffer_torch = cls.buffers.get_buffer(
                    [num_rows * num_ctas_per_row, buffer_numbers, buffer_dim2],
                    torch.int32,
                    buffer_name="multi_cta_buffer",
                    reserve_buffer=reserve)
            else:
                buffer_torch = None

            # Final output tensors
            if output_indices is not None:
                output_indices_torch = output_indices
            else:
                output_indices_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch.int32,
                    buffer_name="multi_cta_output_indices",
                    reserve_buffer=reserve)
            if return_val:
                output_values_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch_dtype,
                    buffer_name="multi_cta_output_values",
                    reserve_buffer=reserve)
            else:
                output_values_torch = None

            # Execute first kernel: per-chunk top-k
            compiled_kernel_first(
                input_values,
                None,  # indices
                buffer_torch,
                seq_lens,
                first_output_indices,
                first_output_values,
            )

            # Execute second kernel: merge partial results
            compiled_kernel_second(
                first_output_values,
                first_output_indices,
                buffer_torch,
                seq_lens,
                output_indices_torch,
                output_values_torch,
            )

            return output_indices_torch, output_values_torch

    class CuteDSLTopKDecodeSinglePassMultiCTARunner:
        """Runner for single-pass multi-CTA radix top-k (FlashInfer-style fused multi-CTA).

        All CTAs in a group cooperatively find the global pivot via multi-round
        radix select with global histogram merging, then each CTA collects
        results from its own chunk.  Single kernel launch, no intermediate
        buffer, no merge kernel.

        All methods are class-level — no instantiation needed.

        Attributes:
            kernel_cache: Class-level dict mapping config tuples to compiled
                         kernels.
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()
        _row_states_initialized = False
        _row_states_buffer_name = "sp_mcta_row_states"
        _buf_prefix = "sp_mcta_"
        _kernel_class = SinglePassMultiCTARadixTopKKernel
        _state_size = DISTRIBUTED_TOPK_STATE_SIZE

        @classmethod
        def _compile(cls, dtype, chunk_size, top_k, next_n, num_copy_bits,
                     ctas_per_group, num_sms, return_val):
            """Compile and cache a single-pass multi-CTA radix top-k kernel."""
            key = (dtype, chunk_size, top_k, next_n, num_copy_bits,
                   ctas_per_group, num_sms, return_val)
            if key in cls.kernel_cache:
                return
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            n_groups = cute.sym_int()

            input_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=32,
            )
            row_states_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_groups, cls._state_size),
                stride_order=(1, 0),
                assumed_align=32,
            )
            seqlen_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_batch, ),
                stride_order=(0, ),
            )
            output_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
            if return_val:
                output_values_fake = cute.runtime.make_fake_compact_tensor(
                    dtype,
                    (n_rows, top_k),
                    stride_order=(1, 0),
                )
            else:
                output_values_fake = None
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            kernel_obj = cls._kernel_class(
                dtype=dtype,
                chunk_size=chunk_size,
                top_k=top_k,
                next_n=next_n,
                num_copy_bits=num_copy_bits,
                ctas_per_group=ctas_per_group,
                num_sms=num_sms,
            )
            compiled_kernel = cute.compile(
                kernel_obj,
                input_fake,
                row_states_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled_kernel

        @classmethod
        def _compute_max_chunk(cls, dtype, num_copy_bits: int = 256):
            """Compute the maximum chunk_size a single CTA can handle."""
            max_smem = cutlass.utils.get_smem_capacity_in_bytes()
            # Fixed shared memory overhead (excludes shared_ordered[chunk_size]):
            # local_histogram[256]*4 + prefix_buf[256]*4 + scalars[4]*4 + warp_sums[8]*4
            overhead = 256 * 4 * 2 + 4 * 4 + 8 * 4
            if dtype == cutlass.Float32:
                ordered_elem_size = 4
            else:
                ordered_elem_size = 2
            vec_size = num_copy_bits // dtype.width
            max_chunk = (max_smem - overhead) // ordered_elem_size
            max_chunk = (max_chunk // vec_size) * vec_size
            return max_chunk, vec_size

        @classmethod
        def _get_chunk_config(cls,
                              dtype,
                              num_cols: int,
                              chunk_size: Optional[int] = None,
                              num_copy_bits: int = 256,
                              num_rows: int = 1):
            """Resolve chunk_size and ctas_per_group.

            If chunk_size is provided, use it (clamped and aligned).
            Otherwise use an SM-aware heuristic that targets
            total_ctas ≈ num_sms by balancing parallelism against
            per-CTA reduce overhead.

            Returns:
                (chunk_size, ctas_per_group, vec_size)
            """
            max_chunk, vec_size = cls._compute_max_chunk(dtype, num_copy_bits)

            if chunk_size is not None:
                # User-specified: clamp and align
                chunk_size = min(chunk_size, max_chunk)
                chunk_size = (chunk_size // vec_size) * vec_size
                if chunk_size < vec_size:
                    chunk_size = vec_size
            else:
                # Auto: SM-aware heuristic
                num_sms = _get_num_sms()

                # Target total_ctas ≈ num_sms
                ideal_ctas_per_group = max(1, num_sms // max(num_rows, 1))

                if ideal_ctas_per_group <= 1:
                    # Large batch: use FlashInfer-style logic —
                    # minimize ctas_per_group based on max_chunk capacity
                    ctas_per_group = math.ceil(num_cols / max_chunk)
                    if ctas_per_group < 1:
                        ctas_per_group = 1
                    chunk_size = math.ceil(num_cols / ctas_per_group)
                    chunk_size = (
                        (chunk_size + vec_size - 1) // vec_size) * vec_size
                    if chunk_size > max_chunk:
                        chunk_size = max_chunk
                else:
                    chunk_size = math.ceil(num_cols / ideal_ctas_per_group)

                    # Minimum chunk to avoid per-CTA overhead dominating
                    chunk_size = max(chunk_size, 8192)

                    # Avoid ctas_per_group=2 with small chunks: reduce
                    # overhead (~5us) exceeds 2-way parallelism benefit
                    ctas_per_group = math.ceil(num_cols / chunk_size)
                    if ctas_per_group == 2 and chunk_size < 32768:
                        chunk_size = num_cols

                    # Snap to power-of-2 for JIT cache friendliness
                    snap_up = 1 << math.ceil(math.log2(max(chunk_size, 1)))
                    if snap_up > max_chunk:
                        snap_up = 1 << int(math.log2(max_chunk))
                    chunk_size = snap_up

            ctas_per_group = math.ceil(num_cols / chunk_size)
            return chunk_size, ctas_per_group, vec_size

        @classmethod
        def _get_possible_chunk_sizes(cls, dtype, num_copy_bits: int = 256):
            """Return all possible chunk_size values the auto heuristic can produce.

            These are powers of 2 from 8192 up to the largest power of 2
            that fits within max_chunk (for the SM-aware multi-CTA path).
            """
            max_chunk, _ = cls._compute_max_chunk(dtype, num_copy_bits)
            sizes = []
            cs = 8192
            while cs <= max_chunk:
                sizes.append(cs)
                cs *= 2
            return sizes

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            seq_lens: torch.Tensor,
            top_k: int,
            next_n: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            chunk_size: Optional[int] = None,
            output_indices: Optional[torch.Tensor] = None,
        ):
            """Execute single-pass multi-CTA radix top-k selection.

            Args:
                chunk_size: Optional chunk size per CTA. If None, uses the
                    maximum chunk that fits in shared memory. Smaller values
                    increase ctas_per_group (more parallelism) at the cost of
                    more inter-CTA synchronization.
            """
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape
            num_sms = _get_num_sms()

            chunk_size, ctas_per_group, _ = cls._get_chunk_config(
                dtype, num_cols, chunk_size, num_copy_bits, num_rows=num_rows)

            num_groups = min(num_sms // ctas_per_group, num_rows)
            if num_groups < 1:
                num_groups = 1

            key = (dtype, chunk_size, top_k, next_n, num_copy_bits,
                   ctas_per_group, num_sms, return_val)
            cls._compile(*key)
            compiled_kernel = cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            # Allocate row_states once with num_sms rows — large enough for
            # any ctas_per_group config because group_id < num_groups
            # <= num_sms // ctas_per_group <= num_sms.  The kernel resets
            # the slots it used at end-of-kernel, so the buffer stays clean
            # across calls without re-zeroing (FlashInfer pattern).
            # extra buffer: 148 * 770 * 4 bytes = 452960 bytes = 440 KB
            buf_name = cls._row_states_buffer_name
            row_states = cls.buffers.get_buffer([num_sms, cls._state_size],
                                                torch.int32,
                                                buffer_name=buf_name,
                                                reserve_buffer=reserve)
            if not cls._row_states_initialized:
                row_states.zero_()
                cls._row_states_initialized = True

            # Allocate outputs
            if output_indices is not None:
                output_indices_torch = output_indices
            else:
                output_indices_torch = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch.int32,
                    buffer_name=cls._buf_prefix + "output_indices",
                    reserve_buffer=reserve)
            if return_val:
                output_values = cls.buffers.get_buffer(
                    [num_rows, top_k],
                    torch_dtype,
                    buffer_name=cls._buf_prefix + "output_values",
                    reserve_buffer=reserve)
            else:
                output_values = None

            compiled_kernel(
                input_values,
                row_states,
                seq_lens,
                output_indices_torch,
                output_values,
            )

            return output_indices_torch, output_values

    class CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner(
            CuteDSLTopKDecodeSinglePassMultiCTARunner):
        """Runner for cluster-accelerated single-pass multi-CTA radix top-k.

        Uses Blackwell cluster barriers and DSMEM for inter-CTA histogram
        merging instead of global memory atomics.  Only 1 int32 per group
        is needed in global memory (the output counter).

        Inherits compile, chunk heuristics, and forward from the base runner;
        overrides _get_chunk_config (cluster-size clamping) and forward
        (unsupported-size fallback).
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()
        _row_states_initialized = False
        _row_states_buffer_name = "sp_mcta_cluster_row_states"
        _buf_prefix = "sp_mcta_cluster_"
        _kernel_class = SinglePassMultiCTARadixTopKClusterKernel
        _state_size = CLUSTER_TOPK_STATE_SIZE

        @classmethod
        def _get_chunk_config(cls,
                              dtype,
                              num_cols: int,
                              chunk_size: Optional[int] = None,
                              num_copy_bits: int = 256,
                              num_rows: int = 1):
            """Resolve chunk_size and ctas_per_group, clamped to hw max cluster.

            Returns:
                (chunk_size, ctas_per_group, vec_size) or (None, None, None)
            """
            chunk_size, ctas_per_group, vec_size = super()._get_chunk_config(
                dtype, num_cols, chunk_size, num_copy_bits, num_rows)

            hw_max_cluster = _query_max_cluster_size()
            if ctas_per_group > hw_max_cluster:
                max_chunk, vec_size = cls._compute_max_chunk(
                    dtype, num_copy_bits)
                chunk_size = math.ceil(num_cols / hw_max_cluster)
                chunk_size = (
                    (chunk_size + vec_size - 1) // vec_size) * vec_size
                if chunk_size > max_chunk:
                    logger.warning(
                        f"Cluster top-k: num_cols={num_cols} requires "
                        f"chunk_size={chunk_size} which exceeds max shared "
                        f"memory capacity ({max_chunk}). Cannot handle this "
                        f"problem size with cluster kernel.")
                    return None, None, None
                ctas_per_group = math.ceil(num_cols / chunk_size)

            return chunk_size, ctas_per_group, vec_size

        @classmethod
        def forward(
            cls,
            input_values: torch.Tensor,
            seq_lens: torch.Tensor,
            top_k: int,
            next_n: int,
            return_val: bool = False,
            num_copy_bits: int = 256,
            chunk_size: Optional[int] = None,
            output_indices: Optional[torch.Tensor] = None,
        ):
            """Execute cluster-accelerated single-pass multi-CTA radix top-k.

            Returns (None, None) if the problem size exceeds what the cluster
            kernel can handle (caller should fall back to the non-cluster runner).
            """
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_cols = input_values.shape[1]

            max_chunk, _ = cls._compute_max_chunk(dtype, num_copy_bits)
            hw_max_cluster = _query_max_cluster_size()
            max_supported_cols = max_chunk * hw_max_cluster
            if num_cols > max_supported_cols:
                logger.warning(
                    f"Cluster top-k does not support num_cols={num_cols} "
                    f"(max supported: {max_supported_cols} = "
                    f"max_chunk={max_chunk} x max_cluster={hw_max_cluster} "
                    f"for dtype={torch_dtype}). "
                    f"Falling back to non-cluster runner.")
                return None, None

            result = super().forward(input_values, seq_lens, top_k, next_n,
                                     return_val, num_copy_bits, chunk_size,
                                     output_indices)
            if result[0] is None:
                return None, None
            return result

    @torch.library.custom_op("trtllm::cute_dsl_topk_decode_multi_cta_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_topk_decode_multi_cta_blackwell(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        chunk_size_per_cta: int = 16384,
        dynamic: bool = True,
    ) -> torch.Tensor:
        """CuteDSL-based multi-CTA Top-K selection optimized for Blackwell decode phase.

        Splits each row into chunks processed by separate CTAs, then merges results.
        Suitable for large vocabulary sizes where single-CTA is insufficient.

        Args:
            input_values: Input logits tensor [batch_size * next_n, vocab_size]
            seq_lens: Sequence lengths for each batch [batch_size]
            top_k: Number of top elements to select (max 16384)
            next_n: Number of candidates per sequence (for speculative decoding)
            num_copy_bits: Number of bits for vectorized memory copy (128 or 256)
            chunk_size_per_cta: Number of columns each CTA processes
            dynamic: Use dynamic multi-CTA scheduling (1D grid + binary search)

        Returns:
            indices: Top-k indices [batch_size * next_n, top_k]

        Note:
            This function requires Blackwell architecture (SM100+) and CuTE DSL support.
        """
        # Validate SM version
        sm_version = get_sm_version()
        if sm_version < 100:
            raise ValueError(
                f"CuTE DSL top-k requires Blackwell (SM100+), but got SM {sm_version}. "
                "Use standard top-k implementation for older architectures.")

        # Validate inputs
        if top_k <= 0 or top_k > 16384:
            raise ValueError(
                f"top_k must be in range [1, 16384], got {top_k}. "
                "Maximum supported top_k is 16384 for Blackwell architecture.")

        if next_n <= 0:
            raise ValueError(f"next_n must be positive, got {next_n}")

        if num_copy_bits not in [128, 256]:
            raise ValueError(
                f"num_copy_bits must be 128 or 256, got {num_copy_bits}")

        if chunk_size_per_cta <= 0:
            raise ValueError(
                f"chunk_size_per_cta must be positive, got {chunk_size_per_cta}"
            )

        if input_values.dim() != 2:
            raise ValueError(
                f"input_values must be 2D [num_rows, vocab_size], got shape {input_values.shape}"
            )

        if seq_lens.dim() != 1:
            raise ValueError(
                f"seq_lens must be 1D [batch_size], got shape {seq_lens.shape}")

        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if input_values.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported dtype {input_values.dtype}. "
                             f"Supported dtypes: {supported_dtypes}")

        indices, _ = CuteDSLTopKDecodeMultiCTARunner.forward(
            input_values=input_values,
            seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=False,  # Only return indices
            num_copy_bits=num_copy_bits,
            chunk_size_per_cta=chunk_size_per_cta,
            dynamic=dynamic,
        )
        return indices

    @torch.library.register_fake(
        "trtllm::cute_dsl_topk_decode_multi_cta_blackwell")
    def _(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        chunk_size_per_cta: int = 16384,
        dynamic: bool = True,
    ):
        num_rows = input_values.shape[0]

        indices = input_values.new_empty((num_rows, top_k), dtype=torch.int32)
        return indices

    def _radix_select_preferred(dtype, num_tokens: int, num_rows: int) -> bool:
        """radix-SELECT SP beats radix-FILTER at small N: select loads the whole
        chunk into SMEM and radix-selects in place, which is leaner than filter's
        histogram + refine when the chunk fits SMEM and the fixed overhead is not
        amortized over few elements. SMEM-capacity driven, so distribution-robust.
        fp32 (4B) fills SMEM at half the N and its 4 refine rounds favor filter ->
        always filter. Tuned on a B200 randn sweep (filter-vs-select, best-vs-best
        + tuned): bf16 wins up to ~32%, fp16 up to ~12%.

        bf16 N=32768 is batch-split: select wins only at large batch (grid
        pressure forces both to single-CTA, where select single beats filter
        single); at small batch filter's cluster is better, so keep filter.
        """
        # Large batch (num_rows > num_sms) forces the filter path to single-CTA
        # (auto_cluster_size -> cs=1), which beats the radix-SELECT cluster runner
        # here; only prefer select while the batch still fits within one SM wave.
        num_sms = _get_num_sms()
        if dtype == torch.bfloat16:
            return num_rows <= num_sms and (num_tokens <= 16384 or
                                            (num_tokens == 32768
                                             and num_rows >= 74))
        if dtype == torch.float16:
            return num_rows <= num_sms and num_tokens <= 16384
        return False

    @torch.library.custom_op("trtllm::cute_dsl_indexer_topk_decode",
                             mutates_args=("output_indices", ),
                             device_types="cuda")
    def cute_dsl_indexer_topk_decode(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        dynamic: bool = True,
        single_pass_multi_cta: bool = False,
        single_pass_multi_cta_cluster: bool = False,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
        radix_filter_single_pass_multi_cta: bool = True,
    ) -> None:
        """Unified CuTE DSL decode Top-K. Writes results directly into the
        pre-allocated ``output_indices`` buffer.

        Three mutually exclusive dispatch modes, selected by boolean with
        precedence ``radix_filter_single_pass_multi_cta`` >
        ``single_pass_multi_cta`` > 2-pass (evaluated as an if/elif chain, so
        the first True wins and the others are ignored):

        1. ``radix_filter_single_pass_multi_cta=True`` (default) -- ADAPTIVE,
           best-performance path. Internally auto-selects among radix-SELECT
           (small N), radix-FILTER single-pass multi-CTA cluster, and
           single-CTA by (dtype, N, num_rows, overflow_policy); no caller
           tuning needed. Prefer this. When it is enabled the other two mode
           booleans MUST be left False (asserted below) -- they would be
           silently ignored otherwise.
        2. ``single_pass_multi_cta=True`` (only when mode 1 is False) -- legacy
           single-pass multi-CTA path (kept for A/B and fallback). Auto-selects
           single-CTA vs single-pass multi-CTA by the SM-wave heuristic below;
           ``single_pass_multi_cta_cluster=True`` forces the cluster variant
           within this mode (no effect in the other modes).
        3. neither set -- legacy 2-pass multi-CTA path (A/B and fallback);
           vocab-threshold + SM-utilization heuristic.

        Dispatch logic (``single_pass_multi_cta=True`` path):

        The key insight is that the single-pass multi-CTA kernel wins when all CTAs fit
        in a single SM wave (no inter-CTA barrier serialization across waves).
        For fp32, the 4 radix rounds double the sync overhead vs fp16/bf16's
        2 rounds, so the crossover favors single-CTA much earlier.

        - **ctas_per_group >= 2** (single-pass multi-CTA):
          Use single-pass multi-CTA when ``num_rows * ctas_per_group <= num_sms``
          (single wave). For fp32, additionally require ``vocab >= 65536``
          since smaller vocab doesn't benefit enough from parallelism.
        - **ctas_per_group == 1** (effectively single-CTA single-pass multi-CTA):
          fp16/bf16: use single-pass multi-CTA when ``num_rows <= num_sms`` (no
          inter-CTA sync, single-pass multi-CTA kernel is faster due to better
          memory access patterns).
          fp32: always use single-CTA (single-pass multi-CTA overhead not worth it).

        When ``single_pass_multi_cta_cluster=True`` (requires ``single_pass_multi_cta=True``),
        the cluster-accelerated variant (DSMEM + cluster barriers) is used unconditionally
        instead of the auto cluster/distributed heuristic.

        Benchmark: overhead vs oracle ~2.4%, speedup vs always-single ~1.14x
        (Blackwell SM100 148 SMs, top_k=2048, fp32/bf16/fp16).

        Legacy dispatch (``single_pass_multi_cta=False``) uses the original vocab
        threshold + SM utilization heuristic for the 2-pass multi-CTA kernel.

        Args:
            input_values: Input logits tensor [batch_size * next_n, vocab_size]
            seq_lens: Sequence lengths for each batch [batch_size]
            output_indices: Pre-allocated output buffer [batch_size * next_n, top_k]
            top_k: Number of top elements to select (max 16384)
            next_n: Number of candidates per sequence (for speculative decoding)
            num_copy_bits: Number of bits for vectorized memory copy (128 or 256)
            dynamic: Use dynamic multi-CTA scheduling (for 2-pass multi-CTA)
            single_pass_multi_cta: Mode-2 override -- use the legacy single-pass
                multi-CTA path. Only takes effect when
                radix_filter_single_pass_multi_cta=False.
            single_pass_multi_cta_cluster: Force the cluster-accelerated variant
                within mode 2 (only effective when single_pass_multi_cta=True).
            overflow_policy: Threshold-bucket SMEM overflow handling
                ("REREAD" default, exact). See FilteredTopKKernelVarlen.
            cache_smem_values: Cache ordered values in SMEM to skip a reload.
            radix_filter_single_pass_multi_cta: Mode-1 (default True) -- the
                adaptive best-performance path; see the mode list above. Set
                False to select mode 2 or 3.
        """
        # Validate inputs
        if top_k <= 0 or top_k > 16384:
            raise ValueError(
                f"top_k must be in range [1, 16384], got {top_k}. "
                "Maximum supported top_k is 16384 for Blackwell architecture.")

        num_rows = input_values.shape[0]
        num_tokens = input_values.shape[1]

        if radix_filter_single_pass_multi_cta:
            # Mode 1 is the adaptive default and dominates the if/elif chain
            # below. Reject a conflicting mode-2/3 override rather than silently
            # ignoring it (set radix_filter_single_pass_multi_cta=False to opt
            # into the legacy single_pass_multi_cta / 2-pass paths).
            assert not single_pass_multi_cta and not single_pass_multi_cta_cluster, (
                "radix_filter_single_pass_multi_cta (adaptive default) takes "
                "precedence over single_pass_multi_cta / "
                "single_pass_multi_cta_cluster; set it False to use those "
                "legacy overrides.")
            _R = CuteDSLTopKDecodeRadixFilterSPMultiCTARunner
            _is_fp32 = input_values.dtype == torch.float32
            # At small N (bf16 <= 32K, fp16 <= 16K) radix-SELECT SP beats
            # radix-FILTER; route there. The select cluster runner auto-picks
            # ctas (=1 => select single-CTA for small batch, else cluster), so
            # this covers both. Falls through to filter only if select can't
            # fit the problem (capacity -> None), which small N never hits.
            if _radix_select_preferred(input_values.dtype, num_tokens,
                                       num_rows):
                _sel = CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    output_indices=output_indices,
                )
                if _sel[0] is not None:
                    return
            # radix-FILTER single-CTA vs SP multi-CTA (cluster DSMEM). Heuristic
            # is overflow-policy-coupled: REREAD has a large_occupancy re-scan
            # cliff, cliff-free policies don't -> pick the matching tune.
            if overflow_policy == "REREAD":
                cluster_size = _R.auto_cluster_size(num_tokens, num_rows,
                                                    _is_fp32)
            else:
                cluster_size = _R.auto_cluster_size_truncate(
                    num_tokens, num_rows, _is_fp32)
            if cluster_size >= 2:
                CuteDSLTopKDecodeRadixFilterSPMultiCTARunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    cluster_size=cluster_size,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    overflow_policy=overflow_policy,
                    output_indices=output_indices,
                    cache_smem_values=cache_smem_values,
                )
            else:
                CuteDSLTopKDecodeSingleCTARunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    overflow_policy=overflow_policy,
                    output_indices=output_indices,
                    cache_smem_values=cache_smem_values,
                )
        elif single_pass_multi_cta:
            # --- heuristic for single-CTA vs single-pass multi-CTA ---
            # Determines whether the single-pass multi-CTA kernel
            # is faster than single-CTA based on SM wave occupancy analysis.
            #
            # Core rules:
            # 1. ctas_per_group >= 2: single-pass multi-CTA wins iff all CTAs fit in one
            #    SM wave (num_rows * ctas_per_group <= num_sms). Multi-wave
            #    causes inter-CTA barrier serialization → perf collapse.
            #    For fp32, also require vocab >= 65536 (small vocab: sync
            #    overhead from 4 radix rounds > parallelism benefit).
            # 2. ctas_per_group == 1: no inter-CTA sync needed.
            #    fp16/bf16: single-pass multi-CTA wins when num_rows <= num_sms.
            #    fp32: single-CTA always wins (single-pass multi-CTA overhead too high).
            is_fp32 = (input_values.dtype == torch.float32)

            # Short-circuit: fp32 with small vocab never benefits from
            # single-pass multi-CTA (sync overhead from 4 radix rounds > parallelism
            # gain). Skip _get_chunk_config entirely.
            if is_fp32 and num_tokens < 65536:
                use_single_pass_multi_cta = False
            else:
                num_sms = _get_num_sms()
                cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[input_values.dtype]
                _, ctas_per_group, _ = (
                    CuteDSLTopKDecodeSinglePassMultiCTARunner._get_chunk_config(
                        cutlass_dtype,
                        num_tokens,
                        num_copy_bits=num_copy_bits,
                        num_rows=num_rows))

                if ctas_per_group >= 2:
                    use_single_pass_multi_cta = (num_rows * ctas_per_group
                                                 <= num_sms)
                    if is_fp32:
                        use_single_pass_multi_cta = (use_single_pass_multi_cta
                                                     and num_tokens >= 65536)
                else:  # ctas_per_group == 1
                    use_single_pass_multi_cta = (not is_fp32
                                                 and num_rows <= num_sms)

            if use_single_pass_multi_cta:
                # Use cluster variant when explicitly requested or when
                # SM resources are sufficient (small batch); fall back to
                # distributed (global memory atomics) for large batch.
                # TODO:
                # use_cluster = (single_pass_multi_cta_cluster
                #                or num_rows * ctas_per_group <= num_sms * 2)
                use_cluster = (single_pass_multi_cta_cluster)
                if use_cluster:
                    result = CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner.forward(
                        input_values=input_values,
                        seq_lens=seq_lens,
                        top_k=top_k,
                        next_n=next_n,
                        return_val=False,
                        num_copy_bits=num_copy_bits,
                        output_indices=output_indices,
                    )
                    if result[0] is None:
                        use_cluster = False

                if not use_cluster:
                    CuteDSLTopKDecodeSinglePassMultiCTARunner.forward(
                        input_values=input_values,
                        seq_lens=seq_lens,
                        top_k=top_k,
                        next_n=next_n,
                        return_val=False,
                        num_copy_bits=num_copy_bits,
                        output_indices=output_indices,
                    )
            else:
                CuteDSLTopKDecodeSingleCTARunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    overflow_policy=overflow_policy,
                    output_indices=output_indices,
                    cache_smem_values=cache_smem_values,
                )
        else:
            # --- 2-pass multi-CTA dispatch ---
            # Kept for A/B comparison and as fallback when single_pass_multi_cta=False.
            # Uses vocab threshold + SM utilization < 25% heuristic.
            chunk_size_per_cta = 16384

            # Multi-CTA vocab thresholds by dtype.
            # fp32: multi-CTA wins at vocab >= 65536 (4+ CTAs per row)
            # fp16/bf16: multi-CTA wins at vocab >= 131072 (8+ CTAs per row)
            if input_values.dtype == torch.float32:
                use_multi_cta = num_tokens >= 65536
            else:
                use_multi_cta = num_tokens >= 131072

            # Only use multi-CTA when SM utilization from single-CTA is low
            # (< 25%). Beyond this, single-CTA already saturates the SMs and
            # multi-CTA 2-pass overhead hurts.
            if use_multi_cta:
                num_sms = _get_num_sms()
                use_multi_cta = num_rows < num_sms // 4

            if use_multi_cta:
                CuteDSLTopKDecodeMultiCTARunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    chunk_size_per_cta=chunk_size_per_cta,
                    overflow_policy=overflow_policy,
                    dynamic=dynamic,
                    output_indices=output_indices,
                    cache_smem_values=cache_smem_values,
                )
            else:
                CuteDSLTopKDecodeSingleCTARunner.forward(
                    input_values=input_values,
                    seq_lens=seq_lens,
                    top_k=top_k,
                    next_n=next_n,
                    return_val=False,
                    num_copy_bits=num_copy_bits,
                    overflow_policy=overflow_policy,
                    output_indices=output_indices,
                    cache_smem_values=cache_smem_values,
                )

    @torch.library.register_fake("trtllm::cute_dsl_indexer_topk_decode")
    def _(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        dynamic: bool = True,
        single_pass_multi_cta: bool = False,
        single_pass_multi_cta_cluster: bool = False,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
        radix_filter_single_pass_multi_cta: bool = True,
    ) -> None:
        return None

    def warmup_cute_dsl_radix_topk_decode(
        top_k: int,
        num_cols: int,
        next_n: int = 1,
        dtype: torch.dtype = torch.float32,
        num_copy_bits: int = 256,
        num_sms: Optional[int] = None,
    ) -> None:
        """Pre-compile the radix-filter DSL decode top-k for every
        ``cluster_size`` variant the runtime dispatch can pick for this
        deployment.

        ``cute_dsl_indexer_topk_decode`` JIT-compiles a fresh CuTe DSL kernel
        per compile-key ``(dtype, bucketed_num_cols, top_k, next_n, ...,
        cluster_size)`` on first touch (~seconds). Every key dimension is
        fixed for a deployment (``num_cols = indexer_max_seq_len``) except
        ``cluster_size = auto_cluster_size(num_cols, num_rows, ...)``, which
        steps across the coarse ``num_rows`` occupancy bands
        (<=4 / <=8 / <=32 / <=64 / <=num_sms / >num_sms). CUDA-graph warmup
        only exercises ``cuda_graph_batch_sizes``; eager iters (mixed
        prefill+decode batch, or ``cuda_graph`` disabled) whose ``num_rows``
        lands in an uncovered band otherwise pay the JIT stall on a live
        request. Issuing one decode per representative ``num_rows`` funnels
        every ``cluster_size`` compile into warmup — the op's own dispatch
        (radix-SELECT / radix-FILTER cluster / single-CTA) picks and compiles
        exactly what the runtime would.

        Meant to run during warmup, before serving. Captured geometries are
        already compiled by the warmup-step forwards; this fills in the bands
        the eager (non-captured) path can still hit. Best-effort: per-band
        failures are logged and skipped so one broken bucket does not abort
        startup.
        """
        if top_k <= 0 or num_cols <= 0 or next_n <= 0:
            return
        num_sms = num_sms or _get_num_sms()
        device = torch.device("cuda")
        # One representative num_rows per auto_cluster_size occupancy band.
        # Rounded up to a multiple of next_n (kernel shape contract:
        # num_rows % next_n == 0); identical num_rows are de-duplicated.
        band_targets = (4, 8, 32, 64, num_sms, num_sms + 1)
        seen = set()
        for target in band_targets:
            num_gen = max(1, -(-target // next_n))  # ceil(target / next_n)
            num_rows = num_gen * next_n
            if num_rows in seen:
                continue
            seen.add(num_rows)
            logits = torch.zeros((num_rows, num_cols),
                                 dtype=dtype,
                                 device=device)
            seq_lens = torch.full((num_gen, ),
                                  num_cols,
                                  dtype=torch.int32,
                                  device=device)
            output_indices = torch.empty((num_rows, top_k),
                                         dtype=torch.int32,
                                         device=device)
            try:
                torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                    logits, seq_lens, output_indices, top_k, next_n,
                    num_copy_bits)
            except RuntimeError as e:
                logger.warning(
                    f"[DSL topk warmup] radix-filter prewarm failed for "
                    f"num_rows={num_rows} (num_cols={num_cols}, top_k={top_k}, "
                    f"next_n={next_n}); skipping band. "
                    f"{type(e).__name__}: {e}")
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    #  CuTe DSL GVR Top-K Decode                                         #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import \
        GvrTopKKernel as _GvrTopKKernel
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_dispatch import \
        is_tiered_topk_supported as _is_tiered_topk_supported
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_dispatch import \
        tiered_topk as _tiered_topk

    class CuteDSLGvrTopKDecodeRunner:
        """Runner for the GVR Top-K cuTe DSL kernel (Blackwell SM100).

        Owns the JIT cache and the (T, V, min_blocks_per_mp,
        warp_parallel_reduce) heuristic. ``forward()`` dispatches three
        paths from ``(counters, order_row)`` — see its docstring. All
        share :meth:`_pick_tuning`; only the compiled kernel class and
        launch signature differ.
        """
        kernel_cache: dict = {}

        @staticmethod
        def _pick_tuning(
            torch_dtype: torch.dtype,
            num_rows: int,
            N_per_cta: int,
            num_sms: int,
            max_seq_len: Optional[int],
            data_ptr: int,
        ) -> dict:
            """Adapter over :meth:`GvrTopKKernel.pick_tuning` (the single
            source of truth for the T / V / min_blocks_per_mp /
            warp-reduce policy), shared by the single-CTA / sort and LB
            compile paths. Returned keys match ``_compile`` /
            ``_compile_lb`` param names for ``**tuning`` spreading.

            Intentional shell divergence from ``GvrTopKKernel.launch``:
            a 32B-misaligned logits pointer is a CONTRACT VIOLATION here
            (assert), while ``launch`` silently downgrades to 128-bit
            loads (dev convenience for ad-hoc tensors).
            """
            cfg = _GvrTopKKernel.pick_tuning(
                torch_dtype,
                num_rows,
                N_per_cta,
                num_sms,
                graph_capture=max_seq_len is not None,
            )
            if cfg["use_256bit_load"]:
                assert data_ptr % 32 == 0, (
                    f"use_256bit_load=True requires 32B-aligned "
                    f"logits.data_ptr(), got {data_ptr} % 32 = "
                    f"{data_ptr % 32}.")
            return dict(
                enable_unroll_4=True,
                enable_phase3_unroll=True,
                use_constant_hint=False,
                num_threads_per_block=cfg["num_threads"],
                use_256bit_load=cfg["use_256bit_load"],
                enable_warp_parallel_reduce=cfg["enable_warp_parallel_reduce"],
                min_blocks_per_mp=cfg["min_blocks_per_mp"],
            )

        @classmethod
        def _compile(
            cls,
            dtype,
            top_k: int,
            next_n: int,
            enable_unroll_4: bool,
            enable_phase3_unroll: bool,
            use_constant_hint: bool,
            min_blocks_per_mp: int,
            use_256bit_load: bool,
            num_threads_per_block: int,
            enable_warp_parallel_reduce: bool,
            compress_ratio: int,
            return_output_values: bool,
            cluster_size: int,
            seqlen_sorted: bool,
            enable_block_skip: bool = False,
            use_ext_counts: bool = False,
            emit_xstate: bool = False,
            use_ext_cand: bool = False,
            ext_rungs: bool = False,
            cand_cap: int = 5120,
            accept_cap: Optional[int] = None,
            kc_override: Optional[int] = None,
        ) -> tuple:
            key = (dtype, top_k, next_n, enable_unroll_4, enable_phase3_unroll,
                   use_constant_hint, min_blocks_per_mp, use_256bit_load,
                   num_threads_per_block, enable_warp_parallel_reduce,
                   compress_ratio, return_output_values, cluster_size,
                   seqlen_sorted, enable_block_skip, use_ext_counts,
                   emit_xstate, use_ext_cand, ext_rungs, cand_cap, accept_cap,
                   kc_override)
            if key in cls.kernel_cache:
                return key
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            # 32B alignment required by 256-bit vec loads.
            in_align = 32 if use_256bit_load else 16
            input_fake = cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=in_align)
            pre_idx_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            seq_lens_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, ), stride_order=(0, ))
            # None → kernel skips STG.value path (cute.compile won't
            # materialize the fake either).
            out_values_fake = (cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, top_k), stride_order=(1, 0), assumed_align=16)
                               if return_output_values else None)
            out_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            # seqlen_sorted=False → const_expr's the indirection out, no
            # order_row read at runtime. True → request-level fake
            # (shape n_batch, not n_rows).
            order_row_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_batch, ), stride_order=(0, )) if seqlen_sorted else None)
            # emission-assisted tier fake tensors (list/counts/rungs)
            block_max_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (n_rows, cute.sym_int()),
                stride_order=(1, 0),
                assumed_align=16) if enable_block_skip else None)
            seed_thr_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (n_rows, 8 if use_ext_counts else 3),
                stride_order=(1, 0),
                assumed_align=4) if (use_ext_counts or ext_rungs) else None)
            xstate_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (n_rows, 8),
                stride_order=(1, 0),
                assumed_align=4) if emit_xstate else None)
            cand_vals_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Float32, (n_rows, cand_cap),
                stride_order=(1, 0),
                assumed_align=4) if use_ext_cand else None)
            cand_idx_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, cand_cap),
                stride_order=(1, 0),
                assumed_align=4) if use_ext_cand else None)
            cand_ctl_fake = (cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, 4),
                stride_order=(1, 0),
                assumed_align=8) if use_ext_cand else None)
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            kernel = _GvrTopKKernel(
                dtype=dtype,
                top_k=top_k,
                next_n=next_n,
                num_threads=num_threads_per_block,
                enable_unroll_4=enable_unroll_4,
                enable_phase3_unroll=enable_phase3_unroll,
                use_constant_hint=use_constant_hint,
                min_blocks_per_mp=min_blocks_per_mp,
                use_256bit_load=use_256bit_load,
                enable_warp_parallel_reduce=enable_warp_parallel_reduce,
                compress_ratio=compress_ratio,
                return_output_values=return_output_values,
                cluster_size=cluster_size,
                seqlen_sorted=seqlen_sorted,
                enable_block_skip=enable_block_skip,
                use_ext_counts=use_ext_counts,
                emit_xstate=emit_xstate,
                use_ext_cand=use_ext_cand,
                ext_rungs=ext_rungs,
                cand_cap=cand_cap,
                accept_cap=accept_cap,
                kc_override=kc_override,
                # ext modes need 3 rung slots (M_thr == 3); only the
                # slot count matters, not the qfrac values (P1b skipped)
                r0_qfracs=((0.85, 0.35) if
                           (use_ext_counts or ext_rungs) else None),
            )
            cls.kernel_cache[key] = cute.compile(
                kernel,
                input_fake,
                pre_idx_fake,
                seq_lens_fake,
                out_values_fake,
                out_indices_fake,
                order_row_fake,
                stream=fake_stream,
                block_max=block_max_fake,
                seed_thr=seed_thr_fake,
                seed_counts=None,
                xstate=xstate_fake,
                cand_vals=cand_vals_fake,
                cand_idx=cand_idx_fake,
                cand_ctl=cand_ctl_fake,
                options="--enable-tvm-ffi",
            )
            logger.debug(f"[compile cute_dsl gvr_topk_decode] {key}")
            return key

        @classmethod
        def _compile_lb(
            cls,
            dtype,
            top_k: int,
            next_n: int,
            compress_ratio: int,
            max_batch_size: int,
            num_threads_per_block: int,
            cluster_size: int,
            enable_unroll_4: bool,
            enable_phase3_unroll: bool,
            use_constant_hint: bool,
            min_blocks_per_mp: int,
            use_256bit_load: bool,
            enable_warp_parallel_reduce: bool,
            return_output_values: bool,
        ) -> tuple:
            """JIT-compile the LB (hybrid multi-CTA + single-CTA) kernel.

            ``num_rows`` / ``N`` are ``cute.sym_int()`` — one compiled
            kernel covers all shapes within a tuning bucket. Grid is
            sized by ``max_batch_size`` (which IS in the cache key).
            """
            key = ("lb", dtype, top_k, next_n, compress_ratio, max_batch_size,
                   num_threads_per_block, cluster_size, enable_unroll_4,
                   enable_phase3_unroll, use_constant_hint, min_blocks_per_mp,
                   use_256bit_load, enable_warp_parallel_reduce,
                   return_output_values)
            if key in cls.kernel_cache:
                return key
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            in_align = 32 if use_256bit_load else 16
            input_fake = cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=in_align)
            pre_idx_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            seq_lens_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, ), stride_order=(0, ))
            out_values_fake = (cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, top_k), stride_order=(1, 0), assumed_align=16)
                               if return_output_values else None)
            out_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            order_row_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (max_batch_size, ), stride_order=(0, ))
            counters_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (2, ), stride_order=(0, ))
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)
            kernel = _GvrTopKLBKernel(
                dtype=dtype,
                top_k=top_k,
                next_n=next_n,
                num_threads=num_threads_per_block,
                compress_ratio=compress_ratio,
                return_output_values=return_output_values,
                cluster_size=cluster_size,
                max_batch_size=max_batch_size,
                enable_unroll_4=enable_unroll_4,
                enable_phase3_unroll=enable_phase3_unroll,
                use_constant_hint=use_constant_hint,
                min_blocks_per_mp=min_blocks_per_mp,
                use_256bit_load=use_256bit_load,
                enable_warp_parallel_reduce=enable_warp_parallel_reduce,
            )
            cls.kernel_cache[key] = cute.compile(
                kernel,
                input_fake,
                pre_idx_fake,
                seq_lens_fake,
                out_values_fake,
                out_indices_fake,
                order_row_fake,
                counters_fake,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
            logger.debug(f"[compile cute_dsl gvr_topk_lb_decode] {key}")
            return key

        @classmethod
        def forward(
            cls,
            logits: torch.Tensor,
            pre_idx: torch.Tensor,
            seq_lens: torch.Tensor,
            output_indices: torch.Tensor,
            top_k: int,
            next_n: int = 1,
            compress_ratio: int = 1,
            max_seq_len: Optional[int] = None,
            cluster_size: Optional[int] = None,
            order_row: Optional[torch.Tensor] = None,
            counters: Optional[torch.Tensor] = None,
            max_batch_size: Optional[int] = None,
            seed_thr: Optional[torch.Tensor] = None,
            xstate: Optional[torch.Tensor] = None,
            cand_vals: Optional[torch.Tensor] = None,
            cand_idx: Optional[torch.Tensor] = None,
            cand_ctl: Optional[torch.Tensor] = None,
            block_max: Optional[torch.Tensor] = None,
            num_threads: Optional[int] = None,
            accept_cap: Optional[int] = None,
            kc_override: Optional[int] = None,
        ) -> None:
            """Three paths, picked by ``(counters, order_row)``:

            - (None, None)   single-CTA.
            - (None, tensor) single-CTA + sort indirect; ``order_row`` is a
              descending argsort of ``seq_lens`` (shape == seq_lens.shape).
            - (tensor, tensor) LB; ``order_row`` is the long-first partition
              from ``cute_dsl_gvr_topk_lb_prepare`` (shape == max_batch_size;
              valid prefix in ``counters`` = [n_long, n_short]).

            ``counters`` without ``order_row`` is rejected.
            """
            # Tiered-GVR fast path: fp32 / next_n >= 1 (MTP) /
            # cr in {1, 4} / npad <= 262144 decode rows route to the
            # direct/reg/tp CuTe DSL tiers; everything else (half-prec, LB,
            # oversize npad, hw cluster cap) falls through to the in-tree
            # kernel below. ``order_row`` (the LJF hint dsa.py computes for
            # num_rows >= 2 * num_sms) is accepted and ignored: the GVR
            # tiers launch per-row CTAs and do not consume the permutation.
            # Host-only guard — no device sync. The op signature and output
            # contract are unchanged (unordered int32 indices, -1 pad only
            # for degenerate rows).
            if (seed_thr is None and cand_vals is None and xstate is None
                    and block_max is None and _is_tiered_topk_supported(
                        logits, pre_idx, seq_lens, output_indices, top_k,
                        next_n, compress_ratio, order_row, counters)):
                _tiered_topk(logits, pre_idx, seq_lens, output_indices, top_k,
                             next_n, compress_ratio)
                return

            cute_dtype = _TORCH_TO_CUTLASS_DTYPE[logits.dtype]
            num_rows = logits.shape[0]
            # seq_lens is request-level, logits is row-level (next_n
            # rows per request).
            assert num_rows % next_n == 0 and seq_lens.shape[
                0] == num_rows // next_n, (
                    f"shape contract: seq_lens.shape[0] (={seq_lens.shape[0]}) "
                    f"must equal logits.shape[0] / next_n "
                    f"(={num_rows} / {next_n} = {num_rows // next_n})")
            # DSA indexer only reads indices (mirrors CUDA
            # indexer_topk_decode). Kernel keeps True/False branches.
            return_output_values = False
            # Under graph capture, max_seq_len = peak runtime N so the
            # heuristic picks the large-N variant.
            N_row = max_seq_len if max_seq_len is not None else logits.shape[1]
            num_sms = _get_num_sms()

            # cluster_size policy:
            #   LB: caller-pinned in {2,4,8}; baked into cache key →
            #       reject (not clamp) on hw mismatch.
            #   single-CTA / sort: auto-pick from (N, BS) when unset;
            #       safe to clamp to hw cap.
            lb_mode = counters is not None
            if lb_mode:
                assert order_row is not None, (
                    "counters requires order_row (both come from "
                    "trtllm::cute_dsl_gvr_topk_lb_prepare).")
                assert max_batch_size is not None, (
                    "max_batch_size is required in LB mode and must "
                    "match the value used at LB prepare time.")
                assert (order_row.dtype == torch.int32 and order_row.is_cuda
                        and order_row.shape == (max_batch_size, )), (
                            f"LB order_row must be int32, CUDA, shape "
                            f"({max_batch_size},); got dtype={order_row.dtype} "
                            f"shape={tuple(order_row.shape)}")
                assert (counters.dtype == torch.int32 and counters.is_cuda
                        and counters.shape == (2, )), (
                            f"LB counters must be int32, CUDA, shape (2,); "
                            f"got dtype={counters.dtype} "
                            f"shape={tuple(counters.shape)}")
                if cluster_size is None:
                    cluster_size = 4  # GvrTopKLBKernel ctor default
                assert cluster_size in (2, 4, 8), (
                    f"LB cluster_size must be 2, 4, or 8; got {cluster_size}")
                hw_max_cluster = _query_max_cluster_size()
                if cluster_size > hw_max_cluster:
                    raise ValueError(
                        f"LB cluster_size={cluster_size} exceeds device "
                        f"max ({hw_max_cluster}); pin a smaller cs at LB "
                        f"prepare, or use the single-CTA path.")
            else:
                if cluster_size is None:
                    cluster_size = _GvrTopKKernel.pick_cluster_size(
                        num_rows, N_row, num_sms)
                if cluster_size > 1:
                    hw_max_cluster = _query_max_cluster_size()
                    if cluster_size > hw_max_cluster:
                        logger.warning_once(
                            f"cute_dsl_gvr_topk_decode: cluster_size="
                            f"{cluster_size} exceeds device max "
                            f"({hw_max_cluster}); clamping.",
                            key="cute_dsl_gvr_topk_decode_cluster_clamp",
                        )
                        cluster_size = hw_max_cluster

            # Cluster CTAs split the row, so heuristics target per-CTA work.
            N_per_cta = N_row // cluster_size
            # tuning keys mirror _compile / _compile_lb param names; spread
            # with **tuning at the call sites.
            tuning = cls._pick_tuning(logits.dtype, num_rows, N_per_cta,
                                      num_sms, max_seq_len, logits.data_ptr())

            if lb_mode:
                key = cls._compile_lb(
                    cute_dtype,
                    top_k,
                    next_n,
                    compress_ratio,
                    max_batch_size=max_batch_size,
                    cluster_size=cluster_size,
                    return_output_values=return_output_values,
                    **tuning,
                )
                cls.kernel_cache[key](logits, pre_idx, seq_lens, None,
                                      output_indices, order_row, counters)
                return

            # seqlen_sorted=True compiles in order_row[req] * next_n + nn
            # (longer rows first); False const_expr's it out.
            seqlen_sorted = order_row is not None
            if seqlen_sorted:
                assert (
                    order_row.dtype == torch.int32 and order_row.is_cuda
                    and order_row.shape == seq_lens.shape
                ), ("order_row must be int32, CUDA, shape == seq_lens.shape "
                    f"(={tuple(seq_lens.shape)}); got dtype={order_row.dtype} "
                    f"shape={tuple(order_row.shape)}")
            # emission-assisted tiers: mode from which ext tensors the
            # caller handed in (see gvr_routing.plan_emission)
            if seed_thr is not None:
                assert seed_thr.shape[1] == 3 or seed_thr.shape[1] == 8, (
                    "seed_thr must be [rows, 3] (rungs) or [rows, 8] "
                    "(packed lines + counts row, the width the kernel "
                    f"is compiled for); got width {seed_thr.shape[1]}")
            use_ext_counts = seed_thr is not None and seed_thr.shape[1] == 8
            ext_rungs = seed_thr is not None and seed_thr.shape[1] == 3
            use_ext_cand = cand_vals is not None
            enable_block_skip = block_max is not None
            emit_xstate = xstate is not None
            if use_ext_cand:
                assert use_ext_counts, (
                    "candidate list requires the packed seed row "
                    "([rows, 8]: lines + counts)")
                assert (cand_idx is not None and cand_ctl is not None
                        and cand_vals.shape == cand_idx.shape
                        and cand_ctl.shape == (num_rows, 4)), (
                            "list tier needs cand_vals/cand_idx same shape "
                            "+ cand_ctl [rows, 4]")
            if (use_ext_counts or ext_rungs or use_ext_cand
                    or enable_block_skip):
                assert not lb_mode and order_row is None, (
                    "ext tiers are single-CTA/sort-path only")
                assert logits.dtype == torch.float32 and next_n == 1, (
                    "ext tiers are compiled for fp32 logits and next_n==1; "
                    f"got dtype={logits.dtype} next_n={next_n}")
            if num_threads is not None:
                tuning = dict(tuning, num_threads_per_block=num_threads)
            elif use_ext_cand and top_k <= 512:
                # small-K list rule: hit rows do O(list) work
                tuning = dict(tuning, num_threads_per_block=512)
            key = cls._compile(
                cute_dtype,
                top_k,
                next_n,
                compress_ratio=compress_ratio,
                return_output_values=return_output_values,
                cluster_size=cluster_size,
                seqlen_sorted=seqlen_sorted,
                enable_block_skip=enable_block_skip,
                use_ext_counts=use_ext_counts,
                emit_xstate=emit_xstate,
                use_ext_cand=use_ext_cand,
                ext_rungs=ext_rungs,
                cand_cap=(cand_vals.shape[1] if use_ext_cand else 5120),
                accept_cap=accept_cap,
                kc_override=kc_override,
                **tuning,
            )
            cls.kernel_cache[key](logits, pre_idx, seq_lens, None,
                                  output_indices, order_row, block_max,
                                  seed_thr, None, xstate, cand_vals, cand_idx,
                                  cand_ctl)

    # TODO(dsa.py): wire ``order_row = argsort(seq_lens, descending=True)``
    # (device-side, graph-safe) into the LJF row-reorder branch when
    # ``num_rows >= 2 * num_sms``. Physical meaning: wave-2 must fit a
    # full SM-row's worth of CTAs so the sort has long-vs-short rows to
    # swap. Below that threshold the win is noise / can regress a few
    # percent (B200 N∈{8K,16K,32K} sweep 2026-06-23).
    # xstate is written by the kernel but stays out of mutates_args
    # (optional-mutate None-default IndexError; see the fp4 op note)
    @torch.library.custom_op("trtllm::cute_dsl_gvr_topk_decode",
                             mutates_args=("output_indices", ),
                             device_types="cuda")
    def cute_dsl_gvr_topk_decode(
        logits: torch.Tensor,
        pre_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        cluster_size: Optional[int] = None,
        order_row: Optional[torch.Tensor] = None,
        counters: Optional[torch.Tensor] = None,
        max_batch_size: Optional[int] = None,
        seed_thr: Optional[torch.Tensor] = None,
        xstate: Optional[torch.Tensor] = None,
        cand_vals: Optional[torch.Tensor] = None,
        cand_idx: Optional[torch.Tensor] = None,
        cand_ctl: Optional[torch.Tensor] = None,
        block_max: Optional[torch.Tensor] = None,
        num_threads: Optional[int] = None,
        accept_cap: Optional[int] = None,
        kc_override: Optional[int] = None,
    ) -> None:
        """CuTe DSL GVR (Guess-Verify-Refine) Top-K decode for Blackwell.

        Writes per-row top-K indices into ``output_indices`` (indices
        only, mirroring CUDA ``indexer_topk_decode``).

        Args:
            logits: ``[num_rows, max_seq_len]`` fp32 / bf16 / fp16.
            pre_idx: ``[num_rows // next_n, top_k]`` int32.
                ``pre_idx[..., 0]`` must be the argmax index.
            seq_lens: ``[num_rows // next_n]`` int32, request-level.
            output_indices: ``[num_rows, top_k]`` int32.
            top_k: K ∈ {512, 1024, 2048} — compile-time specialized.
            next_n: Temporal stride (V3.2
                ``preIdxOffset = (row % next_n) + 1``).
            compress_ratio: 1 = DSv3.2, 4 = DSv4.
            max_seq_len: Peak N at replay; pass under CUDA Graph capture
                so the heuristic picks the large-N kernel.
            cluster_size: 1 = single-CTA; 2/4/8 = N CTAs cooperate via
                DSMEM. ``None`` → auto-pick from (N, BS) (single-CTA /
                sort path) or 4 (LB path).
            order_row: Request-level dispatch order (int32, CUDA).
                Without ``counters``: descending argsort of ``seq_lens``
                (shape == seq_lens.shape). With ``counters``: LB
                long-first partition (shape == max_batch_size) from
                :func:`cute_dsl_gvr_topk_lb_prepare`. ``None`` skips
                the indirection.
            counters: LB ``[n_long, n_short]`` from
                :func:`cute_dsl_gvr_topk_lb_prepare`. Selects the LB
                path and requires ``order_row`` + ``max_batch_size``.
            max_batch_size: Required with ``counters``; ignored
                otherwise. Power of 2 in ``[64, 1024]``, must match
                the value passed to LB prepare.

        Of the optional hint tensors the kernel WRITES ``xstate`` (the
        closed-loop publish); it cannot be declared in ``mutates_args``:
        torch.library raises IndexError when a declared-mutable Optional
        arg is None at call time (re-verified on the pinned torch), and
        most calls pass no hints. Under torch.compile/functionalization
        the undeclared write is invisible, so the hint path is eager /
        CUDA-graph only. The ``use_gvr_emission`` sparse-attention config
        field gates the emission-assisted wiring that feeds these tensors
        (opt-in; temporal-hint path only).
        """
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL GVR Top-K Decode only supports SM 100 family.")
        if logits.shape[0] % next_n != 0:
            raise ValueError(
                f"logits.shape[0] (={logits.shape[0]}) must be divisible by "
                f"next_n (={next_n}); kernel derives batch_size as "
                f"logits.shape[0] / next_n.")
        # Log once per (dtype, shape) so each new shape gets a line.
        _log_sig = (
            f"{logits.dtype}|{tuple(logits.shape)}|"
            f"k={top_k}|nn={next_n}|cr={compress_ratio}|msl={max_seq_len}")
        logger.info_once(
            f"cute_dsl_gvr_topk_decode inputs: "
            f"logits dtype={logits.dtype} shape={tuple(logits.shape)} stride={logits.stride()}; "
            f"pre_idx dtype={pre_idx.dtype} shape={tuple(pre_idx.shape)}; "
            f"seq_lens dtype={seq_lens.dtype} shape={tuple(seq_lens.shape)}; "
            f"output_indices dtype={output_indices.dtype} shape={tuple(output_indices.shape)}; "
            f"top_k={top_k} next_n={next_n} compress_ratio={compress_ratio} "
            f"max_seq_len={max_seq_len}",
            key=f"cute_dsl_gvr_topk_decode_inputs|{_log_sig}",
        )
        CuteDSLGvrTopKDecodeRunner.forward(
            logits=logits,
            pre_idx=pre_idx,
            seq_lens=seq_lens,
            output_indices=output_indices,
            top_k=top_k,
            next_n=next_n,
            compress_ratio=compress_ratio,
            max_seq_len=max_seq_len,
            cluster_size=cluster_size,
            order_row=order_row,
            counters=counters,
            max_batch_size=max_batch_size,
            seed_thr=seed_thr,
            xstate=xstate,
            cand_vals=cand_vals,
            cand_idx=cand_idx,
            cand_ctl=cand_ctl,
            block_max=block_max,
            num_threads=num_threads,
            accept_cap=accept_cap,
            kc_override=kc_override,
        )

    @torch.library.register_fake("trtllm::cute_dsl_gvr_topk_decode")
    def _(
        logits: torch.Tensor,
        pre_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        cluster_size: Optional[int] = None,
        order_row: Optional[torch.Tensor] = None,
        counters: Optional[torch.Tensor] = None,
        max_batch_size: Optional[int] = None,
        seed_thr: Optional[torch.Tensor] = None,
        xstate: Optional[torch.Tensor] = None,
        cand_vals: Optional[torch.Tensor] = None,
        cand_idx: Optional[torch.Tensor] = None,
        cand_ctl: Optional[torch.Tensor] = None,
        block_max: Optional[torch.Tensor] = None,
        num_threads: Optional[int] = None,
        accept_cap: Optional[int] = None,
        kc_override: Optional[int] = None,
    ) -> None:
        return None

    # ---- GVR Top-K Load-Balance (hybrid multi-CTA + single-CTA) ----
    # Two ops:
    #   1. cute_dsl_gvr_topk_lb_prepare (once per decode step) — writes
    #      (order_row, counters) by classifying seq_lens into long/short.
    #   2. cute_dsl_gvr_topk_decode with counters set (once per layer) —
    #      long rows ride a cluster (cs=2/4) via DSMEM; short rows go
    #      single-CTA. Both branches share the grid for graph capture.
    # (order_row, counters) are layer-invariant within a decode step.
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_load_balance import \
        GvrTopKLBKernel as _GvrTopKLBKernel
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_load_balance import \
        GvrTopKLBPrepareKernel as _GvrTopKLBPrepareKernel

    # No Runner class for prepare: no tuning knobs, no cluster dispatch.
    @functools.cache
    def _compile_lb_prepare(
        num_threads: int,
        batch_size: int,
        long_threshold: int,
        compress_ratio: int,
    ):
        """JIT-compile the LB prepare kernel.

        ``num_threads`` = block size = max_batch_size. ``batch_size``
        must equal runtime ``seq_lens.shape[0]`` (TVM-FFI marshalling).
        ``compress_ratio`` puts the classifier in scan-length space.
        """
        prep = _GvrTopKLBPrepareKernel(
            long_threshold=long_threshold,
            compress_ratio=compress_ratio,
            num_threads=num_threads,
        )
        fake_seq = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                         (batch_size, ),
                                                         stride_order=(0, ))
        fake_order = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                           (num_threads, ),
                                                           stride_order=(0, ))
        fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2, ),
                                                         stride_order=(0, ))
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            prep,
            fake_seq,
            fake_order,
            fake_ctr,
            cutlass.Int32(0),
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

    @torch.library.custom_op(
        "trtllm::cute_dsl_gvr_topk_lb_prepare",
        mutates_args=("order_row", "counters"),
        device_types="cuda",
    )
    def cute_dsl_gvr_topk_lb_prepare(
        seq_lens: torch.Tensor,
        order_row: torch.Tensor,
        counters: torch.Tensor,
        max_batch_size: int,
        long_threshold: int = 64 * 1024,
        compress_ratio: int = 1,
    ) -> None:
        """LB partition prepare — run once per decode step; outputs
        are layer-invariant and feed every per-layer decode call.

        Writes:
          ``order_row[max_batch_size]`` int32 — long group at
              ``[0, n_long)``, short group at ``[n_long, n_long+n_short)``,
              tail untouched (caller should pre-fill with -1).
          ``counters[2]`` int32 — ``[n_long_req, n_short_req]``.

        Args:
            seq_lens: ``[batch_size]`` int32, UNCOMPRESSED tokens
                (classifier divides by ``compress_ratio`` internally).
            order_row: caller-allocated ``[max_batch_size]`` int32.
                Fixed shape so CUDA Graph capture sees a single grid.
            counters: caller-allocated ``[2]`` int32.
            long_threshold: scan-length-space threshold (default 64K =
                B200 cs=4 break-even ≈ 3.2us / row).
            max_batch_size: power of 2 in [64, 1024]
                (block_prefix_sum_kernel constraint).
            compress_ratio: 1 = DSv3.2, 4 = DSv4.
        """
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL GVR Top-K LB prepare only supports SM 100 family.")
        # block_prefix_sum needs num_warps > 1 and pow2 →
        # max_batch_size ∈ {64, 128, 256, 512, 1024}.
        if not (64 <= max_batch_size <= 1024) or (max_batch_size &
                                                  (max_batch_size - 1)) != 0:
            raise ValueError(
                f"max_batch_size must be a power of 2 in [64, 1024] "
                f"(block_prefix_sum_kernel constraint); got {max_batch_size}")
        batch_size = seq_lens.shape[0]
        if batch_size > max_batch_size:
            # Block is sized to max_batch_size (1 thread / request);
            # requests with idx >= max_batch_size have no thread.
            raise ValueError(
                f"batch_size ({batch_size}) must be <= max_batch_size "
                f"({max_batch_size}).")
        assert (order_row.dtype == torch.int32 and order_row.is_cuda
                and order_row.shape == (max_batch_size, )), (
                    f"order_row must be int32, CUDA, shape "
                    f"({max_batch_size},); got dtype={order_row.dtype} "
                    f"shape={tuple(order_row.shape)}")
        assert (counters.dtype == torch.int32 and counters.is_cuda
                and counters.shape == (2, )), (
                    f"counters must be int32, CUDA, shape (2,); got "
                    f"dtype={counters.dtype} shape={tuple(counters.shape)}")
        compiled = _compile_lb_prepare(max_batch_size, batch_size,
                                       long_threshold, compress_ratio)
        compiled(seq_lens, order_row, counters, cutlass.Int32(batch_size))

    @torch.library.register_fake("trtllm::cute_dsl_gvr_topk_lb_prepare")
    def _(
        seq_lens: torch.Tensor,
        order_row: torch.Tensor,
        counters: torch.Tensor,
        max_batch_size: int,
        long_threshold: int = 64 * 1024,
        compress_ratio: int = 1,
    ) -> None:
        return None

    # LB decode lives inside CuteDSLGvrTopKDecodeRunner — see
    # ``_compile_lb`` and the ``counters is not None`` branch of
    # ``forward`` (shares ``_pick_tuning`` with the single-CTA path).

    # ------------------------------------------------------------------ #
    #  CuTE DSL FP8 Paged MQA Logits (Blackwell SM100)                   #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.paged_mqa_logits import (
        FP4MQALogitsKernel, FP8MQALogitsKernel)

    class CuteDSLPagedMQALogitsRunner:
        """Runner for CuTe DSL FP8 Paged MQA Logits kernel (Blackwell SM100).

        Caches compiled kernels keyed by static params
        (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n, num_sms).
        """

        kernel_cache = dict()

        @classmethod
        def _compile(cls, compute_block_kv, phys_block_kv, num_heads, head_dim,
                     next_n, num_sms, num_epi_subtiles, epi_dtype, acc_dtype,
                     output_dtype):
            """Compile kernel using fake tensors + TVM FFI."""
            key = (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n,
                   num_sms, num_epi_subtiles, epi_dtype, acc_dtype,
                   output_dtype)
            if key in cls.kernel_cache:
                return

            to_cutlass = _TORCH_TO_CUTLASS_DTYPE
            N = next_n * num_heads
            block_bytes = phys_block_kv * (head_dim + 4)

            sym_num_phys_blocks = cute.sym_int()
            sym_B = cute.sym_int()
            max_ctx = cute.sym_int()
            max_blocks_per_seq = cute.sym_int()
            num_ctas = cute.sym_int()

            # KV may come from the indexer K-cache pool view, which is
            # strided in dim 0 (pool layout interleaves layers:
            # [num_blocks, num_layers, kvFactor, blockSize]). Declare outer
            # stride as sym so the actual per-block stride is read at
            # runtime; innermost stride is fixed to 1 (byte-contig within a
            # logical block view).
            kv_fake = cute.runtime.make_fake_tensor(
                cutlass.Uint8, (sym_num_phys_blocks, block_bytes),
                stride=(cute.sym_int64(), 1))

            q_fake = cute.runtime.make_fake_compact_tensor(cutlass.Uint8,
                                                           (N, head_dim, sym_B),
                                                           stride_order=(1, 0,
                                                                         2))

            w_dtype = (cutlass.Float16
                       if epi_dtype == torch.float16 else to_cutlass[epi_dtype])
            w_fake = cute.runtime.make_fake_compact_tensor(w_dtype, (N, sym_B),
                                                           stride_order=(0, 1))

            logits_fake = cute.runtime.make_fake_tensor(
                to_cutlass[output_dtype], (cute.sym_int(), max_ctx),
                stride=(cute.sym_int64(), 1))

            bt_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_B, max_blocks_per_seq), stride_order=(1, 0))

            cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                            (sym_B, ),
                                                            stride_order=(0, ))

            sm_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                            (num_ctas, 2),
                                                            stride_order=(1, 0))

            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            kernel = FP8MQALogitsKernel(
                block_kv=compute_block_kv,
                phys_block_kv=phys_block_kv,
                num_heads=num_heads,
                head_dim=head_dim,
                next_n=next_n,
                num_sms=num_sms,
                num_epi_subtiles=num_epi_subtiles,
                epi_dtype=to_cutlass[epi_dtype],
                acc_dtype=to_cutlass[acc_dtype],
                output_dtype=to_cutlass[output_dtype],
            )

            compiled = cute.compile(
                kernel,
                kv_fake,
                q_fake,
                w_fake,
                logits_fake,
                bt_fake,
                cl_fake,
                sm_fake,
                cutlass.Int32(1),
                cutlass.Int32(1),
                fake_stream,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled
            logger.debug(f"[compile cute_dsl fp8_paged_mqa_logits] {key}"
                         f" kv_stages={kernel.num_kv_stages}"
                         f" umma_stages={kernel.num_umma_stages}")

        @classmethod
        def forward(
            cls,
            q: torch.Tensor,
            kv_fused: torch.Tensor,
            weights: torch.Tensor,
            context_lens: torch.Tensor,
            block_table: torch.Tensor,
            schedule_meta: torch.Tensor,
            max_context_len: int,
            num_epi_subtiles: int = 1,
            epi_dtype: torch.dtype = torch.float32,
            acc_dtype: torch.dtype = torch.float32,
            output_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            """Execute FP8 paged MQA logits kernel.

            Args:
                q: [B, next_n, H, D] FP8
                kv_fused: [num_blocks, phys_block_kv, 1, D+4] uint8
                weights: [B*next_n, H] float32
                context_lens: [B] int32
                block_table: [B, max_blocks] int32
                schedule_meta: [num_sms+1, 2] int32
                max_context_len: int
                num_epi_subtiles: epilogue sub-tile count (1, 2, or 4)
                epi_dtype: epilogue compute dtype
                acc_dtype: MMA accumulator dtype
                output_dtype: output logits dtype
            Returns:
                logits: [B*next_n, max_context_len] output_dtype
            """
            B, next_n, H, D = q.shape
            N = next_n * H
            phys_block_kv = kv_fused.shape[1]
            compute_block_kv = 128
            num_phys_blocks = kv_fused.shape[0]
            num_sms = _get_num_sms()

            # Reshape Q: [B, next_n, H, D] -> [B, N, D] -> [N, D, B]
            q_3d = q.reshape(B, N, D).permute(1, 2, 0)

            # Reshape weights: [B*next_n, H] -> [B, N] -> [N, B]
            if epi_dtype == torch.float16:
                # TODO: move type conversion to weight loading
                w_2d = weights.reshape(B, N).half().t()
            else:
                w_2d = weights.reshape(B, N).t()

            # Flatten fused KV to [num_phys_blocks, block_bytes]
            kv_flat = kv_fused.reshape(num_phys_blocks, -1)

            # Allocate output with alignment padding
            SPLIT_KV = compute_block_kv * 2  # NUM_MATH_WG = 2
            aligned_max_ctx = (
                (max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
            # Use a persistent arena buffer instead of a per-forward torch.empty
            # so the output address stays stable across CUDA-graph replays.
            _reserve = torch.cuda.is_current_stream_capturing()
            logits = get_memory_buffers().get_buffer(
                [B * next_n, aligned_max_ctx],
                output_dtype,
                buffer_name="cute_dsl_mqa_logits",
                reserve_buffer=_reserve,
            )
            logits = logits[:, :max_context_len]

            # Compile if needed (fake tensors, no real data required)
            key = (compute_block_kv, phys_block_kv, H, D, next_n, num_sms,
                   num_epi_subtiles, epi_dtype, acc_dtype, output_dtype)
            if key not in cls.kernel_cache:
                cls._compile(compute_block_kv, phys_block_kv, H, D, next_n,
                             num_sms, num_epi_subtiles, epi_dtype, acc_dtype,
                             output_dtype)
            compiled = cls.kernel_cache[key]

            # FP8 q needs uint8 view to match compile-time dtype
            q_for_ffi = (q_3d.view(torch.uint8) if q_3d.dtype
                         in (torch.float8_e4m3fn, torch.float8_e5m2) else q_3d)

            # TVM FFI: pass raw tensors, no dlpack/stream needed
            compiled(kv_flat, q_for_ffi, w_2d, logits, block_table,
                     context_lens, schedule_meta, num_phys_blocks, B)
            return logits

    @torch.library.custom_op("trtllm::cute_dsl_fp8_paged_mqa_logits",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_fp8_paged_mqa_logits(
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL FP8 Paged MQA Logits only supports SM 100 family.")
        # Caller (dsa.py) prepares all tensors with metadata-guaranteed
        # dtype/shape; skip per-call validation to keep decode-hot-path
        # latency low. Log inputs once for debugging.
        logger.info_once(
            f"cute_dsl_fp8_paged_mqa_logits inputs: "
            f"q dtype={q.dtype} shape={tuple(q.shape)} stride={q.stride()}; "
            f"kv_fused dtype={kv_fused.dtype} shape={tuple(kv_fused.shape)} stride={kv_fused.stride()}; "
            f"weights dtype={weights.dtype} shape={tuple(weights.shape)} stride={weights.stride()}; "
            f"context_lens dtype={context_lens.dtype} shape={tuple(context_lens.shape)}; "
            f"block_table dtype={block_table.dtype} shape={tuple(block_table.shape)} stride={block_table.stride()}; "
            f"schedule_meta dtype={schedule_meta.dtype} shape={tuple(schedule_meta.shape)}; "
            f"max_context_len={max_context_len} num_epi_subtiles={num_epi_subtiles} "
            f"epi_dtype={epi_dtype} acc_dtype={acc_dtype} output_dtype={output_dtype}",
            key="cute_dsl_fp8_paged_mqa_logits_inputs",
        )
        return CuteDSLPagedMQALogitsRunner.forward(
            q,
            kv_fused,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            num_epi_subtiles=num_epi_subtiles,
            epi_dtype=epi_dtype,
            acc_dtype=acc_dtype,
            output_dtype=output_dtype)

    @torch.library.register_fake("trtllm::cute_dsl_fp8_paged_mqa_logits")
    def _(
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        B = q.shape[0]
        next_n = q.shape[1]
        return torch.empty(B * next_n,
                           max_context_len,
                           dtype=output_dtype,
                           device=q.device)

    # ------------------------------------------------------------------ #
    #  CuTe DSL MiniMax-M3 index decode scoring (Blackwell SM100)         #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.cute_ptx_utils import \
        TORCH_TO_CUTE_DTYPE as _M3_TORCH_TO_CUTE_DTYPE
    from ..cute_dsl_kernels.blackwell.minimax_m3_index_decode_score import \
        IndexDecodeScoreKernel

    class CuteDSLMiniMaxM3IndexDecodeScoreRunner:
        """Runner for the MiniMax-M3 indexer decode block-scoring kernel.

        Caches compiled kernels keyed on the static params
        (dtype, num_heads, max_decode_query_len, head_dim); batch, query-token
        count, page count and block-table width stay symbolic, so one compile
        covers every decode step of a given model shape.
        """

        kernel_cache = dict()
        # One CTA per (request, split); each walks blocks split, split + 256,
        # ... so a request longer than 256 pages just loops. Matches upstream.
        SPLIT_K = 256
        # The only geometry the kernel has been validated on.
        SUPPORTED_HEAD_DIM = 128
        SUPPORTED_PAGE_SIZE = IndexDecodeScoreKernel.BLOCK_K
        # BLOCK_Q must fit one warp's worth of epilogue lanes.
        MAX_BLOCK_Q = 32

        @classmethod
        def is_supported(
            cls,
            *,
            q_dtype: torch.dtype,
            num_heads: int,
            head_dim: int,
            page_size: int,
            max_decode_query_len: int,
        ) -> bool:
            """Whether this kernel can serve the given decode geometry.

            Callers use this to pick between the CuTe DSL scorer and the
            fallback rather than catching an exception on the hot path.
            """
            return (is_sm_100f() and q_dtype in _M3_TORCH_TO_CUTE_DTYPE
                    and head_dim == cls.SUPPORTED_HEAD_DIM
                    and page_size == cls.SUPPORTED_PAGE_SIZE
                    and num_heads * max_decode_query_len <= cls.MAX_BLOCK_Q
                    and max_decode_query_len >= 1)

        @classmethod
        def _compile(cls, q_dtype: torch.dtype, num_heads: int,
                     max_decode_query_len: int, head_dim: int):
            key = (q_dtype, num_heads, max_decode_query_len, head_dim)
            if key in cls.kernel_cache:
                return

            cute_dtype = _M3_TORCH_TO_CUTE_DTYPE[q_dtype]
            page_size = cls.SUPPORTED_PAGE_SIZE

            sym_total_tokens = cute.sym_int()
            sym_batch = cute.sym_int()

            # 16-element divisibility on every non-innermost stride is what the
            # TMA descriptors for Q and K assume.
            def _sym_stride():
                return cute.sym_int64(divisibility=16)

            q_fake = cute.runtime.make_fake_tensor(
                cute_dtype, (sym_total_tokens, num_heads, head_dim),
                stride=(_sym_stride(), _sym_stride(), 1))

            # The index-K pool may be coalesced with the main K/V cache, in
            # which case the per-page stride exceeds page_size * head_dim, so
            # dim 0 is read at runtime.
            k_fake = cute.runtime.make_fake_tensor(
                cute_dtype, (cute.sym_int(), page_size, head_dim),
                stride=(_sym_stride(), _sym_stride(), 1))

            bt_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_batch, cute.sym_int()), stride_order=(1, 0))

            # Every stride is symbolic because production passes a transposed
            # view of the [heads, blocks, tokens] selector buffer, whose
            # innermost logical stride is the token count rather than 1.
            score_fake = cute.runtime.make_fake_tensor(
                cutlass.Float32, (num_heads, sym_total_tokens, cute.sym_int()),
                stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()))

            sl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                            (sym_batch, ),
                                                            stride_order=(0, ))

            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            kernel = IndexDecodeScoreKernel(
                cute_dtype,
                num_heads,
                max_decode_query_len,
                cls.SPLIT_K,
                head_dim,
            )
            cls.kernel_cache[key] = cute.compile(
                kernel,
                q_fake,
                k_fake,
                bt_fake,
                score_fake,
                sl_fake,
                fake_stream,
                options="--enable-tvm-ffi",
            )
            logger.debug(
                f"[compile cute_dsl minimax_m3_index_decode_score] {key}")

        @classmethod
        def forward(
            cls,
            idx_q: torch.Tensor,
            index_k_cache: torch.Tensor,
            block_table: torch.Tensor,
            seq_lens: torch.Tensor,
            score: torch.Tensor,
            max_decode_query_len: int,
        ) -> None:
            """Score one decode step, compiling the kernel on first use.

            The compile allocates, so it must not land inside a CUDA graph
            capture. It does not: CUDAGraphRunner.capture runs eager warmup
            forwards first, and those cover every geometry the graphs it then
            captures will replay.
            """
            _, num_heads, head_dim = idx_q.shape
            key = (idx_q.dtype, num_heads, max_decode_query_len, head_dim)
            if key not in cls.kernel_cache:
                cls._compile(idx_q.dtype, num_heads, max_decode_query_len,
                             head_dim)
            cls.kernel_cache[key](idx_q, index_k_cache, block_table, score,
                                  seq_lens)

    @torch.library.custom_op("trtllm::cute_dsl_minimax_m3_index_decode_score",
                             mutates_args=("score", ),
                             device_types="cuda")
    def cute_dsl_minimax_m3_index_decode_score(
        idx_q: torch.Tensor,
        index_k_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        score: torch.Tensor,
        max_decode_query_len: int,
    ) -> None:
        """Write per-block max index scores for one decode step, in place.

        Args:
            idx_q: [total_q, num_index_heads, head_dim], BF16 or FP8 E4M3.
                total_q must be batch * decode_query_len with a uniform
                decode_query_len, which the kernel infers.
            index_k_cache: [num_pages, page_size, head_dim], same dtype as
                idx_q. May be a strided view of a coalesced pool.
            block_table: [batch, max_blocks_per_seq] int32 page table.
            seq_lens: [batch] int32 attended KV length per request.
            score: [num_index_heads, total_q, max_blocks] float32, mutated in
                place. Only blocks below ceil(seq_len / page_size) are written,
                which is exactly the range the block selector reads. Arbitrary
                strides are accepted so a transposed selector buffer can be
                passed without a copy.
            max_decode_query_len: compile-time bound on decode_query_len;
                num_index_heads * max_decode_query_len must not exceed 32.
        """
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL MiniMax-M3 index decode score only supports SM 100 "
                f"family.")
        logger.info_once(
            f"cute_dsl_minimax_m3_index_decode_score inputs: "
            f"idx_q dtype={idx_q.dtype} shape={tuple(idx_q.shape)} stride={idx_q.stride()}; "
            f"index_k_cache dtype={index_k_cache.dtype} shape={tuple(index_k_cache.shape)} "
            f"stride={index_k_cache.stride()}; "
            f"block_table shape={tuple(block_table.shape)} stride={block_table.stride()}; "
            f"seq_lens shape={tuple(seq_lens.shape)}; "
            f"score dtype={score.dtype} shape={tuple(score.shape)} stride={score.stride()}; "
            f"max_decode_query_len={max_decode_query_len}",
            key="cute_dsl_minimax_m3_index_decode_score_inputs",
        )
        CuteDSLMiniMaxM3IndexDecodeScoreRunner.forward(
            idx_q,
            index_k_cache,
            block_table,
            seq_lens,
            score,
            max_decode_query_len,
        )

    @torch.library.register_fake(
        "trtllm::cute_dsl_minimax_m3_index_decode_score")
    def _(
        idx_q: torch.Tensor,
        index_k_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        score: torch.Tensor,
        max_decode_query_len: int,
    ) -> None:
        return None

    # ======================================================================
    # BF16 Dense Persistent BMM (CuTe DSL) for Blackwell
    # ======================================================================

    def _bf16_preferred_cluster_has_launchable_grid(
        m: int,
        n: int,
        batch_size: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Return whether mixed preferred/fallback launch has any preferred cluster.

        The preferred-cluster kernel derives preferred_grid.z from the fallback
        grid CTA count divided by the preferred cluster size.  If the autotuner
        profiles a small M bucket and that quotient is zero, CUDA rejects the
        launch with cudaErrorInvalidValue.

        The per-CTA M-tile is mma_tiler_mn[0] for 1-CTA MMA but mma_tiler_mn[0]//2
        for 2-CTA MMA (the grid is built from the per-CTA tile), so the CTA-tile
        count uses the halved tile when use_2cta_instrs -- otherwise the fallback
        CTA count is undercounted and valid preferred-cluster tactics are pruned.
        """
        cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        ctas_m = ceil_div(m, cta_tile_m)
        ctas_n = ceil_div(n, mma_tiler_mn[1])
        fallback_ctas_m = pad_up(ctas_m, fallback_cluster_shape_mn[0])
        fallback_ctas_n = pad_up(ctas_n, fallback_cluster_shape_mn[1])
        fallback_ctas = fallback_ctas_m * fallback_ctas_n * batch_size
        preferred_cluster_ctas = (preferred_cluster_shape_mn[0] *
                                  preferred_cluster_shape_mn[1])
        return fallback_ctas >= preferred_cluster_ctas

    def _bf16_cluster_m_fits(
        m: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Whether the M dimension provides enough CTA-tiles for the M-cluster.

        An M-cluster wider than the available M CTA-tiles leaves phantom CTAs
        whose cluster-multicast peers are never launched, which probabilistically
        triggers an illegal memory access / hang during autotuner profiling
        (observed on SM107 with the M=1 decode MLA absorb BMM, cluster_m=4). The
        per-CTA M-tile is mma_tiler_mn[0] for 1-CTA MMA but mma_tiler_mn[0]//2 for
        2-CTA MMA (the kernel builds the grid from the per-CTA tile), so the
        CTA-tile count must use the halved tile when use_2cta_instrs -- otherwise
        valid 2-CTA / preferred-cluster tactics (e.g. m=128, tile_m=128, 2cta,
        cluster_m=2 -> 2 real M-CTAs) get pruned. N over-padding is handled by the
        kernel, so only the M axis is gated; cluster_m=4 / preferred (4,2) stay
        available for large-M shapes.
        """
        cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        return ceil_div(m, cta_tile_m) >= cluster_shape_mn[0]

    class CuteDSLBf16BlackwellBmmRunner(TunableRunner):
        kernel_class = PersistentDenseGemmKernel
        kernel_cache = dict()

        tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
            0, 1, get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2), ), )

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__()
            self.use_tvm_ffi = use_tvm_ffi

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[int]:

            if not is_sm_100f():
                logger.debug(
                    f"CuteDSL: SM version {get_sm_version()} is not supported. "
                    f"CuteDSL BF16 BMM only supports SM 100 family. Skipping all tactics."
                )
                return []
            # [b, m, k]
            batch_size, m, k = inputs[0].shape[0], inputs[0].shape[1], inputs[
                0].shape[2]
            # [b, n, k]
            n = inputs[1].shape[1]
            # m,k
            a_major = "k"
            # n, k
            b_major = "k"
            # m, n
            c_major = "n"

            use_2cta_instrs_candi = [False, True]
            mma_tiler_mn_candi = [(64, 128), (128, 128), (256, 128)]
            cluster_shape_mn_candi = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            return [
                (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
                for use_2cta_instrs in use_2cta_instrs_candi
                for mma_tiler_mn in mma_tiler_mn_candi
                for cluster_shape_mn in cluster_shape_mn_candi
                if self.__class__.kernel_class.can_implement(
                    cutlass.BFloat16,  # ab_dtype
                    cutlass.Float32,  # acc_dtype
                    cutlass.BFloat16,  # c_dtype
                    use_2cta_instrs,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    m,
                    n,
                    k,
                    batch_size,
                    a_major,
                    b_major,
                    c_major,
                )
            ]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
        ) -> None:
            """
            Performs bf16 dense persistent batched gemm using CuTe DSL.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (batch_size, m, k), dtype: bf16.
                    inputs[1]: Weight tensor of shape (batch_size, n, k), dtype: bf16.
                    inputs[2]: Output tensor of shape (batch_size, m, n), dtype: bf16.
                tactic: Tiling and cluster strategy, typically a tuple
                    (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn).
            """
            if isinstance(tactic, tuple):
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
            else:
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = [
                    False,
                    (128, 128),
                    (1, 1),
                ]

            a_tensor, b_tensor, c_tensor = inputs

            # Permute C from [B, M, N] to [M, N, B] for CuTe layout.
            # from_dlpack captures the actual strides, so non-contiguous
            # views (e.g. from .transpose(0,1)) are handled natively by
            # TMA without an extra copy.
            c_tmp = c_tensor.permute(1, 2, 0)

            batch_size = a_tensor.shape[0]
            m = a_tensor.shape[1]
            k = a_tensor.shape[2]
            n = b_tensor.shape[1]

            # Compute A strides so the kernel can handle non-contiguous
            # views (e.g. [M,B,K].transpose(0,1) → [B,M,K] with
            # non-standard strides) without a .contiguous() copy.
            # CuTe tensor is (M, K, B) so strides map as:
            #   M stride  = a_tensor.stride(1)
            #   K stride  = 1  (always innermost)
            #   B stride  = a_tensor.stride(0)
            a_stride_m = a_tensor.stride(1)
            a_stride_batch = a_tensor.stride(0)
            b_stride_n = b_tensor.stride(1)
            b_stride_batch = b_tensor.stride(0)

            if not self.use_tvm_ffi:
                a_ptr = make_ptr(
                    cutlass.BFloat16,
                    a_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_ptr = make_ptr(
                    cutlass.BFloat16,
                    b_tensor.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                c_cute_tensor = cute.runtime.from_dlpack(
                    c_tmp).mark_layout_dynamic(leading_dim=1)

                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            cache_key = (
                use_2cta_instrs,
                mma_tiler_mn,
                cluster_shape_mn,
                self.use_tvm_ffi,
            )
            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = make_ptr(
                        cutlass.BFloat16,
                        a_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_ptr = make_ptr(
                        cutlass.BFloat16,
                        b_tensor.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    c_cute_tensor = cute.runtime.from_dlpack(
                        c_tmp).mark_layout_dynamic(leading_dim=1)
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    cutlass.Float32,  # acc_dtype
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper_strided,
                    m,
                    n,
                    k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    c_cute_tensor,
                    a_stride_m,
                    a_stride_batch,
                    b_stride_n,
                    b_stride_batch,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # launch gemm kernel
            if self.use_tvm_ffi:
                compiled_gemm(
                    m,
                    n,
                    k,
                    batch_size,
                    a_tensor.data_ptr(),
                    b_tensor.data_ptr(),
                    c_tmp,
                    a_stride_m,
                    a_stride_batch,
                    b_stride_n,
                    b_stride_batch,
                )
            else:
                compiled_gemm(
                    m,
                    n,
                    k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    c_cute_tensor,
                    a_stride_m,
                    a_stride_batch,
                    b_stride_n,
                    b_stride_batch,
                    stream=stream,
                )

    # a/b: bf16, output: bf16
    @torch.library.custom_op("trtllm::cute_dsl_bf16_bmm_blackwell",
                             mutates_args=("output", ),
                             device_types="cuda")
    def cute_dsl_bf16_bmm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> None:
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL BF16 BMM only supports SM 100 family.")

        tuner = AutoTuner.get()

        runner = CuteDSLBf16BlackwellBmmRunner(use_tvm_ffi=use_tvm_ffi)

        inputs = [input, weight, output]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_bf16_bmm_blackwell::gemm",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_bf16_bmm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        output: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> None:
        batch_size, m, k = mat_a.shape[0], mat_a.shape[1], mat_a.shape[2]
        n = mat_b.shape[1]
        assert output.dtype == torch.bfloat16, "CuTe DSL bf16 bmm output dtype must be bf16"
        assert output.shape == (
            batch_size, m, n), "CuTe DSL bf16 bmm output shape is incorrect"

    # ======================================================================
    # BF16 Dense Persistent GEMM (CuTe DSL) for Blackwell - Linear layers
    # ======================================================================

    class CuteDSLBf16BlackwellGemmRunner(TunableRunner):
        """
        CuTe DSL BF16 GEMM runner for Linear layers.

        Unlike BMM which operates on [B, M, K] @ [B, N, K] -> [B, M, N],
        GEMM operates on [M, K] @ [N, K]^T -> [M, N] (standard Linear).

        We reuse PersistentDenseGemmKernel with batch_size=1.
        """
        kernel_class = PersistentDenseGemmKernel
        kernel_cache = dict()

        tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
            0, 0, get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2), ), )

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__()
            self.use_tvm_ffi = use_tvm_ffi

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[int]:

            if not is_sm_100f():
                logger.debug(
                    f"CuteDSL: SM version {get_sm_version()} is not supported. "
                    f"CuteDSL BF16 GEMM only supports SM 100 family. Skipping all tactics."
                )
                return []

            # input: [M, K], weight: [N, K], output: [M, N]
            m, k = inputs[0].shape[0], inputs[0].shape[1]
            n = inputs[1].shape[0]
            batch_size = 1

            # Detect output dtype from the output tensor (supports BF16 and FP32)
            c_dtype_cutlass = _TORCH_TO_CUTLASS_DTYPE[inputs[2].dtype]

            # Layouts: A is [M, K] K-major, B is [N, K] K-major
            a_major = "k"
            b_major = "k"
            c_major = "n"

            use_2cta_instrs_candi = [False, True]
            mma_tiler_mn_candi = [(64, 128), (128, 128), (256, 128)]
            cluster_shape_mn_candi = [
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 1),
                (4, 2),
                (4, 4),
            ]
            return [
                (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
                for use_2cta_instrs in use_2cta_instrs_candi
                for mma_tiler_mn in mma_tiler_mn_candi
                for cluster_shape_mn in cluster_shape_mn_candi
                if self.__class__.kernel_class.can_implement(
                    cutlass.BFloat16,  # ab_dtype
                    cutlass.Float32,  # acc_dtype
                    c_dtype_cutlass,  # c_dtype
                    use_2cta_instrs,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    m,
                    n,
                    k,
                    batch_size,
                    a_major,
                    b_major,
                    c_major,
                )
            ]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
        ) -> None:
            """
            Performs bf16 dense persistent GEMM using CuTe DSL.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (m, k), dtype: bf16.
                    inputs[1]: Weight tensor of shape (n, k), dtype: bf16.
                    inputs[2]: Output tensor of shape (m, n), dtype: bf16 or fp32.
                tactic: Tiling and cluster strategy, typically a tuple
                    (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn).
            """
            if isinstance(tactic, tuple):
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
            else:
                use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = [
                    False,
                    (128, 128),
                    (1, 1),
                ]

            a_tensor, b_tensor, c_tensor = inputs

            # Input: [M, K], Weight: [N, K], Output: [M, N]
            m, k = a_tensor.shape[0], a_tensor.shape[1]
            n = b_tensor.shape[0]
            batch_size = 1

            # Ensure inputs are contiguous
            a_tensor = a_tensor.contiguous()
            b_tensor = b_tensor.contiguous()

            # For output, use contiguous buffer if needed
            c_needs_copy = not c_tensor.is_contiguous()
            if c_needs_copy:
                c_buf = torch.empty_like(c_tensor)
            else:
                c_buf = c_tensor

            # Reshape to [1, M, K], [1, N, K], [1, M, N] for the batched kernel
            a_batched = a_tensor.unsqueeze(0)  # [1, M, K]
            b_batched = b_tensor.unsqueeze(0)  # [1, N, K]
            # c_buf is [M, N], permute to [M, N, 1] for cute layout
            c_tmp = c_buf.unsqueeze(-1)  # [M, N, 1]

            # Detect output dtype (supports BF16 and FP32)
            c_dtype_cutlass = _TORCH_TO_CUTLASS_DTYPE[c_tensor.dtype]

            if not self.use_tvm_ffi:
                a_ptr = make_ptr(
                    cutlass.BFloat16,
                    a_batched.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                b_ptr = make_ptr(
                    cutlass.BFloat16,
                    b_batched.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                c_cute_tensor = cute.runtime.from_dlpack(
                    c_tmp).mark_layout_dynamic(leading_dim=1)

                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            cache_key = (
                use_2cta_instrs,
                mma_tiler_mn,
                cluster_shape_mn,
                self.use_tvm_ffi,
                c_dtype_cutlass,
            )
            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = make_ptr(
                        cutlass.BFloat16,
                        a_batched.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    b_ptr = make_ptr(
                        cutlass.BFloat16,
                        b_batched.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    c_cute_tensor = cute.runtime.from_dlpack(
                        c_tmp).mark_layout_dynamic(leading_dim=1)
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    cutlass.Float32,  # acc_dtype
                    use_2cta_instrs=use_2cta_instrs,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm.wrapper,
                    m,
                    n,
                    k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    c_cute_tensor,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # launch gemm kernel
            if self.use_tvm_ffi:
                compiled_gemm(
                    m,
                    n,
                    k,
                    batch_size,
                    a_batched.data_ptr(),
                    b_batched.data_ptr(),
                    c_tmp,
                )
            else:
                compiled_gemm(
                    m,
                    n,
                    k,
                    batch_size,
                    a_ptr,
                    b_ptr,
                    c_cute_tensor,
                    stream=stream,
                )

            # Copy result back if original output was non-contiguous
            if c_needs_copy:
                c_tensor.copy_(c_buf)

    # input: [M, K], weight: [N, K], output: [M, N]
    @torch.library.custom_op("trtllm::cute_dsl_bf16_gemm_blackwell",
                             mutates_args=("output", ),
                             device_types="cuda")
    def cute_dsl_bf16_gemm_blackwell(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> None:
        """
        CuTe DSL BF16 GEMM for Linear layers on Blackwell.

        Computes: output = input @ weight^T
        - input: [M, K] (num_tokens, in_features)
        - weight: [N, K] (out_features, in_features)
        - output: [M, N] (num_tokens, out_features)
        """
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL BF16 GEMM only supports SM 100 family.")

        tuner = AutoTuner.get()

        runner = CuteDSLBf16BlackwellGemmRunner(use_tvm_ffi=use_tvm_ffi)

        inputs = [input, weight, output]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_bf16_gemm_blackwell::gemm",
            [runner],
            runner.__class__.tuning_config,
            inputs,
        )
        runner(inputs, tactic=best_tactic)

    @torch.library.register_fake("trtllm::cute_dsl_bf16_gemm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        output: torch.Tensor,
        use_tvm_ffi: bool = True,
    ) -> None:
        m, k = mat_a.shape[0], mat_a.shape[1]
        n = mat_b.shape[0]
        assert output.dtype in (torch.bfloat16, torch.float32), \
            "CuTe DSL bf16 gemm output dtype must be bf16 or fp32"
        assert output.shape == (
            m, n), "CuTe DSL bf16 gemm output shape is incorrect"

    # ------------------------------------------------------------------ #
    #  CuTE DSL FP4 Paged MQA Logits (Blackwell SM100)                   #
    # ------------------------------------------------------------------ #

    class CuteDSLFP4PagedMQALogitsRunner:
        """Runner for CuTe DSL FP4 Paged MQA Logits kernel (Blackwell SM100).

        Caches compiled kernels keyed by static params
        (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n,
         num_sms, num_epi_subtiles, epi_dtype, output_dtype).
        FP4 locks acc_dtype to fp32 internally.
        """

        kernel_cache = dict()

        @classmethod
        def _compile(cls,
                     compute_block_kv,
                     phys_block_kv,
                     num_heads,
                     head_dim,
                     next_n,
                     num_sms,
                     num_epi_subtiles,
                     epi_dtype,
                     output_dtype,
                     remove_online_sf_transpose=False,
                     emit_block_meta=False,
                     emit_hit_stats=True,
                     emit_seed_counts=False,
                     seed_packed=False,
                     emit_cand=False,
                     cand_cap=5120,
                     emit_cand_bucketed=False,
                     accept_cap=8192):
            """Compile kernel using fake tensors + TVM FFI."""
            key = (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n,
                   num_sms, num_epi_subtiles, epi_dtype, output_dtype,
                   remove_online_sf_transpose, emit_block_meta, emit_hit_stats,
                   emit_seed_counts, seed_packed, emit_cand, cand_cap,
                   emit_cand_bucketed, accept_cap)
            if key in cls.kernel_cache:
                return

            to_cutlass = _TORCH_TO_CUTLASS_DTYPE
            N = next_n * num_heads
            half_head_dim = head_dim // 2
            # FP4 fused per-block bytes: data (phys_block_kv * D/2) + SF (phys_block_kv * 4)
            block_bytes = phys_block_kv * (half_head_dim + 4)

            sym_num_phys_blocks = cute.sym_int()
            sym_B = cute.sym_int()
            max_ctx = cute.sym_int()
            max_blocks_per_seq = cute.sym_int()
            num_ctas = cute.sym_int()

            # KV may come from the indexer K-cache pool view, which is
            # strided in dim 0 (pool layout interleaves layers:
            # [num_blocks, num_layers, kvFactor, blockSize]). Declare outer
            # stride as sym so the actual per-block stride is read at
            # runtime; innermost stride is fixed to 1 (byte-contig within a
            # logical block view).
            kv_fake = cute.runtime.make_fake_tensor(
                cutlass.Uint8, (sym_num_phys_blocks, block_bytes),
                stride=(cute.sym_int64(), 1))

            # Q is FP4 packed bytes: head_dim/2 bytes per row
            q_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (N, half_head_dim, sym_B),
                stride_order=(1, 0, 2))

            # sf_q has shape (N, B); kernel TMA descriptor tile = real N
            # (no GMEM pad). SMEM/UTCCP padding to N_padded is handled inside
            # the kernel.
            sf_q_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                              (N, sym_B),
                                                              stride_order=(0,
                                                                            1))

            if epi_dtype == torch.float16:
                w_dtype = cutlass.Float16
            elif epi_dtype == torch.bfloat16:
                w_dtype = cutlass.BFloat16
            else:
                w_dtype = cutlass.Float32
            w_fake = cute.runtime.make_fake_compact_tensor(w_dtype, (N, sym_B),
                                                           stride_order=(0, 1))

            logits_fake = cute.runtime.make_fake_tensor(
                to_cutlass[output_dtype], (cute.sym_int(), max_ctx),
                stride=(cute.sym_int64(), 1))

            bt_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_B, max_blocks_per_seq), stride_order=(1, 0))

            cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                            (sym_B, ),
                                                            stride_order=(0, ))

            sm_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32,
                                                            (num_ctas, 2),
                                                            stride_order=(1, 0))

            # Block-meta tensors (fused-GVR support): nb_pad*4 records.
            block_max_fake = None
            hit_stats_fake = None
            hit_bitmap_fake = None
            if emit_block_meta:
                nb_sym = cute.sym_int()
                block_max_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Float32, (cute.sym_int(), nb_sym),
                    stride_order=(1, 0),
                    assumed_align=16)
                if emit_hit_stats:
                    hit_stats_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Float32, (cute.sym_int(), 4),
                        stride_order=(1, 0),
                        assumed_align=16)
                    hit_bitmap_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Int32, (sym_B, cute.sym_int()),
                        stride_order=(1, 0),
                        assumed_align=16)
            cand_fake = None
            cand_ctl_fake = None
            if emit_cand:
                cand_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (cute.sym_int(), cand_cap * 2),
                    stride_order=(1, 0),
                    assumed_align=8)
                cand_ctl_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (cute.sym_int(), 2),
                    stride_order=(1, 0),
                    assumed_align=8)
            cand_idx_fake = None
            cand_cur_fake = None
            if emit_cand_bucketed:
                # bucketed SoA: cand slot reused as the fp32 VALUES tensor
                wtot = 2 * accept_cap + cand_cap
                cand_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Float32, (cute.sym_int(), wtot),
                    stride_order=(1, 0),
                    assumed_align=4)
                cand_idx_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (cute.sym_int(), wtot),
                    stride_order=(1, 0),
                    assumed_align=4)
                cand_ctl_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (cute.sym_int(), 4),
                    stride_order=(1, 0),
                    assumed_align=4)
                cand_cur_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (cute.sym_int(), 4),
                    stride_order=(1, 0),
                    assumed_align=4)
            seed_thr_fake = None
            seed_counts_fake = None
            if emit_seed_counts:
                if seed_packed:
                    # [rows, 8] fp32 packed seed row: lines at cols 0..2,
                    # counts at cols 3..5; same tensor bound to both params.
                    seed_thr_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Float32, (cute.sym_int(), 8),
                        stride_order=(1, 0),
                        assumed_align=4)
                    seed_counts_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Float32, (cute.sym_int(), 8),
                        stride_order=(1, 0),
                        assumed_align=4)
                else:
                    seed_thr_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Float32, (cute.sym_int(), 3),
                        stride_order=(1, 0),
                        assumed_align=4)
                    seed_counts_fake = cute.runtime.make_fake_compact_tensor(
                        cutlass.Int32, (cute.sym_int(), 3),
                        stride_order=(1, 0),
                        assumed_align=4)

            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)

            kernel = FP4MQALogitsKernel(
                block_kv=compute_block_kv,
                phys_block_kv=phys_block_kv,
                num_heads=num_heads,
                head_dim=head_dim,
                next_n=next_n,
                num_sms=num_sms,
                num_epi_subtiles=num_epi_subtiles,
                epi_dtype=to_cutlass[epi_dtype],
                output_dtype=to_cutlass[output_dtype],
                remove_online_sf_transpose=remove_online_sf_transpose,
                emit_block_meta=emit_block_meta,
                emit_hit_stats=emit_hit_stats,
                emit_seed_counts=emit_seed_counts,
                seed_packed=seed_packed,
                emit_cand=emit_cand,
                cand_cap=cand_cap,
                emit_cand_bucketed=emit_cand_bucketed,
                accept_cap=accept_cap,
            )

            compiled = cute.compile(
                kernel,
                kv_fake,
                q_fake,
                sf_q_fake,
                w_fake,
                logits_fake,
                bt_fake,
                cl_fake,
                sm_fake,
                cutlass.Int32(1),
                cutlass.Int32(1),
                # keep __call__'s argument order: stream, then emission
                # slots (the runtime call drops the stream, same sequence)
                fake_stream,
                block_max_fake,
                hit_stats_fake,
                hit_bitmap_fake,
                seed_thr=seed_thr_fake,
                seed_counts=seed_counts_fake,
                cand=cand_fake,
                cand_ctl=cand_ctl_fake,
                cand_idx_t=cand_idx_fake,
                cand_cur=cand_cur_fake,
                options="--enable-tvm-ffi",
            )
            cls.kernel_cache[key] = compiled
            logger.debug(f"[compile cute_dsl fp4_paged_mqa_logits] {key}")

        @classmethod
        def forward(
            cls,
            q: torch.Tensor,
            sf_q: torch.Tensor,
            kv_fused: torch.Tensor,
            weights: torch.Tensor,
            context_lens: torch.Tensor,
            block_table: torch.Tensor,
            schedule_meta: torch.Tensor,
            max_context_len: int,
            num_epi_subtiles: int = 1,
            epi_dtype: torch.dtype = torch.float32,
            output_dtype: torch.dtype = torch.float32,
            remove_online_sf_transpose: bool = False,
            emit_block_meta: bool = False,
            emit_hit_stats: bool = True,
            hit_bitmap: Optional[torch.Tensor] = None,
            block_max_out: Optional[torch.Tensor] = None,
            hit_stats_out: Optional[torch.Tensor] = None,
            emit_seed_counts: bool = False,
            seed_thr: Optional[torch.Tensor] = None,
            seed_counts_out: Optional[torch.Tensor] = None,
            emit_cand: bool = False,
            cand_out: Optional[torch.Tensor] = None,
            cand_ctl_out: Optional[torch.Tensor] = None,
            emit_cand_bucketed: bool = False,
            accept_cap: int = 8192,
            cand_idx_out: Optional[torch.Tensor] = None,
            cand_cur_out: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Execute FP4 paged MQA logits kernel.

            Args:
                q: [B, next_n, H, D//2] uint8 (FP4 packed)
                sf_q: [B, next_n, H] int32 (4 UE8M0 packed per token)
                kv_fused: [num_blocks, phys_block_kv, 1, D//2 + 4] uint8
                weights: [B*next_n, H] float32
                context_lens: [B] int32
                block_table: [B, max_blocks] int32
                schedule_meta: [num_sms+1, 2] int32
                max_context_len: int
                num_epi_subtiles: epilogue sub-tile count (1, 2, or 4)
                epi_dtype: epilogue compute dtype
                output_dtype: output logits dtype
                emit_block_meta: also emit per-128-block metadata for the
                    fused GVR top-k. Requires ``hit_bitmap``
                    [B, >= nb_pad*4] int32 (1 bit per compressed kv
                    position, request-level). ``block_max_out``
                    [B*next_n, nb_pad*4] fp32 (4 warp-partial records per
                    block) is allocated when not supplied;
                    ``hit_stats_out`` [B*next_n, 4] fp32 must be supplied
                    when ``emit_hit_stats`` is set.
            Returns:
                logits [B*next_n, max_context_len]; with emit_block_meta,
                the tuple (logits, block_max, hit_stats).

            The optional emission tensors (``block_max_out`` /
            ``seed_thr`` / ``cand_*``) are written by the kernel but
            cannot be declared in ``mutates_args``: torch.library raises
            IndexError when a declared-mutable Optional arg is None at
            call time (re-verified on the pinned torch), and plain
            logits-only calls pass none of them. Emission is therefore
            eager / CUDA-graph only.
            """
            B, next_n, H, half_D = q.shape
            N = next_n * H
            D = half_D * 2
            phys_block_kv = kv_fused.shape[1]
            compute_block_kv = 128
            num_phys_blocks = kv_fused.shape[0]
            num_sms = _get_num_sms()

            # Reshape Q: [B, next_n, H, D/2] -> [B, N, D/2] -> [N, D/2, B]
            # NOTE: do NOT call .contiguous() — that would repack memory and
            # produce strides depending on B, breaking the fake tensor compile
            # cache (which assumes stride_order with half_D innermost).
            # The permute view alone gives strides (half_D, 1, N*half_D) which
            # are B-independent and match the compile-time fake stride.
            q_3d = q.reshape(B, N, half_D).permute(1, 2, 0)

            # Reshape sf_q: [B, next_n, H] -> [B, N] -> [N, B]
            # No GMEM pad — kernel TMA descriptor uses tile=N (real), so TMA
            # only fetches N int32 from GMEM. SMEM is still N_padded for UTCCP
            # alignment; the SMEM tail (N..N_padded) is left as garbage and
            # never read by MMA (UMMA_N=N) or epilogue (acc cols [0,N) only).
            # Mirrors DeepGEMM's pattern (kRealNumSFQAtom=N, kNumSFQAtom=N_pad).
            sf_q_2d = sf_q.reshape(B, N).t()  # (N, B), strides (1, N)

            # Reshape weights: [B*next_n, H] -> [B, N] -> [N, B] (cast to epi_dtype)
            # NOTE: no .contiguous() — same reason as q_3d above.
            if epi_dtype == torch.float16:
                w_2d = weights.reshape(B, N).half().t()
            elif epi_dtype == torch.bfloat16:
                w_2d = weights.reshape(B, N).bfloat16().t()
            else:
                w_2d = weights.reshape(B, N).t()

            # Flatten fused KV to [num_phys_blocks, block_bytes]
            kv_flat = kv_fused.reshape(num_phys_blocks, -1)

            # Allocate output with alignment padding
            SPLIT_KV = compute_block_kv * 2  # NUM_MATH_WG = 2
            aligned_max_ctx = (
                (max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
            # Use a persistent arena buffer instead of a per-forward torch.empty
            # so the output address stays stable across CUDA-graph replays.
            _reserve = torch.cuda.is_current_stream_capturing()
            logits = get_memory_buffers().get_buffer(
                [B * next_n, aligned_max_ctx],
                output_dtype,
                buffer_name="cute_dsl_mqa_logits",
                reserve_buffer=_reserve,
            )
            logits = logits[:, :max_context_len]

            # Block-meta buffers (fused-GVR support). nb_pad mirrors the
            # logits padding so WG1's odd-num_kv OOB tile lands in padding.
            if emit_block_meta:
                nb_pad = aligned_max_ctx // compute_block_kv
                # 4 warp-partial records per block (see FP4MQALogitsKernel).
                nrec = nb_pad * 4
                if emit_hit_stats:
                    assert (
                        hit_bitmap is not None
                        and hit_bitmap.dtype == torch.int32
                        and hit_bitmap.is_cuda and hit_bitmap.is_contiguous()
                        and hit_bitmap.dim() == 2 and hit_bitmap.shape[0] == B
                        and hit_bitmap.shape[1] >= nb_pad * 4
                    ), (f"emit_hit_stats requires hit_bitmap int32 "
                        f"[{B}, >= {nb_pad * 4}]; got "
                        f"{None if hit_bitmap is None else (hit_bitmap.dtype, tuple(hit_bitmap.shape))}"
                        )
                    # Per-row aggregate {enc_min, enc_max, sum, cnt}. The
                    # kernel only ATOMICALLY MERGES into this buffer — the
                    # caller must pre-initialize it to the identities
                    # {enc(+FLT_MAX), enc(-FLT_MAX), 0, 0} each step.
                    assert hit_stats_out is not None, (
                        "emit_hit_stats requires a caller-initialized "
                        "hit_stats_out [B*next_n, 4] fp32 (identity-filled)")
                    assert (hit_stats_out.shape == (B * next_n, 4)
                            and hit_stats_out.is_contiguous())
                else:
                    hit_bitmap = None
                    hit_stats_out = None
                if block_max_out is None:
                    block_max_out = torch.empty((B * next_n, nrec),
                                                device=q.device,
                                                dtype=torch.float32)
                assert (block_max_out.shape == (B * next_n, nrec)
                        and block_max_out.is_contiguous())

            seed_packed = False
            if emit_seed_counts:
                assert emit_block_meta, (
                    "emit_seed_counts requires emit_block_meta")
                if seed_counts_out is None:
                    # Packed contract: seed_thr IS the [rows, 8] fp32 seed
                    # row; lines at cols 0..2, counts accumulate as fp32 at
                    # cols 3..5. Caller zeroes cols 3..7 and writes lines
                    # each step.
                    seed_packed = True
                    assert (
                        seed_thr is not None and seed_thr.dtype == torch.float32
                        and seed_thr.is_cuda and seed_thr.is_contiguous()
                        and seed_thr.shape == (B * next_n, 8)
                    ), (f"packed emit_seed_counts requires seed_thr fp32 "
                        f"[{B * next_n}, 8]; got "
                        f"{None if seed_thr is None else (seed_thr.dtype, tuple(seed_thr.shape))}"
                        )
                    seed_counts_out = seed_thr
                else:
                    # Legacy split contract: 3 thresholds per row (fp32),
                    # counts accumulated with red.global.add.s32 into a
                    # caller-zeroed int32 [rows, 3].
                    assert (
                        seed_thr is not None and seed_thr.dtype == torch.float32
                        and seed_thr.is_cuda and seed_thr.is_contiguous()
                        and seed_thr.shape == (B * next_n, 3)
                    ), (f"emit_seed_counts requires seed_thr fp32 "
                        f"[{B * next_n}, 3]; got "
                        f"{None if seed_thr is None else (seed_thr.dtype, tuple(seed_thr.shape))}"
                        )
                    assert (seed_counts_out.dtype == torch.int32
                            and seed_counts_out.is_cuda
                            and seed_counts_out.is_contiguous()
                            and seed_counts_out.shape == (B * next_n, 3)), (
                                "emit_seed_counts requires a caller-zeroed "
                                "seed_counts_out int32 [B*next_n, 3]")
            else:
                seed_thr = None
                seed_counts_out = None

            cand_cap = 0
            if emit_cand:
                assert emit_seed_counts, (
                    "emit_cand requires emit_seed_counts (t_0 threshold)")
                # Unordered (value, index) pair scatter; the caller zeroes
                # cand_ctl_out {claimed, void} each step. cand_out is
                # [B*next_n, CAP*2] int32 (fp32 bits in even words).
                assert (
                    cand_out is not None and cand_out.dtype == torch.int32
                    and cand_out.is_cuda and cand_out.is_contiguous()
                    and cand_out.dim() == 2 and cand_out.shape[0] == B * next_n
                    and cand_out.shape[1] % 2 == 0 and cand_out.shape[1] > 0), (
                        "emit_cand requires cand_out int32 [B*next_n, CAP*2]")
                assert (cand_ctl_out is not None
                        and cand_ctl_out.dtype == torch.int32
                        and cand_ctl_out.is_cuda
                        and cand_ctl_out.is_contiguous()
                        and cand_ctl_out.shape == (B * next_n, 2)), (
                            "emit_cand requires a caller-zeroed cand_ctl_out "
                            "int32 [B*next_n, 2]")
                cand_cap = cand_out.shape[1] // 2
            elif emit_cand_bucketed:
                assert emit_seed_counts, (
                    "emit_cand_bucketed requires emit_seed_counts")
                # SoA contract: cand_out = fp32 VALUES [rows, 2*segA+capC],
                # cand_idx_out = int32 positions (same width), cand_cur_out =
                # int32 [rows, 4] cursors (caller-zeroed), cand_ctl_out =
                # int32 [rows, 4] {n0, void, n1, n2} (caller-zeroed)
                assert cand_out is not None, (
                    "emit_cand_bucketed requires cand_out")
                W = cand_out.shape[1]
                assert (
                    cand_out.dtype == torch.float32 and cand_out.is_cuda
                    and cand_out.is_contiguous() and cand_out.dim() == 2
                    and cand_out.shape[0] == B * next_n
                    and W > 2 * accept_cap), (
                        "bucketed requires cand_out fp32 [rows, 2*segA+capC]")
                assert (cand_idx_out is not None
                        and cand_idx_out.dtype == torch.int32
                        and cand_idx_out.is_cuda
                        and cand_idx_out.is_contiguous()
                        and cand_idx_out.shape == cand_out.shape), (
                            "bucketed requires cand_idx_out int32, same shape")
                assert (cand_ctl_out is not None
                        and cand_ctl_out.dtype == torch.int32
                        and cand_ctl_out.is_cuda
                        and cand_ctl_out.is_contiguous()
                        and cand_ctl_out.shape == (B * next_n, 4)), (
                            "bucketed requires caller-zeroed cand_ctl_out "
                            "int32 [rows, 4]")
                assert (cand_cur_out is not None
                        and cand_cur_out.dtype == torch.int32
                        and cand_cur_out.is_cuda
                        and cand_cur_out.is_contiguous()
                        and cand_cur_out.shape == (B * next_n, 4)), (
                            "bucketed requires caller-zeroed cand_cur_out "
                            "int32 [rows, 4]")
                cand_cap = W - 2 * accept_cap
            else:
                cand_out = None
                cand_ctl_out = None
            if not emit_cand_bucketed:
                cand_idx_out = None
                cand_cur_out = None

            # Compile if needed (fake tensors, no real data required)
            key = (compute_block_kv, phys_block_kv, H, D, next_n, num_sms,
                   num_epi_subtiles, epi_dtype, output_dtype,
                   remove_online_sf_transpose, emit_block_meta, emit_hit_stats,
                   emit_seed_counts, seed_packed, emit_cand, cand_cap,
                   emit_cand_bucketed, accept_cap)
            if key not in cls.kernel_cache:
                cls._compile(
                    compute_block_kv,
                    phys_block_kv,
                    H,
                    D,
                    next_n,
                    num_sms,
                    num_epi_subtiles,
                    epi_dtype,
                    output_dtype,
                    remove_online_sf_transpose=remove_online_sf_transpose,
                    emit_block_meta=emit_block_meta,
                    emit_hit_stats=emit_hit_stats,
                    emit_seed_counts=emit_seed_counts,
                    seed_packed=seed_packed,
                    emit_cand=emit_cand,
                    cand_cap=cand_cap,
                    emit_cand_bucketed=emit_cand_bucketed,
                    accept_cap=accept_cap)
            compiled = cls.kernel_cache[key]

            # TVM FFI: pass raw tensors, no dlpack/stream needed
            if emit_block_meta:
                compiled(kv_flat, q_3d, sf_q_2d, w_2d, logits, block_table,
                         context_lens, schedule_meta, num_phys_blocks, B,
                         block_max_out, hit_stats_out, hit_bitmap, seed_thr,
                         seed_counts_out, cand_out, cand_ctl_out, cand_idx_out,
                         cand_cur_out)
                return logits, block_max_out, hit_stats_out
            compiled(kv_flat, q_3d, sf_q_2d, w_2d, logits, block_table,
                     context_lens, schedule_meta, num_phys_blocks, B, None,
                     None, None, None, None, None, None, None, None)
            return logits

    # NOTE: the optional emission tensors ARE written by the kernel but must
    # stay out of mutates_args (torch.library IndexErrors on None defaults).
    @torch.library.custom_op("trtllm::cute_dsl_fp4_paged_mqa_logits",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_fp4_paged_mqa_logits(
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
        remove_online_sf_transpose: bool = False,
        block_max_out: Optional[torch.Tensor] = None,
        seed_thr: Optional[torch.Tensor] = None,
        cand_out: Optional[torch.Tensor] = None,
        cand_idx_out: Optional[torch.Tensor] = None,
        cand_ctl_out: Optional[torch.Tensor] = None,
        cand_cur_out: Optional[torch.Tensor] = None,
        accept_cap: int = 8192,
    ) -> torch.Tensor:
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL FP4 Paged MQA Logits only supports SM 100 family.")
        if num_epi_subtiles not in (1, 2, 4):
            raise ValueError(
                f"num_epi_subtiles must be one of (1, 2, 4), got {num_epi_subtiles}"
            )
        # Caller (dsa.py) prepares all tensors with metadata-guaranteed
        # dtype/shape; skip per-call validation to keep decode-hot-path
        # latency low. Log inputs once for debugging.
        logger.info_once(
            f"cute_dsl_fp4_paged_mqa_logits inputs: "
            f"q dtype={q.dtype} shape={tuple(q.shape)} stride={q.stride()}; "
            f"sf_q dtype={sf_q.dtype} shape={tuple(sf_q.shape)} stride={sf_q.stride()}; "
            f"kv_fused dtype={kv_fused.dtype} shape={tuple(kv_fused.shape)} stride={kv_fused.stride()}; "
            f"weights dtype={weights.dtype} shape={tuple(weights.shape)} stride={weights.stride()}; "
            f"context_lens dtype={context_lens.dtype} shape={tuple(context_lens.shape)}; "
            f"block_table dtype={block_table.dtype} shape={tuple(block_table.shape)} stride={block_table.stride()}; "
            f"schedule_meta dtype={schedule_meta.dtype} shape={tuple(schedule_meta.shape)}; "
            f"max_context_len={max_context_len} num_epi_subtiles={num_epi_subtiles} "
            f"epi_dtype={epi_dtype} output_dtype={output_dtype}",
            key="cute_dsl_fp4_paged_mqa_logits_inputs",
        )
        ret = CuteDSLFP4PagedMQALogitsRunner.forward(
            q,
            sf_q,
            kv_fused,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            num_epi_subtiles=num_epi_subtiles,
            epi_dtype=epi_dtype,
            output_dtype=output_dtype,
            remove_online_sf_transpose=remove_online_sf_transpose,
            emit_block_meta=block_max_out is not None,
            emit_hit_stats=False,
            block_max_out=block_max_out,
            emit_seed_counts=seed_thr is not None,
            seed_thr=seed_thr,
            emit_cand_bucketed=cand_out is not None,
            accept_cap=accept_cap,
            cand_out=cand_out,
            cand_idx_out=cand_idx_out,
            cand_ctl_out=cand_ctl_out,
            cand_cur_out=cand_cur_out)
        # with emission on the runner returns a tuple; the op returns
        # logits only (emission buffers are caller-owned)
        return ret[0] if isinstance(ret, tuple) else ret

    @torch.library.register_fake("trtllm::cute_dsl_fp4_paged_mqa_logits")
    def _(
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
        remove_online_sf_transpose: bool = False,
        block_max_out: Optional[torch.Tensor] = None,
        seed_thr: Optional[torch.Tensor] = None,
        cand_out: Optional[torch.Tensor] = None,
        cand_idx_out: Optional[torch.Tensor] = None,
        cand_ctl_out: Optional[torch.Tensor] = None,
        cand_cur_out: Optional[torch.Tensor] = None,
        accept_cap: int = 8192,
    ) -> torch.Tensor:
        B = q.shape[0]
        next_n = q.shape[1]
        return torch.empty(B * next_n,
                           max_context_len,
                           dtype=output_dtype,
                           device=q.device)

    # =========================================================================
    # MLA decode (Blackwell) - wraps the CuTe DSL kernels that live at
    # tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla/.
    # Used by the cute_dsl_mla FMHA library (see attention_backend/fmha/cute_dsl_mla.py).
    # =========================================================================

    from ..cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import \
        BlackwellMultiHeadLatentAttentionForwardFP8
    from ..cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp16 import \
        BlackwellMultiHeadLatentAttentionForwardFP16

    class CuteDSLNVMlaDecodeBlackwellRunner(TunableRunner):
        """Generic TunableRunner for the Blackwell CuTe DSL MLA decode kernels.

        Works for FP8, FP16, and BF16 - pass the cutlass input dtype at
        construction; the kernel class is derived from it:

            CuteDSLNVMlaDecodeBlackwellRunner(
                in_dtype=cutlass.Float8E4M3FN, ...)   # -> ...ForwardFP8
            CuteDSLNVMlaDecodeBlackwellRunner(
                in_dtype=cutlass.Float16, ...)        # -> ...ForwardFP16
            CuteDSLNVMlaDecodeBlackwellRunner(
                in_dtype=cutlass.BFloat16, ...)       # -> ...ForwardFP16
        """
        kernel_cache = dict()
        tuning_config_cache = dict()

        _CLUSTER_SHAPE_MNK = (2, 1, 1)

        # in_dtype -> kernel class. The kernels' own ``can_implement`` is
        # what ultimately rejects unsupported dtypes, but this lookup
        # picks which kernel we even try to compile.
        _KERNEL_CLASS_BY_DTYPE = {
            cutlass.Float8E4M3FN: BlackwellMultiHeadLatentAttentionForwardFP8,
            cutlass.Float16: BlackwellMultiHeadLatentAttentionForwardFP16,
            cutlass.BFloat16: BlackwellMultiHeadLatentAttentionForwardFP16,
        }

        # Fixed kernel-construction flags for this integration path (var-seq
        # decode, scalar split): not tunable, not part of any cache key.
        _IS_VAR_SEQ = True
        _IS_VAR_SPLIT_KV = False
        _SKIP_CORRECTION_THRESHOLD = 0.0

        _WORKSPACE_ALIGN = 128
        _LSE_DTYPE_BYTES = 4  # float32

        def __init__(
            self,
            in_dtype,
            num_heads: int,
            seq_len_q: int,
            page_size: int,
            max_batch_size: int = 0,
            emit_softmax_stats: bool = False,
        ):
            super().__init__()
            kernel_class = self.__class__._KERNEL_CLASS_BY_DTYPE.get(in_dtype)
            if kernel_class is None:
                raise ValueError(
                    f"CuteDSLNVMlaDecodeBlackwellRunner: unsupported "
                    f"in_dtype={in_dtype}. Supported: "
                    f"{list(self.__class__._KERNEL_CLASS_BY_DTYPE.keys())}")
            self.in_dtype = in_dtype
            self.kernel_class = kernel_class
            self.num_heads = num_heads
            self.seq_len_q = seq_len_q
            self.page_size = page_size
            self.max_batch_size = max_batch_size
            self.emit_softmax_stats = emit_softmax_stats

        def unique_id(self):
            # seq_len_q is part of the id: each decode variant (the MTP
            # target step's sq = 1 + draft_len, the draft steps' sq = 1)
            # constructs its own runner and is tuned independently during
            # the autotuner warmup's generation forward.
            base_id = (
                self.in_dtype,
                self.num_heads,
                self.seq_len_q,
                self.page_size,
            )
            return base_id + (True, ) if self.emit_softmax_stats else base_id

        @classmethod
        def _get_max_active_blocks(cls) -> int:
            """``max_active_clusters * cluster_shape[0]`` -- the occupancy ceiling
            the split_kv heuristic divides. Queried once via HardwareInfo and
            cached at class scope; must be populated by an eager warmup before
            CUDA-graph capture (HardwareInfo cannot run during capture)."""
            cached = getattr(cls, "_cute_dsl_max_active_blocks", None)
            if cached is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CuteDSLNVMlaDecodeBlackwellRunner: max_active_blocks was "
                        "not cached before CUDA graph capture (run an eager "
                        "warmup first).")
                cluster_product = (cls._CLUSTER_SHAPE_MNK[0] *
                                   cls._CLUSTER_SHAPE_MNK[1] *
                                   cls._CLUSTER_SHAPE_MNK[2])
                max_active_clusters = cutlass.utils.HardwareInfo(
                ).get_max_active_clusters(cluster_product)
                cached = int(max_active_clusters) * cls._CLUSTER_SHAPE_MNK[0]
                cls._cute_dsl_max_active_blocks = cached
            return cached

        @staticmethod
        def get_default_split_kv(B: int, S: int, max_active_blocks: int) -> int:
            max_split_kv = 32
            blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
            split_kv = min(blocks_per_batch, max_split_kv)
            return split_kv

        @staticmethod
        def get_default_is_persistent(B: int) -> bool:
            if B >= 64:
                return True
            else:
                return False

        @staticmethod
        def get_split_kv_candidates(B: int, S: int,
                                    max_active_blocks: int) -> List[int]:
            # TODO: default split_kv is not always the best choice. We need to optimize it.
            return [
                CuteDSLNVMlaDecodeBlackwellRunner.get_default_split_kv(
                    B, S, max_active_blocks)
            ]

        @staticmethod
        def get_is_persistent_candidates() -> List[bool]:
            return [True, False]

        @classmethod
        def get_max_split_kv_workspace_size(
            cls,
            H: int,
            D: int,
            acc_dtype: Type[cutlass.Numeric],
        ) -> int:
            """Raw bytes reserved for split-KV intermediates.

            Batch-INDEPENDENT: CUDA graphs are captured per batch size in
            descending order, and a later capture that needed a larger workspace
            would resize the buffer, dangling the address baked into every
            previously captured graph.

            # cuda graph capture(B=8):   eager warmup N times  →  capture graph_8
            # cuda graph capture(B=4):   eager warmup N times  →  capture graph_4
            # ...
            # cuda graph replay
            # A later capture with a bigger workspace would resize it, so the
            # bound covers every batch size up-front."""
            max_active_blocks = cls._get_max_active_blocks()
            return (2 * H * (max_active_blocks // 2) * (D + 1) *
                    acc_dtype.width // 8)

        @classmethod
        def get_workspace_layout(
            cls,
            H: int,
            seq_len_q: int,
            D: int,
            max_batch_size: int,
            acc_dtype: Type[cutlass.Numeric],
        ) -> Tuple[int, int, int, int, int]:
            """Return LSE/split-KV offsets and sizes, then total bytes."""
            lse_offset = 0
            lse_size = cls.get_lse_workspace_size(H, seq_len_q, max_batch_size)
            split_kv_offset = pad_up(lse_offset + lse_size,
                                     cls._WORKSPACE_ALIGN)
            split_kv_size = cls.get_max_split_kv_workspace_size(H, D, acc_dtype)
            workspace_size = split_kv_offset + pad_up(split_kv_size,
                                                      cls._WORKSPACE_ALIGN)
            return (lse_offset, lse_size, split_kv_offset, split_kv_size,
                    workspace_size)

        @classmethod
        def get_lse_workspace_size(
            cls,
            H: int,
            seq_len_q: int,
            batch_size: int,
        ) -> int:
            """Raw bytes for an LSE tensor shaped ``(batch_size, seq_len_q, H)``."""
            return seq_len_q * batch_size * H * cls._LSE_DTYPE_BYTES

        @classmethod
        def get_max_padded_workspace_size(
            cls,
            H: int,
            seq_len_q: int,
            D: int,
            max_batch_size: int,
            acc_dtype: Type[cutlass.Numeric],
        ) -> int:
            """Padded bytes for the max-batch LSE and split-KV regions."""
            return cls.get_workspace_layout(H, seq_len_q, D, max_batch_size,
                                            acc_dtype)[4]

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
            """Filter the candidate tilers via the kernel's
            ``can_implement``. Returns a list of
            ``(mma_qk_tiler_mn, mma_pv_tiler_mn)`` tuples; AutoTuner picks
            one and passes it to ``forward`` as ``tactic``.
            """
            if get_sm_version() not in (100, 103):
                return []
            q_latent, q_rope, _c_latent, _c_rope, _page_table, cache_seqs, \
                o, *_rest = inputs
            h, latent_dim, seq_len_q, _ = q_latent.shape
            rope_dim = q_rope.shape[1]
            batch_size = cache_seqs.shape[0]
            if o.dtype == torch.float16:
                out_dtype = cutlass.Float16
            elif o.dtype == torch.bfloat16:
                out_dtype = cutlass.BFloat16
            else:
                out_dtype = self.in_dtype

            candidate_tiler_tactics = [
                ((128, 128), (128, 256)),
            ]
            max_active_blocks = self._get_max_active_blocks()
            split_candidates = self.get_split_kv_candidates(
                batch_size, seq_len_q, max_active_blocks)
            persistent_candidates = self.get_is_persistent_candidates()

            valid = []
            for (mma_qk_tiler_mn,
                 mma_pv_tiler_mn), split_kv, is_persistent in itertools.product(
                     candidate_tiler_tactics, split_candidates,
                     persistent_candidates):
                if self.kernel_class.can_implement(
                        batch_size,
                        seq_len_q,
                        self.page_size,
                        h,
                        latent_dim,
                        rope_dim,
                        self.in_dtype,  # in_dtype
                        out_dtype,
                        cutlass.Float32,  # acc_dtype
                        cutlass.Float32,  # lse_dtype
                        mma_qk_tiler_mn,
                        mma_pv_tiler_mn,
                        split_kv,
                        is_persistent,
                        self._IS_VAR_SEQ,
                        self._IS_VAR_SPLIT_KV,
                        self.page_size,
                ):
                    valid.append((mma_qk_tiler_mn, mma_pv_tiler_mn, split_kv,
                                  is_persistent))
                else:
                    logger.debug(
                        "CuteDSLNVMlaDecodeBlackwellRunner.can_implement "
                        "rejected tactic: kernel=%s in_dtype=%s "
                        "H=%d L=%d R=%d S=%d B=%d page_size=%d "
                        "mma_qk=%s mma_pv=%s persistent=%s var_seq=%s "
                        "var_split=%s", self.kernel_class.__name__,
                        self.in_dtype, h, latent_dim, rope_dim, seq_len_q,
                        batch_size, self.page_size, mma_qk_tiler_mn,
                        mma_pv_tiler_mn, is_persistent, self._IS_VAR_SEQ,
                        self._IS_VAR_SPLIT_KV)
            return valid

        def _tuning_inputs_pre_hook(
                self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            """Fix up the RECONSTRUCTED profiling tensors so the decode kernel
            compiles and runs in-bounds during AutoTuner profiling."""
            inputs = list(inputs)

            def _relayout(t, base_shape, permute_order):
                # Same logical shape as ``t``, but with the real
                # (leading-dim-contiguous) strides.
                out = torch.empty(base_shape, dtype=t.dtype,
                                  device=t.device).permute(*permute_order)
                out.copy_(t)
                return out

            # q_latent [H, D, S_q, B] <- (B, S_q, H, D).permute(2, 3, 1, 0)
            H, d_latent, seq_len_q, batch = inputs[0].shape
            inputs[0] = _relayout(inputs[0], (batch, seq_len_q, H, d_latent),
                                  (2, 3, 1, 0))
            d_rope = inputs[1].shape[1]
            inputs[1] = _relayout(inputs[1], (batch, seq_len_q, H, d_rope),
                                  (2, 3, 1, 0))  # q_rope
            inputs[6] = _relayout(inputs[6], (batch, seq_len_q, H, d_latent),
                                  (2, 3, 1, 0))  # o

            # page_table [max_blocks, B] <- (B, max_blocks).transpose(0, 1),
            # with in-bounds page ids ([0, num_pages) from the c_latent pool).
            page_table = inputs[4]
            max_blocks = int(page_table.shape[0])
            num_pages = int(inputs[2].shape[2])
            pt_valid = (page_table.to(torch.long).abs() % num_pages).to(
                page_table.dtype)
            pt_out = torch.empty((batch, max_blocks),
                                 dtype=page_table.dtype,
                                 device=page_table.device).transpose(0, 1)
            pt_out.copy_(pt_valid)
            inputs[4] = pt_out

            cache_seqs = inputs[5]
            if isinstance(cache_seqs, torch.Tensor) and cache_seqs.numel():
                max_kv = max_blocks * self.page_size
                kv = max(1, min(2048, max_kv))
                inputs[5] = torch.full((cache_seqs.shape[0], ),
                                       kv,
                                       dtype=cache_seqs.dtype,
                                       device=cache_seqs.device)
            return inputs

        def get_tuning_config(self) -> TuningConfig:
            """Batch is the single free tuning dim (on cache_seqs), and its
            power-of-2 bucket ladder is determined by ``max_batch_size``
            (when known, i.e. > 0), NOT by the batch observed while tuning:
            one in-autotune forward at ANY batch then profiles every bucket
            up to the engine's max batch, so no runtime batch falls to
            ``default_tactic`` (whose freshly computed split_kv could
            JIT-compile a kernel inside the timed region).

            Every other dim carrying batch is tied to it via constraints.
            seq_len_q needs no dynamic axis: it is fixed per runner (part of
            ``unique_id``), and each decode variant is tuned by its own
            forward inside the autotuner warmup."""
            key = self.unique_id()
            cache = self.__class__.tuning_config_cache
            if key not in cache:
                # Tensors' (index, name: shape):
                #   0 q_latent:   (H, D, S_q, B)
                #   1 q_rope:     (H, R, S_q, B)
                #   2 c_latent:   (page_size, D, num_pages)
                #   3 c_rope:     (page_size, R, num_pages)
                #   4 page_table: (max_blocks_per_sequence, B)
                #   5 cache_seqs: (B,)
                #   6 o:          (H, D, S_q, B)
                #   7 workspace:     (workspace_size,)
                #   8 softmax_stats: (B * S_q, H, 2), optional

                # cache_seqs (index 5) is the single free dynamic batch dim;
                # every other batch-carrying dim is tied to it by a constraint.
                batch_dims = ((0, 3), (1, 3), (4, 1), (5, 0), (6, 3))
                free = 5  # cache_seqs
                batch_constraints = tuple(
                    ConstraintSpec(
                        i, d, lambda shapes, _free=free: shapes[_free][0])
                    for (i, d) in batch_dims if i != free)

                # page_table dim0 (max_blocks) is a static config quantity, not
                # per-request: kept at its real size for profiling but excluded
                # from the cache key (constraint dims are set to -1 in the key),
                # since it tracks max_seq_len rather than anything this op sees.
                # It is a small int32 max_blocks x B tensor, so rebuilding it
                # for profiling is cheap.
                #
                # workspace (index 7) is a fixed-size slice whose size depends only
                # on (H, seq_len_q, kv_lora_rank, max_num_requests) -- all constant
                # for a given runner -- so the size is stable in the cache key.
                static_size_dims = ((4, 0), )
                static_constraints = tuple(
                    ConstraintSpec(
                        i, d, lambda shapes, _i=i, _d=d: shapes[_i][_d])
                    for (i, d) in static_size_dims)

                stats_constraints = ()
                if self.emit_softmax_stats:
                    stats_constraints = (ConstraintSpec(
                        8,
                        0,
                        lambda shapes: shapes[5][0] * self.seq_len_q,
                    ), )

                # The batch search space, fixed up-front by max_batch_size
                # when the engine max is known.
                batch_buckets = (get_last_power_of_2_num_tokens_buckets(
                    self.max_batch_size) if self.max_batch_size > 0 else
                                 get_last_power_of_2_num_tokens_buckets)
                cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        free,
                        0,
                        batch_buckets,
                        last_positive_power_of_2,
                    ), ),
                    constraint_specs=(batch_constraints + static_constraints +
                                      stats_constraints),
                    inputs_pre_hook=self._tuning_inputs_pre_hook,
                )
            return cache[key]

        def default_tactic(
            self,
            batch_size: int,
        ) -> Tuple[Tuple[int, int], Tuple[int, int], int, bool]:
            """Fallback 4-tuple tactic ``(mma_qk, mma_pv, split_kv,
            is_persistent)`` for when the AutoTuner cache is not warmed and
            ``choose_one`` returns its ``-1`` sentinel.

            ``batch_size`` is rounded down to its tuning bucket
            (``last_positive_power_of_2`` -- the same mapping the tuning
            config uses) before deriving ``split_kv``: tuning profiles (and
            therefore ``cute.compile``s) exactly the bucket-derived
            ``split_kv`` variants, so a bucket-aligned fallback reuses an
            already-compiled kernel where one exists instead of JIT-compiling
            a fresh raw-batch ``split_kv`` variant in the serving loop. The
            ``is_persistent`` choice is unaffected by the rounding (its
            threshold is a power of two, so rounding down to a power of two
            never crosses it), and both candidates are compiled during tuning
            anyway."""
            mma_qk_tiler_mn = (128, 128)
            mma_pv_tiler_mn = (128, 256)
            max_active_blocks = self._get_max_active_blocks()
            bucketed_batch_size = last_positive_power_of_2(batch_size)
            split_kv = self.get_default_split_kv(bucketed_batch_size,
                                                 self.seq_len_q,
                                                 max_active_blocks)
            is_persistent = self.get_default_is_persistent(bucketed_batch_size)
            return (mma_qk_tiler_mn, mma_pv_tiler_mn, split_kv, is_persistent)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
            **kwargs,
        ) -> torch.Tensor:
            """Run the CuTe DSL MLA decode kernel.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0] (q_latent): Query latent tensor of shape
                        (H, D, S_q, B).
                    inputs[1] (q_rope): Query RoPE tensor of shape (H, R, S_q, B).
                    inputs[2] (c_latent): Paged latent-cache tensor of shape
                        (page_size, D, num_pages).
                    inputs[3] (c_rope): Paged RoPE-cache tensor of shape
                        (page_size, R, num_pages).
                    inputs[4] (page_table): Page table tensor of shape
                        (max_blocks_per_sequence, B), dtype: int32.
                    inputs[5] (cache_seqs): Cache sequence lengths tensor of
                        shape (B), dtype: int32.
                    inputs[6] (o): Output tensor of shape (H, D, S_q, B).
                    inputs[7] (workspace): Contiguous raw workspace with at least
                        the workspace_size returned by get_workspace_layout.
                    inputs[8] (softmax_stats): Optional contiguous float32 tensor
                        of shape (B * S_q, H, 2). The kernel writes an equivalent
                        softmax (max, sum) pair for Helix reduction.
                tactic: Tuple containing (mma_qk_tiler_mn, mma_pv_tiler_mn,
                    split_kv, is_persistent).
                **kwargs: Optional softmax_scale and output_scale values.

            Returns:
                torch.Tensor: Output tensor of shape (H, D, S_q, B). The LSE
                    tensor of shape (H, S_q, B) remains in the workspace.
            """
            (q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, o,
             workspace, softmax_stats) = inputs
            softmax_scale = float(kwargs.get("softmax_scale", 1.0))
            output_scale = float(kwargs.get("output_scale", 1.0))

            if not (isinstance(tactic, tuple) and len(tactic) == 4):
                raise RuntimeError(
                    "CuteDSLNVMlaDecodeBlackwellRunner.forward expected a 4-tuple "
                    "tactic (mma_qk, mma_pv, split_kv, is_persistent), got "
                    f"{tactic!r}.")
            mma_qk_tiler_mn, mma_pv_tiler_mn, split_kv, is_persistent = tactic
            split_kv = int(split_kv)
            is_persistent = bool(is_persistent)
            mma_qk_tiler_mn = tuple(mma_qk_tiler_mn)
            mma_pv_tiler_mn = tuple(mma_pv_tiler_mn)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if o.dtype == torch.float16:
                out_dtype = cutlass.Float16
            elif o.dtype == torch.bfloat16:
                out_dtype = cutlass.BFloat16
            else:
                out_dtype = self.in_dtype

            seq_len_q = self.seq_len_q

            # workspace = lse + split_kv_workspace
            batch_size = cache_seqs.shape[0]
            d_latent = q_latent.shape[1]
            softmax_stats_kernel = None
            if softmax_stats is not None:
                expected_shape = (batch_size * seq_len_q, self.num_heads, 2)
                if (softmax_stats.shape != expected_shape
                        or softmax_stats.dtype != torch.float32
                        or softmax_stats.device != o.device
                        or not softmax_stats.is_contiguous()):
                    raise RuntimeError(
                        "CuteDSLNVMlaDecodeBlackwellRunner requires contiguous "
                        "float32 softmax_stats on the output device with shape "
                        f"{expected_shape}, got shape={tuple(softmax_stats.shape)}, "
                        f"dtype={softmax_stats.dtype}, device={softmax_stats.device}, "
                        f"contiguous={softmax_stats.is_contiguous()}.")
                softmax_stats_kernel = softmax_stats.view(
                    batch_size, seq_len_q, self.num_heads,
                    2).permute(2, 1, 0, 3)
            max_batch_size = max(batch_size, self.max_batch_size)
            (lse_offset, lse_size, split_kv_offset, split_kv_size,
             required_workspace_size) = self.get_workspace_layout(
                 self.num_heads, seq_len_q, d_latent, max_batch_size,
                 cutlass.Float32)

            if not workspace.is_contiguous():
                raise RuntimeError(
                    "CuteDSLNVMlaDecodeBlackwellRunner requires a contiguous "
                    "workspace.")
            workspace_bytes = workspace.view(torch.uint8).reshape(-1)
            if workspace_bytes.numel() < required_workspace_size:
                raise RuntimeError(
                    "CuteDSLNVMlaDecodeBlackwellRunner workspace is too small: "
                    f"got {workspace_bytes.numel()} bytes, require "
                    f"{required_workspace_size} bytes for "
                    f"batch_size={batch_size}, max_batch_size={max_batch_size}."
                )

            lse = workspace_bytes[lse_offset:lse_offset + lse_size].view(
                torch.float32).view(max_batch_size, seq_len_q,
                                    self.num_heads)[:batch_size].permute(
                                        2, 1, 0)
            # Kernel split-KV intermediates start AFTER the reserved LSE region.
            split_workspace = workspace_bytes[split_kv_offset:split_kv_offset +
                                              split_kv_size]

            cache_key = self.unique_id() + (
                out_dtype,
                mma_qk_tiler_mn,
                mma_pv_tiler_mn,
                split_kv,
                is_persistent,
            )
            if cache_key not in CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache:
                # A compile outside the tuning window stalls the serving loop
                # for seconds -- always log enough to identify the variant.
                logger.info(
                    f"CuteDSL MLA decode: compiling kernel variant {cache_key} "
                    f"B={cache_seqs.shape[0]} "
                    f"tuning={AutoTuner.get().is_tuning_mode} "
                    f"capturing={torch.cuda.is_current_stream_capturing()}")
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    self._CLUSTER_SHAPE_MNK[0] * self._CLUSTER_SHAPE_MNK[1] *
                    self._CLUSTER_SHAPE_MNK[2])

                # Fold seq_len_q into the head dimension when the head count
                # alone does not fill the MMA M tile (num_heads < M) and there
                # is more than one query token (MTP / spec-decode).
                fold_sq = (self.num_heads < mma_qk_tiler_mn[0]
                           and seq_len_q > 1)

                mla = self.kernel_class(
                    cutlass.Float32,  # acc_dtype
                    cutlass.Float32,  # lse_dtype
                    mma_qk_tiler_mn,
                    mma_pv_tiler_mn,
                    max_active_clusters,
                    self.page_size,
                    self._SKIP_CORRECTION_THRESHOLD,
                    is_persistent,
                    self._IS_VAR_SEQ,
                    self._IS_VAR_SPLIT_KV,
                    num_heads=self.num_heads,
                    seq_len_q=seq_len_q,
                    fold_sq=fold_sq,
                    emit_softmax_stats=self.emit_softmax_stats,
                )
                q_latent_ct = cute.runtime.from_dlpack(
                    q_latent,
                    assumed_align=16).mark_layout_dynamic(leading_dim=1)
                q_rope_ct = cute.runtime.from_dlpack(
                    q_rope, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                c_latent_ct = cute.runtime.from_dlpack(
                    c_latent,
                    assumed_align=16).mark_layout_dynamic(leading_dim=1)
                c_rope_ct = cute.runtime.from_dlpack(
                    c_rope, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                page_table_ct = cute.runtime.from_dlpack(
                    page_table,
                    assumed_align=16).mark_layout_dynamic(leading_dim=0)
                o_ct = cute.runtime.from_dlpack(
                    o, assumed_align=16).mark_layout_dynamic(
                        leading_dim=1).mark_compact_shape_dynamic(
                            mode=1,
                            stride_order=(3, 2, 0, 1),
                            divisibility=(128 // out_dtype.width))
                lse_ct = cute.runtime.from_dlpack(
                    lse, assumed_align=16).mark_layout_dynamic(leading_dim=0)
                softmax_stats_ct = (cute.runtime.from_dlpack(
                    softmax_stats_kernel, assumed_align=16).mark_layout_dynamic(
                        leading_dim=3) if softmax_stats_kernel is not None else
                                    None)
                use_workspace = split_kv > 1 and split_workspace.numel() > 0
                workspace_ct = (cute.runtime.from_dlpack(
                    split_workspace, assumed_align=32).mark_layout_dynamic()
                                if use_workspace else None)
                cache_seqs_ct = cute.runtime.from_dlpack(
                    cache_seqs, assumed_align=16).mark_layout_dynamic()
                # Variable split-KV (block_split_kvs) is not used on this path:
                block_split_kvs_ct = None

                compile_args = [
                    q_latent_ct,
                    q_rope_ct,
                    c_latent_ct,
                    c_rope_ct,
                    page_table_ct,
                    o_ct,
                    lse_ct,
                ]
                compile_target = mla
                if self.emit_softmax_stats:
                    compile_target = mla.run_with_softmax_stats
                    compile_args.append(softmax_stats_ct)
                compile_args.extend([
                    workspace_ct,
                    split_kv,
                    cache_seqs_ct,
                    block_split_kvs_ct,
                    cutlass.Float32(softmax_scale),
                    cutlass.Float32(output_scale),
                    stream,
                ])
                CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache[cache_key] = (
                    cute.compile(
                        compile_target,
                        *compile_args,
                        options="--opt-level 2",
                    ))

            compiled_mla = CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache[
                cache_key]
            runtime_args = [
                q_latent,
                q_rope,
                c_latent,
                c_rope,
                page_table,
                o,
                lse,
            ]
            if self.emit_softmax_stats:
                runtime_args.append(softmax_stats_kernel)
            runtime_args.extend([
                split_workspace if
                (split_kv > 1 and split_workspace.numel() > 0) else None,
                split_kv,
                cache_seqs,
                None,  # block_split_kvs: var-split path unused (is_var_split_kv False)
                softmax_scale,
                output_scale,
                stream,
            ])
            compiled_mla(*runtime_args)
            return o

    @torch.library.custom_op(
        "trtllm::cute_dsl_mla_decode_fp8_blackwell",
        mutates_args=("o", "workspace", "softmax_stats"),
        device_types="cuda",
    )
    def cute_dsl_mla_decode_fp8_blackwell(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: float,
        output_scale: float,
        # Keep the last two arguments required in the custom-op schema. PyTorch
        # elides trailing default-valued arguments before its mutation fallback,
        # while mutates_args retains their positional indices.
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
    ) -> None:
        """CuTe DSL FP8 MLA decode (Blackwell SM100/SM103).
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"trtllm::cute_dsl_mla_decode_fp8_blackwell requires SM 100 or "
                f"SM 103, got SM {sm_version}")

        # split_kv and is_persistent are chosen per shape by the runner's
        # AutoTuner (the 3rd/4th tactic elements), NOT at the op boundary.
        runner = CuteDSLNVMlaDecodeBlackwellRunner(
            in_dtype=cutlass.Float8E4M3FN,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            page_size=page_size,
            max_batch_size=max_batch_size,
            emit_softmax_stats=softmax_stats is not None,
        )
        inputs = [
            q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, o,
            workspace, softmax_stats
        ]
        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_mla_decode_fp8_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        # ``forward`` requires a 4-tuple tactic; if the tuner returned its -1
        # fallback (cache not warmed), supply the default 4-tuple so split_kv +
        # is_persistent still come from a length-4 tactic.
        if not (isinstance(best_tactic, tuple) and len(best_tactic) == 4):
            best_tactic = runner.default_tactic(int(q_latent.shape[-1]))
        runner(
            inputs,
            tactic=best_tactic,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
        )

    @torch.library.register_fake("trtllm::cute_dsl_mla_decode_fp8_blackwell")
    def _(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: float,
        output_scale: float,
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
    ) -> None:
        return None

    @torch.library.custom_op(
        "trtllm::cute_dsl_mla_decode_fp16_blackwell",
        mutates_args=("o", "workspace", "softmax_stats"),
        device_types="cuda",
    )
    def cute_dsl_mla_decode_fp16_blackwell(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: float,
        output_scale: float,
        # See the FP8 op above: these must remain required schema arguments.
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
    ) -> None:
        """CuTe DSL FP16/BF16 MLA decode (Blackwell SM100/SM103).
        """
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(
                f"trtllm::cute_dsl_mla_decode_fp16_blackwell requires SM 100 "
                f"or SM 103, got SM {sm_version}")

        if q_latent.dtype == torch.float16:
            in_dtype = cutlass.Float16
        elif q_latent.dtype == torch.bfloat16:
            in_dtype = cutlass.BFloat16
        else:
            raise ValueError(
                "trtllm::cute_dsl_mla_decode_fp16_blackwell supports "
                "torch.float16 or torch.bfloat16 inputs, got "
                f"{q_latent.dtype}")
        if not (q_rope.dtype == c_latent.dtype == c_rope.dtype == o.dtype ==
                q_latent.dtype):
            raise ValueError(
                "trtllm::cute_dsl_mla_decode_fp16_blackwell requires q, KV, "
                f"and output dtypes to match; got q_latent={q_latent.dtype}, "
                f"q_rope={q_rope.dtype}, c_latent={c_latent.dtype}, "
                f"c_rope={c_rope.dtype}, o={o.dtype}")

        # split_kv / is_persistent are chosen per shape by the runner's
        # AutoTuner (3rd/4th tactic elements), not at the op boundary.
        runner = CuteDSLNVMlaDecodeBlackwellRunner(
            in_dtype=in_dtype,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            page_size=page_size,
            max_batch_size=max_batch_size,
            emit_softmax_stats=softmax_stats is not None,
        )
        inputs = [
            q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, o,
            workspace, softmax_stats
        ]
        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_mla_decode_fp16_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        # ``forward`` requires a 4-tuple tactic; if the tuner returned its -1
        # fallback (cache not warmed), supply the default 4-tuple so split_kv +
        # is_persistent still come from a length-4 tactic.
        if not (isinstance(best_tactic, tuple) and len(best_tactic) == 4):
            best_tactic = runner.default_tactic(int(q_latent.shape[-1]))
        runner(
            inputs,
            tactic=best_tactic,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
        )

    @torch.library.register_fake("trtllm::cute_dsl_mla_decode_fp16_blackwell")
    def _(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: float,
        output_scale: float,
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
    ) -> None:
        return None

    # ============================================================================
    # Rubin (SM107) Support
    # ============================================================================
    # The following code provides CuTe DSL GEMM support for Rubin GPUs.
    # Requires Rubin support in the public nvidia-cutlass-dsl package.

    if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        # Rubin (SM107) MOE Grouped GEMM Support
        # ====================================================================
        # The following code provides CuteDSL NVFP4 grouped GEMM kernels for
        # Mixture-of-Experts (MoE) on Rubin GPUs (SM107).
        # Two fused kernels are provided:
        #   1. Gather + Grouped GEMM + activation fusion (FC1 layer)
        #   2. Grouped GEMM + Finalize (scatter-add) fusion (FC2 layer)

        from ..cute_dsl_kernels.rubin.moe.rubin_contiguous_gather_grouped_blockscaled_gemm_act_fusion import \
            Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel

        class Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner(
                TunableRunner):
            """Rubin runner for gather + grouped GEMM + activation fusion.

            SM107 counterpart to Blackwell's
            ``Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner``.
            Supports SwiGLU and Relu2.
            Key differences from Blackwell:
            - Uses LDGSTS (cp.async) for A/SFA loading instead of TMA
            - Supports B-reuse pattern (mma_tiler_m = 2 * mma_inst_shape_m)
            - Takes mma_inst_shape and mma_tiler as 3-tuples (not 2-tuples)
            """
            kernel_class = Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel
            kernel_cache = dict()
            tuning_config_cache = dict()

            def __init__(
                    self,
                    num_experts: int,
                    top_k: int,
                    num_local_experts: int,
                    local_expert_offset: int,
                    tile_size: int,
                    scaling_vector_size: int = 16,
                    activation_type: ActivationType = ActivationType.Swiglu):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.num_local_experts = num_local_experts
                self.local_expert_offset = local_expert_offset
                self.tile_size = tile_size
                self.scaling_vector_size = scaling_vector_size
                self.activation_type = ActivationType(int(activation_type))
                if self.activation_type not in (ActivationType.Swiglu,
                                                ActivationType.Relu2):
                    raise ValueError(
                        f"Rubin NVFP4 CuteDSL FC1 does not support "
                        f"{self.activation_type.name}")
                self.is_gated = is_gated_activation(self.activation_type)

                if (sm_version := get_sm_version()) != 107:
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports SM 107 (Rubin) only, but got SM {sm_version}"
                    )

                if self.tile_size not in (128, 256, 512):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports tile_size 128, 256 and 512 only, but got {self.tile_size}"
                    )

            def unique_id(self):
                return (
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    self.tile_size,
                    self.scaling_vector_size,
                    int(self.activation_type),
                )

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
                **kwargs,
            ) -> List[Tuple[int, int]]:
                a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, *_ = inputs
                # m is the permuted size from permuted_idx_to_expanded_idx, not from a
                m = permuted_idx_to_expanded_idx.size(0)
                k = a.size(1) * 2
                l, n = b.size(0), b.size(1)  # noqa: E741

                # Rubin 4xFP4 tile sizes:
                # Without B-reuse: mma_tiler_m == mma_inst_shape_m
                #   - (128, 128): 1CTA
                #   - (256, 256): 2CTA
                # With B-reuse: mma_tiler_m == 2 * mma_inst_shape_m
                #   - (256, 128): 1CTA, B-reuse
                #   - (512, 256): 2CTA, B-reuse
                # Fixed K dimensions for FP4: mma_tiler_k=256, mma_inst_k=128
                mma_tiler_k = 256
                mma_inst_k = 128

                # (mma_tiler_m, mma_inst_m) candidates
                mma_m_candidates = [
                    (128, 128),  # no B-reuse, 1CTA
                    (256, 256),  # no B-reuse, 2CTA
                    (256, 128),  # B-reuse, 1CTA
                    (512, 256),  # B-reuse, 2CTA
                ]
                # N dimension candidates (must match between tiler and inst)
                mma_n_candidates = [128, 256]
                # cluster_M is pinned to the MMA CTA-group size by the
                # cta_group_m filter below (1 for 1-CTA, 2 for 2-CTA).
                # cluster_N>1 multicasts A across N-CTAs (one GMEM read
                # broadcast to the cluster); validated for cluster_M==atom_m.
                cluster_shape_mn_candidates = [(1, 1), (1, 2), (1, 4), (2, 1),
                                               (2, 2), (2, 4)]
                raster_along_m_candidates = [False]
                # A-load path: "cpasync" (per-thread LDGSTS gather) vs "tma"
                # (TMA tile::gather4). Autotuned per shape.
                a_path_candidates = ["cpasync", "tma"]

                valid_tactics = []
                for (mma_tiler_m, mma_inst_m), mma_n, cluster_shape_mn, \
                        raster_along_m, a_path in (
                         itertools.product(mma_m_candidates, mma_n_candidates,
                                           cluster_shape_mn_candidates,
                                           raster_along_m_candidates,
                                           a_path_candidates)):

                    # The kernel requires each cluster to cover exactly
                    # one routing tile along M:
                    #   - 1-CTA (mma_inst_m=128): cluster_m must be 1
                    #   - 2-CTA (mma_inst_m=256): cluster_m must be 2
                    # and the CTA-pair's M extent (mma_tiler_m) must equal
                    # the routing tile_size.  Any other combination
                    # (mma_tiler_m != tile_size, or cluster_m spanning
                    # multiple independent M-tiles) produces ~40% element
                    # mismatches empirically.
                    cta_group_m = 2 if mma_inst_m == 256 else 1
                    if cluster_shape_mn[0] != cta_group_m:
                        continue
                    if mma_tiler_m != self.tile_size:
                        continue
                    # The Rubin epilogue stores full output and SFC subtiles
                    # only, so partial GEMM N tiles are not valid.
                    if n % mma_n != 0:
                        continue

                    # cluster_N>1 multicasts A across cluster-N CTAs, each
                    # computing a DISTINCT N-tile. This is only correct when the
                    # N-tiles (n // mma_n) divide evenly across the cluster;
                    # otherwise the multicast misaligns and the output is wrong.
                    # (The autotuner picks by latency with no correctness check,
                    # so an un-pruned bad multicast tactic could be selected in
                    # production — must gate it here.)
                    if cluster_shape_mn[1] > 1 and (
                            n % (mma_n * cluster_shape_mn[1]) != 0):
                        continue

                    # TMA gather4 is stable for per-CTA/2CTA A loads, but the
                    # multicast variant is not reliable on Rubin: cluster_N>1
                    # requires A multicast across N-CTAs. Keep cluster_N
                    # tactics available through the cpasync A path only.
                    if a_path == "tma" and cluster_shape_mn[1] > 1:
                        continue

                    mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
                    mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)

                    if self.__class__.kernel_class.can_implement(
                            a_dtype=cutlass.Float4E2M1FN,
                            b_dtype=cutlass.Float4E2M1FN,
                            sf_dtype=cutlass.Float8E4M3FN,
                            sf_vec_size=self.scaling_vector_size,
                            c_dtype=cutlass.Float4E2M1FN,
                            mma_inst_shape=mma_inst_shape,
                            mma_tiler=mma_tiler,
                            cluster_shape_mn=cluster_shape_mn,
                            m=m,
                            n=n,
                            k=k,
                            l=l,
                            a_major="k",
                            b_major="k",
                            c_major="n",
                    ):
                        valid_tactics.append(
                            (mma_tiler, mma_inst_shape, cluster_shape_mn,
                             raster_along_m, a_path))

                logger.debug(
                    f"CuteDSL Rubin GatherGroupedGemmSwiglu: Found {len(valid_tactics)} valid tactics "
                    f"for M={m}, N={n}, K={k}, L={l}")
                return valid_tactics

            def get_tuning_config(self) -> TuningConfig:
                key = self.unique_id()
                if key not in self.__class__.tuning_config_cache:
                    helper = GatherGroupedGemmInputsHelper(
                        self.num_experts, self.top_k, self.num_local_experts,
                        self.local_expert_offset, self.tile_size)
                    self.__class__.tuning_config_cache[key] = TuningConfig(
                        # Use permuted_idx_to_expanded_idx (IDX_SHAPE_INFER) for tuning
                        dynamic_tensor_specs=(DynamicTensorSpec(
                            GatherGroupedGemmInputsHelper.IDX_SHAPE_INFER, 0,
                            helper.gen_tuning_buckets,
                            helper.map_to_tuning_buckets), ),
                        constraint_specs=(
                            ConstraintSpec(0, 0, helper.infer_shape_num_tokens),
                            ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                            ConstraintSpec(5, 0,
                                           helper.infer_shape_max_num_tiles),
                            ConstraintSpec(6, 0,
                                           helper.infer_shape_max_num_tiles),
                        ),
                        inputs_pre_hook=helper.inputs_pre_hook,
                    )
                return self.__class__.tuning_config_cache[key]

            def forward(self, inputs: List[torch.Tensor],
                        tactic: Optional[tuple], **kwargs) -> torch.Tensor:
                a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf, output_tensor, output_sf_tensor = inputs
                # Verify permuted_idx_to_expanded_idx index matches the class constant
                assert inputs[
                    GatherGroupedGemmInputsHelper.
                    IDX_PERMUTED_IDX_TO_EXPANDED_IDX] is permuted_idx_to_expanded_idx
                assert a.dtype == torch.float4_e2m1fn_x2
                assert a.dim() == 2
                assert b.dtype == torch.float4_e2m1fn_x2
                assert b.dim() == 3
                assert a_sf.dtype == torch.uint8
                assert a_sf.dim() == 2
                assert b_sf.dtype == torch.uint8
                assert b_sf.dim() == 3
                assert alpha.dtype == torch.float32
                assert alpha.dim() == 1

                # a.size(0) is orig_m (original input size before gather)
                # permuted_idx_to_expanded_idx.size(0) is m (permuted size after gather)
                orig_m, k = a.size(0), a.size(1) * 2
                m = permuted_idx_to_expanded_idx.size(0)
                l, n = b.size(0), b.size(1)  # noqa: E741
                scale_k = k // self.scaling_vector_size
                interm_size = n // 2 if self.is_gated else n
                assert m % self.tile_size == 0
                assert k % (self.scaling_vector_size * 4) == 0
                n_alignment = self.scaling_vector_size * 4 * (2 if self.is_gated
                                                              else 1)
                assert n % n_alignment == 0
                assert b.size(2) * 2 == k
                assert a_sf.size(0) == orig_m
                assert a_sf.size(1) == scale_k
                assert b_sf.size(0) == l
                assert b_sf.size(1) == n
                assert b_sf.size(2) == scale_k
                assert alpha.size(0) == l

                num_tiles = m // self.tile_size
                assert tile_idx_to_group_idx.dtype == torch.int32
                assert tile_idx_to_group_idx.size() == (num_tiles, )
                assert tile_idx_to_mn_limit.dtype == torch.int32
                assert tile_idx_to_mn_limit.size() == (num_tiles, )
                assert permuted_idx_to_expanded_idx.dtype == torch.int32
                assert permuted_idx_to_expanded_idx.size() == (m, )
                assert num_non_exiting_tiles.dtype == torch.int32
                assert num_non_exiting_tiles.numel() == 1
                assert global_sf.dtype == torch.float32
                assert global_sf.numel() == 1

                partition_id = kwargs.get("partition_id", -1)
                locality_domain_half_gemm = output_tensor is not None or output_sf_tensor is not None
                if locality_domain_half_gemm:
                    if not self.is_gated:
                        raise ValueError(
                            "Rubin locality domain half-GEMM currently supports SwiGLU only"
                        )
                    if partition_id < 0 or partition_id >= 2:
                        raise ValueError(
                            "partition_id must be 0 or 1 when output tensors are provided."
                        )
                    assert output_tensor is not None and output_sf_tensor is not None
                    assert output_tensor.dim() == 2
                    assert output_tensor.dtype == a.dtype
                    assert output_tensor.shape[0] == m and output_tensor.shape[
                        1] == interm_size // 2 * 2, f"[locality domain] output_tensor.shape={output_tensor.shape}, m={m}, n={n}"
                    assert output_sf_tensor.dim() == 1
                    sf_locality_domain_total_size = m * interm_size // self.scaling_vector_size
                    assert output_sf_tensor.shape[
                        0] == sf_locality_domain_total_size * 2
                    # c: point into shared buffer at column offset
                    # (kernel uses full stride via locality_domain_half_gemm + full_c_shape)
                    c_byte_offset = partition_id * interm_size // 2  # fp4x2 cols
                    c = output_tensor.view(torch.uint8)[:, c_byte_offset:].view(
                        torch.float4_e2m1fn_x2)
                    # Keep the SFC pointer at the start of the full shared
                    # buffer. The kernel applies the partition's N-tile offset
                    # in full-layout coordinates.
                    assert interm_size % 64 == 0
                    c_sf = output_sf_tensor
                    c_sf_n_tile_offset_val = cutlass.Int64(partition_id *
                                                           interm_size // 64)
                else:
                    c = torch.empty(m,
                                    interm_size // 2,
                                    dtype=a.dtype,
                                    device=a.device)
                    c_sf = torch.empty(m * interm_size //
                                       self.scaling_vector_size,
                                       dtype=a_sf.dtype,
                                       device=a_sf.device)
                    c_sf_n_tile_offset_val = cutlass.Int64(0)

                a_ptr = make_ptr(cutlass.Float4E2M1FN,
                                 a.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=32)
                b_ptr = make_ptr(cutlass.Float4E2M1FN,
                                 b.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=32)
                a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                    a_sf.data_ptr(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16)
                b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                    b_sf.data_ptr(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16)
                alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                     cute.AddressSpace.gmem)
                tile_idx_to_group_idx_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                tile_idx_to_mn_limit_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_mn_limit.data_ptr(),
                    cute.AddressSpace.gmem)
                permuted_idx_to_expanded_idx_ptr = make_ptr(
                    cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                num_non_exiting_tiles_ptr = make_ptr(
                    cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                    cute.AddressSpace.gmem)
                global_sf_ptr = make_ptr(cutlass.Float32, global_sf.data_ptr(),
                                         cute.AddressSpace.gmem)
                c_ptr = make_ptr(cutlass.Float4E2M1FN,
                                 c.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=32)
                c_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                    c_sf.data_ptr(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                if isinstance(tactic, tuple):
                    mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m, a_path = tactic
                else:
                    # Default tactic for Rubin
                    mma_tiler, mma_inst_shape, cluster_shape_mn = \
                        _get_sm107_nvfp4_default_mma_config(self.tile_size)
                    raster_along_m = False
                    a_path = "cpasync"
                assert mma_tiler[
                    0] >= self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

                # c_stride_m for locality domain strided output (0 = default contiguous)
                c_stride_m_val = cutlass.Int64(
                    interm_size *
                    2) if locality_domain_half_gemm else cutlass.Int64(0)

                max_active_clusters = get_max_activate_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])
                cache_key = (self.scaling_vector_size, self.tile_size,
                             self.top_k, mma_tiler, mma_inst_shape,
                             cluster_shape_mn, raster_along_m,
                             locality_domain_half_gemm, a_path,
                             int(self.activation_type), max_active_clusters)
                if cache_key not in self.__class__.kernel_cache:
                    gemm = self.__class__.kernel_class(
                        sf_vec_size=self.scaling_vector_size,
                        mma_inst_shape=mma_inst_shape,
                        mma_tiler=mma_tiler,
                        cluster_shape_mn=cluster_shape_mn,
                        vectorized_f32=True,
                        topk=self.top_k,
                        raster_along_m=raster_along_m,
                        locality_domain_half_gemm=locality_domain_half_gemm,
                        a_path=a_path,
                        activation_type=self.activation_type,
                    )
                    compiled_gemm = cute.compile(
                        gemm.wrapper,
                        a_ptr,
                        b_ptr,
                        a_sf_ptr,
                        b_sf_ptr,
                        c_ptr,
                        c_sf_ptr,
                        alpha_ptr,
                        tile_idx_to_group_idx_ptr,
                        tile_idx_to_mn_limit_ptr,
                        permuted_idx_to_expanded_idx_ptr,
                        num_non_exiting_tiles_ptr,
                        global_sf_ptr,
                        orig_m,
                        m,
                        n,
                        k,
                        l,
                        tile_size=self.tile_size,
                        scaling_vector_size=self.scaling_vector_size,
                        max_active_clusters=max_active_clusters,
                        stream=stream,
                        c_stride_m=c_stride_m_val,
                        c_sf_n_tile_offset=c_sf_n_tile_offset_val,
                    )
                    self.__class__.kernel_cache[cache_key] = compiled_gemm
                else:
                    compiled_gemm = self.__class__.kernel_cache[cache_key]

                compiled_gemm(
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    c_sf_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    global_sf_ptr,
                    orig_m,
                    m,
                    n,
                    k,
                    l,
                    stream=stream,
                    c_stride_m=c_stride_m_val,
                    c_sf_n_tile_offset=c_sf_n_tile_offset_val,
                )
                return c, c_sf

        def _run_nvfp4_gather_grouped_gemm_act_fusion_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            global_sf: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: Optional[torch.Tensor],
            output_sf_tensor: Optional[torch.Tensor],
            scaling_vector_size: int,
            partition_id: int,
            activation_type: ActivationType,
            precomputed_tactic: Optional[str],
            tuner_key: str,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            tuner = AutoTuner.get()
            if output_tensor is not None or output_sf_tensor is not None:
                if output_tensor is None or output_sf_tensor is None:
                    raise ValueError(
                        "output_tensor and output_sf_tensor must be provided together."
                    )
                if partition_id < 0 or partition_id >= 2:
                    raise ValueError(
                        "partition_id must be 0 or 1 when output tensors are provided."
                    )
            elif partition_id != -1:
                raise ValueError(
                    "partition_id must be -1 when output tensors are not provided."
                )

            runner = Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner(
                num_experts,
                top_k,
                num_local_experts,
                local_expert_offset,
                tile_size,
                scaling_vector_size,
                activation_type=activation_type,
            )
            inputs = [
                input, weight, input_scale, weight_scale, alpha,
                tile_idx_to_group_idx, tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf,
                output_tensor, output_sf_tensor
            ]
            choose_one_kwargs = {}
            if output_tensor is not None:
                choose_one_kwargs["partition_id"] = partition_id

            if precomputed_tactic is None:
                _, best_tactic = tuner.choose_one(
                    tuner_key,
                    [runner],
                    runner.get_tuning_config(),
                    inputs,
                    **choose_one_kwargs,
                )
            else:
                best_tactic = ast.literal_eval(precomputed_tactic)
            output, output_sf = runner(inputs,
                                       tactic=best_tactic,
                                       partition_id=partition_id)
            if output_tensor is not None:
                return None, None
            return output, output_sf

        @torch.library.custom_op(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin",
            mutates_args=("output_tensor", "output_sf_tensor"),
            schema=
            "(Tensor input, Tensor weight, Tensor input_scale, Tensor weight_scale, Tensor alpha, "
            "Tensor tile_idx_to_group_idx, Tensor tile_idx_to_mn_limit, "
            "Tensor permuted_idx_to_expanded_idx, Tensor num_non_exiting_tiles, Tensor global_sf, "
            "SymInt num_experts, SymInt top_k, SymInt num_local_experts, "
            "SymInt local_expert_offset, SymInt tile_size, "
            "Tensor(a16!)? output_tensor, Tensor(a17!)? output_sf_tensor, "
            "SymInt scaling_vector_size=16, SymInt partition_id=-1, "
            f"SymInt activation_type={int(ActivationType.Swiglu)}, "
            "str? precomputed_tactic=None) -> (Tensor?, Tensor?)",
            device_types="cuda")
        def cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            global_sf: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: Optional[torch.Tensor],
            output_sf_tensor: Optional[torch.Tensor],
            scaling_vector_size: int = 16,
            partition_id: int = -1,
            activation_type: int = int(ActivationType.Swiglu),
            precomputed_tactic: Optional[str] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            return _run_nvfp4_gather_grouped_gemm_act_fusion_rubin(
                input, weight, input_scale, weight_scale, alpha,
                tile_idx_to_group_idx, tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf,
                num_experts, top_k, num_local_experts, local_expert_offset,
                tile_size, output_tensor, output_sf_tensor,
                scaling_vector_size, partition_id,
                ActivationType(activation_type), precomputed_tactic,
                "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin")

        @torch.library.register_fake(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            global_sf: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: Optional[torch.Tensor],
            output_sf_tensor: Optional[torch.Tensor],
            scaling_vector_size: int = 16,
            partition_id: int = -1,
            activation_type: int = int(ActivationType.Swiglu),
            precomputed_tactic: Optional[str] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            m = permuted_idx_to_expanded_idx.size(0)
            n = weight.size(1)
            is_gated = is_gated_activation(ActivationType(activation_type))
            interm_size = n // 2 if is_gated else n
            if output_tensor is not None or output_sf_tensor is not None:
                assert output_tensor is not None
                assert output_sf_tensor is not None
                return None, None
            output = torch.empty(m,
                                 interm_size // 2,
                                 dtype=input.dtype,
                                 device=input.device)
            output_scale = torch.empty(m * interm_size // scaling_vector_size,
                                       dtype=input_scale.dtype,
                                       device=input_scale.device)
            return output, output_scale

        @torch.library.custom_op(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin",
            mutates_args=("output_tensor", "output_sf_tensor"),
            schema=
            "(Tensor input, Tensor weight_0, Tensor weight_1, Tensor input_scale, "
            "Tensor weight_scale_0, Tensor weight_scale_1, Tensor alpha, "
            "Tensor tile_idx_to_group_idx, Tensor tile_idx_to_mn_limit, "
            "Tensor permuted_idx_to_expanded_idx, Tensor num_non_exiting_tiles, "
            "Tensor global_sf, SymInt num_experts, SymInt top_k, "
            "SymInt num_local_experts, SymInt local_expert_offset, "
            "SymInt tile_size, Tensor(a!) output_tensor, "
            "Tensor(b!) output_sf_tensor, SymInt scaling_vector_size=16, "
            f"SymInt activation_type={int(ActivationType.Swiglu)}) -> ()",
            device_types="cuda")
        def cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin(
                input: torch.Tensor,
                weight_0: torch.Tensor,
                weight_1: torch.Tensor,
                input_scale: torch.Tensor,
                weight_scale_0: torch.Tensor,
                weight_scale_1: torch.Tensor,
                alpha: torch.Tensor,
                tile_idx_to_group_idx: torch.Tensor,
                tile_idx_to_mn_limit: torch.Tensor,
                permuted_idx_to_expanded_idx: torch.Tensor,
                num_non_exiting_tiles: torch.Tensor,
                global_sf: torch.Tensor,
                num_experts: int,
                top_k: int,
                num_local_experts: int,
                local_expert_offset: int,
                tile_size: int,
                output_tensor: torch.Tensor,
                output_sf_tensor: torch.Tensor,
                scaling_vector_size: int = 16,
                activation_type: int = int(ActivationType.Swiglu),
        ) -> None:
            """Tune and launch both Rubin locality domain NVFP4 MoE FC1 partitions.

            The MoE outer runner invokes this op during preparation so locality domain
            resources, tactics, and kernels are ready before CUDA graph capture.
            Direct callers must likewise invoke it once before capture.
            """
            if weight_0.shape != weight_1.shape:
                raise ValueError(
                    "locality domain NVFP4 MoE FC1 weight shards must have identical "
                    f"shapes, got {tuple(weight_0.shape)} and "
                    f"{tuple(weight_1.shape)}.")
            if weight_0.dtype != weight_1.dtype:
                raise ValueError(
                    "locality domain NVFP4 MoE FC1 weight shards must have identical "
                    f"dtypes, got {weight_0.dtype} and {weight_1.dtype}.")
            if weight_scale_0.shape != weight_scale_1.shape:
                raise ValueError(
                    "locality domain NVFP4 MoE FC1 weight-scale shards must have "
                    f"identical shapes, got {tuple(weight_scale_0.shape)} and "
                    f"{tuple(weight_scale_1.shape)}.")
            if weight_scale_0.dtype != weight_scale_1.dtype:
                raise ValueError(
                    "locality domain NVFP4 MoE FC1 weight-scale shards must have "
                    f"identical dtypes, got {weight_scale_0.dtype} and "
                    f"{weight_scale_1.dtype}.")

            runtime = LocalityDomainRuntime(num_partitions=2)
            # Preserve the pre-refactor leaf namespace so persisted tactic
            # caches remain reusable after moving ownership into this op.
            tuner_key = (
                "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin")
            op_runner = (
                Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner(
                    num_experts,
                    top_k,
                    num_local_experts,
                    local_expert_offset,
                    tile_size,
                    scaling_vector_size,
                    activation_type=ActivationType(activation_type),
                ))
            inputs = [
                input,
                weight_0,
                input_scale,
                weight_scale_0,
                alpha,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                global_sf,
                output_tensor,
                output_sf_tensor,
            ]

            def launch_partition(
                partition_id: int,
                partition_inputs: List[torch.Tensor],
                tactic,
            ) -> None:
                weight = weight_0 if partition_id == 0 else weight_1
                weight_scale = (weight_scale_0
                                if partition_id == 0 else weight_scale_1)
                torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
                    input=partition_inputs[0],
                    weight=weight,
                    input_scale=partition_inputs[2],
                    weight_scale=weight_scale,
                    alpha=partition_inputs[4],
                    tile_idx_to_group_idx=partition_inputs[5],
                    tile_idx_to_mn_limit=partition_inputs[6],
                    permuted_idx_to_expanded_idx=partition_inputs[7],
                    num_non_exiting_tiles=partition_inputs[8],
                    global_sf=partition_inputs[9],
                    num_experts=num_experts,
                    top_k=top_k,
                    num_local_experts=num_local_experts,
                    local_expert_offset=local_expert_offset,
                    tile_size=tile_size,
                    output_tensor=partition_inputs[10],
                    output_sf_tensor=partition_inputs[11],
                    scaling_vector_size=scaling_vector_size,
                    partition_id=partition_id,
                    activation_type=activation_type,
                    precomputed_tactic=repr(tactic),
                )

            runner, best_tactic = tune_locality_domain_concurrent(
                tuner_key,
                op_runner,
                runtime,
                2,
                launch_partition,
                inputs,
                op_runner.get_tuning_config(),
            )
            runner(inputs, tactic=best_tactic)

        @torch.library.register_fake(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin"
        )
        def _(
                input: torch.Tensor,
                weight_0: torch.Tensor,
                weight_1: torch.Tensor,
                input_scale: torch.Tensor,
                weight_scale_0: torch.Tensor,
                weight_scale_1: torch.Tensor,
                alpha: torch.Tensor,
                tile_idx_to_group_idx: torch.Tensor,
                tile_idx_to_mn_limit: torch.Tensor,
                permuted_idx_to_expanded_idx: torch.Tensor,
                num_non_exiting_tiles: torch.Tensor,
                global_sf: torch.Tensor,
                num_experts: int,
                top_k: int,
                num_local_experts: int,
                local_expert_offset: int,
                tile_size: int,
                output_tensor: torch.Tensor,
                output_sf_tensor: torch.Tensor,
                scaling_vector_size: int = 16,
                activation_type: int = int(ActivationType.Swiglu),
        ) -> None:
            return None

        # ----------------------------------------------------------------
        # Rubin BF16/FP16 Gather + SwiGLU Fusion (FC1 layer)
        # ----------------------------------------------------------------
        from ..cute_dsl_kernels.rubin.moe.rubin_contiguous_gather_grouped_gemm_swiglu_fusion import \
            Sm107ContiguousGatherGroupedGemmSwigluFusionKernel

        class Sm107ContiguousGatherGroupedGemmSwigluFusionRunner(TunableRunner):
            """Rubin (SM107) runner for BF16/FP16 gather + grouped GEMM + SwiGLU fusion (FC1).

            Similar to the blockscaled runner but without scale factors.
            Uses MmaF16BF16Op (K=16) for BFloat16/Float16 inputs.
            """
            kernel_class = Sm107ContiguousGatherGroupedGemmSwigluFusionKernel
            kernel_cache = dict()
            tuning_config_cache = dict()

            def __init__(self,
                         num_experts: int,
                         top_k: int,
                         num_local_experts: int,
                         local_expert_offset: int,
                         tile_size: int,
                         input_dtype: Optional[torch.dtype] = None):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.num_local_experts = num_local_experts
                self.local_expert_offset = local_expert_offset
                self.tile_size = tile_size
                self.input_dtype = input_dtype

                if input_dtype is not None and input_dtype not in (
                        torch.bfloat16, torch.float16):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} requires BF16 or FP16 input, "
                        f"but got {input_dtype}")

                if (sm_version := get_sm_version()) != 107:
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports SM 107 (Rubin) only, but got SM {sm_version}"
                    )

                if self.tile_size not in (64, 128, 256):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports tile_size 64, 128 and 256 only, but got {self.tile_size}"
                    )

            def unique_id(self):
                return (
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    self.tile_size,
                    self.input_dtype,
                )

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
                **kwargs,
            ) -> List[Tuple[int, int]]:
                a, b, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, *_ = inputs
                m = permuted_idx_to_expanded_idx.size(0)
                k = a.size(1)
                l, n = b.size(0), b.size(1)  # noqa: E741

                ab_dtype = cutlass.BFloat16 if a.dtype == torch.bfloat16 else cutlass.Float16
                c_dtype = ab_dtype

                # BF16/FP16 tile sizes (no B-reuse):
                #   - (64, N): 1CTA
                #   - (128, N): 1CTA
                #   - (256, N): 2CTA
                # Fixed K dimensions: mma_tiler_k=64, mma_inst_k=16
                mma_tiler_k = 64
                mma_inst_k = 16

                mma_n_candidates = [128, 256]
                raster_along_m_candidates = [False]

                # BF16 (no B-reuse): CTA tile M must equal tile_size.
                # 2CTA (CtaGroup.TWO) is only valid for tile_size=256
                # (mma_m=256 triggers CtaGroup.TWO internally).
                # cluster_shape_mn is always (max(1, tile_size//128), 1).
                mma_m = self.tile_size
                cluster_shape_mn = (max(1, self.tile_size // 128), 1)

                valid_tactics = []
                for mma_n, raster_along_m in (itertools.product(
                        mma_n_candidates, raster_along_m_candidates)):

                    mma_tiler = (mma_m, mma_n, mma_tiler_k)
                    mma_inst_shape = (mma_m, mma_n, mma_inst_k)

                    if self.__class__.kernel_class.can_implement(
                            a_dtype=ab_dtype,
                            b_dtype=ab_dtype,
                            c_dtype=c_dtype,
                            mma_inst_shape=mma_inst_shape,
                            mma_tiler=mma_tiler,
                            cluster_shape_mn=cluster_shape_mn,
                            m=m,
                            n=n,
                            k=k,
                            l=l,
                            a_major="k",
                            b_major="k",
                            c_major="n",
                    ):
                        valid_tactics.append((mma_tiler, mma_inst_shape,
                                              cluster_shape_mn, raster_along_m))

                logger.debug(
                    f"CuteDSL Rubin BF16 GatherGroupedGemmSwiglu: Found {len(valid_tactics)} valid tactics "
                    f"for M={m}, N={n}, K={k}, L={l}")
                return valid_tactics

            # BF16 input layout (no scale factors):
            #   0: a, 1: b, 2: alpha,
            #   3: tile_idx_to_group_idx, 4: tile_idx_to_mn_limit,
            #   5: permuted_idx_to_expanded_idx, 6: num_non_exiting_tiles
            _BF16_IDX_PERMUTED = 5

            def get_tuning_config(self,
                                  has_output_tensor: bool = False
                                  ) -> TuningConfig:
                key = (*self.unique_id(), has_output_tensor)
                if key not in self.__class__.tuning_config_cache:
                    helper = GatherGroupedGemmInputsHelper(
                        self.num_experts, self.top_k, self.num_local_experts,
                        self.local_expert_offset, self.tile_size)
                    # BF16 has permuted_idx at index 5, not 7 (no scale
                    # factor inputs). Override IDX_SHAPE_INFER so that
                    # infer_shape_num_tokens / infer_shape_max_num_tiles
                    # read the correct tensor.
                    helper.IDX_SHAPE_INFER = self._BF16_IDX_PERMUTED
                    constraint_specs = [
                        ConstraintSpec(0, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(3, 0, helper.infer_shape_max_num_tiles),
                        ConstraintSpec(4, 0, helper.infer_shape_max_num_tiles),
                    ]
                    if has_output_tensor:
                        constraint_specs.append(
                            ConstraintSpec(
                                7, 0,
                                helper.infer_shape_max_num_permuted_tokens))
                    self.__class__.tuning_config_cache[key] = TuningConfig(
                        dynamic_tensor_specs=(DynamicTensorSpec(
                            self._BF16_IDX_PERMUTED, 0,
                            helper.gen_tuning_buckets,
                            helper.map_to_tuning_buckets), ),
                        constraint_specs=tuple(constraint_specs),
                        inputs_pre_hook=self._bf16_inputs_pre_hook,
                    )
                return self.__class__.tuning_config_cache[key]

            def _bf16_inputs_pre_hook(
                    self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
                """Pre-hook adapted for BF16 input layout (no scale factors)."""
                a, b, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, \
                    permuted_idx_to_expanded_idx, num_non_exiting_tiles, *maybe_output = inputs

                helper = GatherGroupedGemmInputsHelper(self.num_experts,
                                                       self.top_k,
                                                       self.num_local_experts,
                                                       self.local_expert_offset,
                                                       self.tile_size)

                max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)
                num_tokens = helper.infer_num_tokens(max_num_permuted_tokens)
                num_tokens_per_expert = helper.generate_num_tokens_per_expert(
                    num_tokens, approx_max_load=True)
                token_selected_experts = helper.generate_token_selected_experts(
                    num_tokens, num_tokens_per_expert)

                token_selected_experts = token_selected_experts.cuda()
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)

                (
                    new_tile_idx_to_group_idx,
                    new_tile_idx_to_mn_limit,
                    _,
                    new_permuted_idx_to_expanded_idx,
                    _,
                    new_num_non_exiting_tiles,
                ) = torch.ops.trtllm.moe_sort(
                    token_selected_experts=token_selected_experts,
                    token_final_scales=token_final_scales,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    local_expert_offset=self.local_expert_offset,
                    local_num_experts=self.num_local_experts,
                    tile_tokens_dim=self.tile_size,
                )

                updated_inputs = [
                    a,
                    b,
                    alpha,
                    new_tile_idx_to_group_idx,
                    new_tile_idx_to_mn_limit,
                    new_permuted_idx_to_expanded_idx,
                    new_num_non_exiting_tiles,
                ]
                if maybe_output:
                    updated_inputs.append(maybe_output[0])
                return updated_inputs

            def forward(self, inputs: List[torch.Tensor],
                        tactic: Optional[tuple], **kwargs) -> torch.Tensor:
                (a, b, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit,
                 permuted_idx_to_expanded_idx, num_non_exiting_tiles,
                 *maybe_output) = inputs
                assert inputs[
                    self._BF16_IDX_PERMUTED] is permuted_idx_to_expanded_idx
                assert a.dtype in (torch.bfloat16, torch.float16)
                assert a.dim() == 2
                assert b.dtype == a.dtype
                assert b.dim() == 3
                assert alpha.dtype == torch.float32
                assert alpha.dim() == 1

                ab_dtype = cutlass.BFloat16 if a.dtype == torch.bfloat16 else cutlass.Float16

                orig_m, k = a.size(0), a.size(1)
                m = permuted_idx_to_expanded_idx.size(0)
                l, n = b.size(0), b.size(1)  # noqa: E741
                interm_size = n // 2
                assert m % self.tile_size == 0
                assert b.size(2) == k
                assert alpha.size(0) == l

                num_tiles = m // self.tile_size
                assert tile_idx_to_group_idx.dtype == torch.int32
                assert tile_idx_to_group_idx.size() == (num_tiles, )
                assert tile_idx_to_mn_limit.dtype == torch.int32
                assert tile_idx_to_mn_limit.size() == (num_tiles, )
                assert permuted_idx_to_expanded_idx.dtype == torch.int32
                assert permuted_idx_to_expanded_idx.size() == (m, )
                assert num_non_exiting_tiles.dtype == torch.int32
                assert num_non_exiting_tiles.numel() == 1

                if maybe_output:
                    partition_id = kwargs.get("partition_id", -1)
                    if partition_id < 0 or partition_id >= 2:
                        raise ValueError(
                            "partition_id must be 0 or 1 when output_tensor is provided."
                        )
                    output_tensor = maybe_output[0]
                    assert output_tensor.dim() == 2
                    assert output_tensor.dtype == a.dtype
                    assert output_tensor.shape[0] == m
                    assert output_tensor.shape[1] == interm_size * 2
                    c = output_tensor[:, partition_id *
                                      interm_size:(partition_id + 1) *
                                      interm_size]
                    c_stride_m_val = cutlass.Int64(output_tensor.shape[1])
                    locality_domain_half_gemm = True
                else:
                    c = torch.empty(m,
                                    interm_size,
                                    dtype=a.dtype,
                                    device=a.device)
                    c_stride_m_val = cutlass.Int64(0)
                    locality_domain_half_gemm = False

                a_ptr = make_ptr(ab_dtype,
                                 a.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)
                b_ptr = make_ptr(ab_dtype,
                                 b.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)
                c_ptr = make_ptr(ab_dtype,
                                 c.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)
                alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                     cute.AddressSpace.gmem)
                tile_idx_to_group_idx_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                tile_idx_to_mn_limit_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_mn_limit.data_ptr(),
                    cute.AddressSpace.gmem)
                permuted_idx_to_expanded_idx_ptr = make_ptr(
                    cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                num_non_exiting_tiles_ptr = make_ptr(
                    cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                    cute.AddressSpace.gmem)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                if isinstance(tactic, tuple):
                    mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m = tactic
                else:
                    mma_tiler = (self.tile_size, 128, 64)
                    mma_inst_shape = (self.tile_size, 128, 16)
                    cluster_shape_mn = (max(1, self.tile_size // 128), 1)
                    raster_along_m = False

                max_active_clusters = get_max_activate_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])
                cache_key = (a.dtype, self.tile_size, self.top_k, mma_tiler,
                             mma_inst_shape, cluster_shape_mn, raster_along_m,
                             locality_domain_half_gemm, max_active_clusters)
                if cache_key not in self.__class__.kernel_cache:
                    gemm = self.__class__.kernel_class(
                        mma_inst_shape=mma_inst_shape,
                        mma_tiler=mma_tiler,
                        cluster_shape_mn=cluster_shape_mn,
                        vectorized_f32=True,
                        topk=self.top_k,
                        raster_along_m=raster_along_m,
                    )
                    compiled_gemm = cute.compile(
                        gemm.wrapper,
                        a_ptr,
                        b_ptr,
                        c_ptr,
                        alpha_ptr,
                        tile_idx_to_group_idx_ptr,
                        tile_idx_to_mn_limit_ptr,
                        permuted_idx_to_expanded_idx_ptr,
                        num_non_exiting_tiles_ptr,
                        orig_m,
                        m,
                        n,
                        k,
                        l,
                        tile_size=self.tile_size,
                        max_active_clusters=max_active_clusters,
                        stream=stream,
                        c_stride_m=c_stride_m_val,
                    )
                    self.__class__.kernel_cache[cache_key] = compiled_gemm
                else:
                    compiled_gemm = self.__class__.kernel_cache[cache_key]

                compiled_gemm(
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    orig_m,
                    m,
                    n,
                    k,
                    l,
                    stream=stream,
                    c_stride_m=c_stride_m_val,
                )
                return c

        @torch.library.custom_op(
            "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin",
            mutates_args=("output_tensor", ),
            schema="(Tensor input, Tensor weight, Tensor alpha, "
            "Tensor tile_idx_to_group_idx, Tensor tile_idx_to_mn_limit, "
            "Tensor permuted_idx_to_expanded_idx, Tensor num_non_exiting_tiles, "
            "SymInt num_experts, SymInt top_k, SymInt num_local_experts, "
            "SymInt local_expert_offset, SymInt tile_size, "
            "Tensor(a!)? output_tensor, SymInt partition_id, "
            "str? precomputed_tactic=None) -> Tensor?",
            device_types="cuda")
        def cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: Optional[torch.Tensor],
            partition_id: int,
            precomputed_tactic: Optional[str] = None,
        ) -> Optional[torch.Tensor]:
            tuner = AutoTuner.get()
            if output_tensor is not None:
                if partition_id < 0 or partition_id >= 2:
                    raise ValueError(
                        "partition_id must be 0 or 1 when output_tensor is provided."
                    )
            elif partition_id != -1:
                raise ValueError(
                    "partition_id must be -1 when output_tensor is not provided."
                )

            runner = Sm107ContiguousGatherGroupedGemmSwigluFusionRunner(
                num_experts,
                top_k,
                num_local_experts,
                local_expert_offset,
                tile_size,
                input_dtype=input.dtype)
            inputs = [
                input,
                weight,
                alpha,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
            ]
            if output_tensor is not None:
                inputs.append(output_tensor)

            choose_one_kwargs = {}
            if output_tensor is not None:
                choose_one_kwargs["partition_id"] = partition_id

            if precomputed_tactic is None:
                _, best_tactic = tuner.choose_one(
                    "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin",
                    [runner],
                    runner.get_tuning_config(output_tensor is not None),
                    inputs,
                    **choose_one_kwargs,
                )
            else:
                best_tactic = ast.literal_eval(precomputed_tactic)

            output = runner(inputs,
                            tactic=best_tactic,
                            partition_id=partition_id)
            if output_tensor is not None:
                return None
            return output

        @torch.library.register_fake(
            "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: Optional[torch.Tensor],
            partition_id: int,
            precomputed_tactic: Optional[str] = None,
        ) -> Optional[torch.Tensor]:
            m = permuted_idx_to_expanded_idx.size(0)
            n = weight.size(1)
            interm_size = n // 2
            if output_tensor is not None:
                return None
            return torch.empty(m,
                               interm_size,
                               dtype=input.dtype,
                               device=input.device)

        @torch.library.custom_op(
            "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin",
            mutates_args=("output_tensor", ),
            schema="(Tensor input, Tensor weight_0, Tensor weight_1, "
            "Tensor alpha, Tensor tile_idx_to_group_idx, "
            "Tensor tile_idx_to_mn_limit, "
            "Tensor permuted_idx_to_expanded_idx, "
            "Tensor num_non_exiting_tiles, SymInt num_experts, SymInt top_k, "
            "SymInt num_local_experts, SymInt local_expert_offset, "
            "SymInt tile_size, Tensor(a!) output_tensor) -> ()",
            device_types="cuda")
        def cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: torch.Tensor,
        ) -> None:
            """Tune and launch both Rubin locality domain BF16 MoE FC1 partitions.

            The MoE outer runner primes this op before CUDA graph capture.
            Direct callers must likewise invoke it once before capture.
            """
            if weight_0.shape != weight_1.shape:
                raise ValueError(
                    "locality domain BF16 MoE FC1 weight shards must have identical "
                    f"shapes, got {tuple(weight_0.shape)} and "
                    f"{tuple(weight_1.shape)}.")
            if weight_0.dtype != weight_1.dtype:
                raise ValueError(
                    "locality domain BF16 MoE FC1 weight shards must have identical "
                    f"dtypes, got {weight_0.dtype} and {weight_1.dtype}.")

            runtime = LocalityDomainRuntime(num_partitions=2)
            # Preserve the pre-refactor leaf namespace for cache compatibility.
            tuner_key = (
                "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin")
            op_runner = Sm107ContiguousGatherGroupedGemmSwigluFusionRunner(
                num_experts,
                top_k,
                num_local_experts,
                local_expert_offset,
                tile_size,
                input_dtype=input.dtype,
            )
            inputs = [
                input,
                weight_0,
                alpha,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                output_tensor,
            ]

            def launch_partition(
                partition_id: int,
                partition_inputs: List[torch.Tensor],
                tactic,
            ) -> None:
                weight = weight_0 if partition_id == 0 else weight_1
                torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
                    input=partition_inputs[0],
                    weight=weight,
                    alpha=partition_inputs[2],
                    tile_idx_to_group_idx=partition_inputs[3],
                    tile_idx_to_mn_limit=partition_inputs[4],
                    permuted_idx_to_expanded_idx=partition_inputs[5],
                    num_non_exiting_tiles=partition_inputs[6],
                    num_experts=num_experts,
                    top_k=top_k,
                    num_local_experts=num_local_experts,
                    local_expert_offset=local_expert_offset,
                    tile_size=tile_size,
                    output_tensor=partition_inputs[7],
                    partition_id=partition_id,
                    precomputed_tactic=repr(tactic),
                )

            runner, best_tactic = tune_locality_domain_concurrent(
                tuner_key,
                op_runner,
                runtime,
                2,
                launch_partition,
                inputs,
                op_runner.get_tuning_config(has_output_tensor=True),
            )
            runner(inputs, tactic=best_tactic)

        @torch.library.register_fake(
            "trtllm::cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin"
        )
        def _(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_tensor: torch.Tensor,
        ) -> None:
            return None

        # ----------------------------------------------------------------
        # Rubin Finalize Fusion (FC2 layer: grouped GEMM + scatter-add)
        # ----------------------------------------------------------------
        from ..cute_dsl_kernels.rubin.moe.rubin_contiguous_grouped_blockscaled_gemm_finalize_fusion import \
            Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel

        class Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
                TunableRunner):
            """Rubin (SM107) runner for grouped GEMM + finalize fusion (FC2).

            This is the Rubin counterpart to
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner.
            Key differences from Blackwell:
            - Takes mma_inst_shape and mma_tiler as 3-tuples (not 2-tuples)
            - Kernel __init__ takes topK parameter
            - Supports B-reuse pattern
            """
            kernel_class = Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel
            kernel_cache = dict()
            tuning_config_cache = dict()

            def __init__(self,
                         num_experts: int,
                         top_k: int,
                         num_local_experts: int,
                         local_expert_offset: int,
                         tile_size: int,
                         output_dtype: torch.dtype,
                         scaling_vector_size: int = 16):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.num_local_experts = num_local_experts
                self.local_expert_offset = local_expert_offset
                self.tile_size = tile_size

                assert output_dtype == torch.bfloat16
                self.output_dtype = output_dtype
                self.scaling_vector_size = scaling_vector_size

                if (sm_version := get_sm_version()) != 107:
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports SM 107 (Rubin) only, but got SM {sm_version}"
                    )

                if self.tile_size not in (128, 256, 512):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports tile_size 128, 256 and 512 only, but got {self.tile_size}"
                    )

            def unique_id(self):
                return (
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    self.tile_size,
                    self.output_dtype,
                    self.scaling_vector_size,
                )

            @staticmethod
            def _is_n_tiling_compatible(n: int, mma_n: int,
                                        cluster_n: int) -> bool:
                """Return whether the kernel can cover N without a tail tile."""
                return n % (mma_n * cluster_n) == 0

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
                **kwargs,
            ) -> List[Tuple[int, int]]:
                a, b, *_ = inputs
                m, k = a.size(0), a.size(1) * 2
                l, n = b.size(0), b.size(1)  # noqa: E741

                # Rubin FP4 K-dimension: mma_tiler_k=256, mma_inst_k=128
                mma_tiler_k = 256
                mma_inst_k = 128

                # (mma_tiler_m, mma_inst_m) candidates
                mma_m_candidates = [
                    (128, 128),  # no B-reuse, 1CTA
                    (256, 256),  # no B-reuse, 2CTA
                    (256, 128),  # B-reuse, 1CTA
                    (512, 256),  # B-reuse, 2CTA
                ]
                # Restrict N candidates to 128 and 256 (matching Blackwell).
                # mma_n=192 and mma_n=64 trigger the kernel's special SFB
                # slicing paths (cta_tile_shape_n=192 / cta_tile_shape_n=64)
                # which cause CUDA_ERROR_ILLEGAL_ADDRESS for certain N
                # dimensions (e.g., Qwen3-30B-A3B with N=2048, mma_n=192).
                mma_n_candidates = [128, 256]
                cluster_shape_mn_candidates = [(1, 1), (2, 1), (1, 2), (2, 2)]
                raster_along_m_candidates = [False]

                valid_tactics = []
                for (mma_tiler_m,
                     mma_inst_m), mma_n, cluster_shape_mn, raster_along_m in (
                         itertools.product(mma_m_candidates, mma_n_candidates,
                                           cluster_shape_mn_candidates,
                                           raster_along_m_candidates)):

                    # The kernel requires each cluster to cover exactly
                    # one routing tile along M:
                    #   - 1-CTA (mma_inst_m=128): cluster_m must be 1
                    #   - 2-CTA (mma_inst_m=256): cluster_m must be 2
                    # and the CTA-pair's M extent (mma_tiler_m) must equal
                    # the routing tile_size.  Any other combination
                    # (mma_tiler_m != tile_size, or cluster_m spanning
                    # multiple independent M-tiles) produces ~40% element
                    # mismatches empirically.
                    cta_group_m = 2 if mma_inst_m == 256 else 1
                    if cluster_shape_mn[0] != cta_group_m:
                        continue
                    if mma_tiler_m != self.tile_size:
                        continue
                    if not self._is_n_tiling_compatible(n, mma_n,
                                                        cluster_shape_mn[1]):
                        continue

                    mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
                    mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)

                    if self.__class__.kernel_class.can_implement(
                            a_dtype=cutlass.Float4E2M1FN,
                            b_dtype=cutlass.Float4E2M1FN,
                            sf_dtype=cutlass.Float8E4M3FN,
                            sf_vec_size=self.scaling_vector_size,
                            c_dtype=cutlass.BFloat16,
                            mma_inst_shape=mma_inst_shape,
                            mma_tiler=mma_tiler,
                            cluster_shape_mn=cluster_shape_mn,
                            m=m,
                            n=n,
                            k=k,
                            l=l,
                            a_major="k",
                            b_major="k",
                            c_major="n",
                    ):
                        valid_tactics.append((mma_tiler, mma_inst_shape,
                                              cluster_shape_mn, raster_along_m))

                logger.debug(
                    f"CuteDSL Rubin GroupedGemmFinalize: Found {len(valid_tactics)} valid tactics "
                    f"for M={m}, N={n}, K={k}, L={l}")
                return valid_tactics

            def get_tuning_config(self) -> TuningConfig:
                key = self.unique_id()
                if key not in self.__class__.tuning_config_cache:
                    helper = GroupedGemmInputsHelper(self.num_experts,
                                                     self.top_k,
                                                     self.num_local_experts,
                                                     self.local_expert_offset,
                                                     self.tile_size)
                    self.__class__.tuning_config_cache[key] = TuningConfig(
                        dynamic_tensor_specs=(DynamicTensorSpec(
                            0, 0, helper.gen_tuning_buckets,
                            helper.map_to_tuning_buckets), ),
                        constraint_specs=(
                            ConstraintSpec(2, 0, fp4_scale_infer_shape),
                            ConstraintSpec(5, 0, helper.infer_shape_num_tokens),
                            ConstraintSpec(6, 0,
                                           helper.infer_shape_max_num_tiles),
                            ConstraintSpec(7, 0,
                                           helper.infer_shape_max_num_tiles),
                            ConstraintSpec(
                                8, 0,
                                helper.infer_shape_max_num_permuted_tokens),
                            ConstraintSpec(10, 0,
                                           helper.infer_shape_num_tokens),
                        ),
                        inputs_pre_hook=helper.inputs_pre_hook_finalize_fusion,
                    )
                return self.__class__.tuning_config_cache[key]

            def forward(self, inputs: List[torch.Tensor],
                        tactic: Optional[tuple]) -> torch.Tensor:
                a, b, a_sf, b_sf, alpha, c, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales = inputs
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
                l, n = b.size(0), b.size(1)  # noqa: E741
                scale_k = k // self.scaling_vector_size
                assert m % self.tile_size == 0
                assert k % (self.scaling_vector_size * 4) == 0
                assert b.size(2) * 2 == k
                assert a_sf.size(0) == m * scale_k
                assert b_sf.size(0) == l
                assert b_sf.size(1) == n
                assert b_sf.size(2) == scale_k
                assert alpha.size(0) == l

                assert c.dtype == self.output_dtype
                assert c.dim() == 2
                num_tokens = c.size(0)
                assert c.size(1) == n or c.size(1) == n * 2

                num_tiles = m // self.tile_size
                assert tile_idx_to_group_idx.dtype == torch.int32
                assert tile_idx_to_group_idx.size() == (num_tiles, )
                assert tile_idx_to_mn_limit.dtype == torch.int32
                assert tile_idx_to_mn_limit.size() == (num_tiles, )
                assert permuted_idx_to_expanded_idx.dtype == torch.int32
                assert permuted_idx_to_expanded_idx.size() == (m, )
                assert num_non_exiting_tiles.dtype == torch.int32
                assert num_non_exiting_tiles.numel() == 1
                assert token_final_scales.dtype == torch.float32
                assert token_final_scales.dim() == 2
                assert token_final_scales.size() == (num_tokens, self.top_k)

                locality_domain_id = get_current_locality_domain()
                locality_domain_half_gemm = locality_domain_id is not None
                if locality_domain_half_gemm:
                    assert locality_domain_id in (
                        0,
                        1), f"Invalid locality domain id: {locality_domain_id}"
                    assert c.size(1) == n * 2, \
                        f"[locality domain] FC2 output must be 2x width: c.size(1)={c.size(1)}, n={n}"
                    # c: stride directly into shared buffer (kernel uses c_stride_row)
                    c = c[:,
                          locality_domain_id * n:(locality_domain_id + 1) * n]

                if isinstance(tactic, tuple):
                    mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m = tactic
                else:
                    # Default tactic for Rubin
                    mma_tiler, mma_inst_shape, cluster_shape_mn = \
                        _get_sm107_nvfp4_default_mma_config(self.tile_size)
                    raster_along_m = False
                assert mma_tiler[
                    0] >= self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"
                if not self._is_n_tiling_compatible(n, mma_tiler[1],
                                                    cluster_shape_mn[1]):
                    raise ValueError(
                        f"Tactic ({tactic}) is incompatible with N={n}: "
                        f"mma_n={mma_tiler[1]} and cluster_n={cluster_shape_mn[1]} "
                        "require N to be divisible by their product.")

                a_ptr = make_ptr(cutlass.Float4E2M1FN,
                                 a.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=32)
                b_ptr = make_ptr(cutlass.Float4E2M1FN,
                                 b.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=32)
                a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                    a_sf.data_ptr(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16)
                b_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                    b_sf.data_ptr(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16)
                alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(),
                                     cute.AddressSpace.gmem)
                tile_idx_to_group_idx_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                tile_idx_to_mn_limit_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_mn_limit.data_ptr(),
                    cute.AddressSpace.gmem)
                permuted_idx_to_expanded_idx_ptr = make_ptr(
                    cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                num_non_exiting_tiles_ptr = make_ptr(
                    cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                    cute.AddressSpace.gmem)
                token_final_scales_ptr = make_ptr(cutlass.Float32,
                                                  token_final_scales.data_ptr(),
                                                  cute.AddressSpace.gmem)
                c_ptr = make_ptr(cutlass.BFloat16,
                                 c.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                # c_stride_row for locality domain strided output (0 = default contiguous)
                c_stride_row_val = cutlass.Int64(
                    n * 2) if locality_domain_half_gemm else cutlass.Int64(0)

                max_active_clusters = get_max_activate_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])
                cache_key = (self.scaling_vector_size, self.tile_size,
                             self.top_k, mma_tiler, mma_inst_shape,
                             cluster_shape_mn, raster_along_m,
                             locality_domain_half_gemm, max_active_clusters)
                if cache_key not in self.__class__.kernel_cache:
                    gemm = self.__class__.kernel_class(
                        sf_vec_size=self.scaling_vector_size,
                        mma_inst_shape=mma_inst_shape,
                        mma_tiler=mma_tiler,
                        cluster_shape_mn=cluster_shape_mn,
                        raster_along_m=raster_along_m,
                        topK=self.top_k,
                    )
                    compiled_gemm = cute.compile(
                        gemm.wrapper,
                        a_ptr,
                        b_ptr,
                        a_sf_ptr,
                        b_sf_ptr,
                        c_ptr,
                        alpha_ptr,
                        tile_idx_to_group_idx_ptr,
                        tile_idx_to_mn_limit_ptr,
                        permuted_idx_to_expanded_idx_ptr,
                        num_non_exiting_tiles_ptr,
                        token_final_scales_ptr,
                        m,
                        n,
                        k,
                        l,
                        num_tokens,
                        self.top_k,
                        tile_size=self.tile_size,
                        scaling_vector_size=self.scaling_vector_size,
                        max_active_clusters=max_active_clusters,
                        stream=stream,
                        c_stride_row=c_stride_row_val,
                    )
                    self.__class__.kernel_cache[cache_key] = compiled_gemm
                else:
                    compiled_gemm = self.__class__.kernel_cache[cache_key]

                compiled_gemm(
                    a_ptr,
                    b_ptr,
                    a_sf_ptr,
                    b_sf_ptr,
                    c_ptr,
                    alpha_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    token_final_scales_ptr,
                    m,
                    n,
                    k,
                    l,
                    num_tokens,
                    self.top_k,
                    stream=stream,
                    c_stride_row=c_stride_row_val,
                )
                # c written via stride — no copy-back needed
                return c

        @torch.library.custom_op(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin",
            mutates_args=("output", ),
            device_types="cuda")
        def cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
            precomputed_tactic: Optional[str] = None,
        ) -> None:
            tuner = AutoTuner.get()

            runner = Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
                num_experts, top_k, num_local_experts, local_expert_offset,
                tile_size, output_dtype, scaling_vector_size)

            inputs = [
                input, weight, input_scale, weight_scale, alpha, output,
                tile_idx_to_group_idx, tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx, num_non_exiting_tiles,
                token_final_scales
            ]

            if precomputed_tactic is None:
                _, best_tactic = tuner.choose_one(
                    "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin",
                    [runner],
                    runner.get_tuning_config(),
                    inputs,
                )
            else:
                best_tactic = ast.literal_eval(precomputed_tactic)

            runner(inputs, tactic=best_tactic)

        @torch.library.custom_op(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_rubin",
            mutates_args=(),
            device_types="cuda")
        def cute_dsl_nvfp4_grouped_gemm_finalize_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
        ) -> torch.Tensor:
            num_tokens = token_final_scales.size(0)
            n = weight.size(1)
            output = torch.zeros(num_tokens,
                                 n,
                                 dtype=output_dtype,
                                 device=input.device)
            torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin(
                input=input,
                weight=weight,
                input_scale=input_scale,
                weight_scale=weight_scale,
                alpha=alpha,
                output=output,
                tile_idx_to_group_idx=tile_idx_to_group_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                token_final_scales=token_final_scales,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                tile_size=tile_size,
                output_dtype=output_dtype,
                scaling_vector_size=scaling_vector_size,
            )
            return output

        @torch.library.register_fake(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
            precomputed_tactic: Optional[str] = None,
        ) -> None:
            return

        @torch.library.custom_op(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin",
            mutates_args=("output", ),
            schema=
            "(Tensor input, Tensor weight_0, Tensor weight_1, Tensor input_scale, "
            "Tensor weight_scale_0, Tensor weight_scale_1, Tensor alpha, "
            "Tensor(a!) output, Tensor tile_idx_to_group_idx, "
            "Tensor tile_idx_to_mn_limit, "
            "Tensor expanded_idx_to_permuted_idx, "
            "Tensor permuted_idx_to_expanded_idx, "
            "Tensor num_non_exiting_tiles, Tensor token_final_scales, "
            "SymInt num_experts, SymInt top_k, SymInt num_local_experts, "
            "SymInt local_expert_offset, SymInt tile_size, "
            "ScalarType output_dtype, SymInt ep_size, "
            "bool enable_alltoall=False, SymInt scaling_vector_size=16) -> ()",
            device_types="cuda")
        def cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale_0: torch.Tensor,
            weight_scale_1: torch.Tensor,
            alpha: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            expanded_idx_to_permuted_idx: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            ep_size: int,
            enable_alltoall: bool = False,
            scaling_vector_size: int = 16,
        ) -> None:
            """Tune and launch both Rubin locality domain NVFP4 MoE FC2 partitions.

            The MoE outer runner primes this op before CUDA graph capture.
            Direct callers must likewise invoke it once before capture.
            """
            if weight_0.shape != weight_1.shape:
                raise ValueError(
                    "locality domain NVFP4 MoE FC2 weight shards must have identical "
                    f"shapes, got {tuple(weight_0.shape)} and "
                    f"{tuple(weight_1.shape)}.")
            if weight_0.dtype != weight_1.dtype:
                raise ValueError(
                    "locality domain NVFP4 MoE FC2 weight shards must have identical "
                    f"dtypes, got {weight_0.dtype} and {weight_1.dtype}.")
            if weight_scale_0.shape != weight_scale_1.shape:
                raise ValueError(
                    "locality domain NVFP4 MoE FC2 weight-scale shards must have "
                    f"identical shapes, got {tuple(weight_scale_0.shape)} and "
                    f"{tuple(weight_scale_1.shape)}.")
            if weight_scale_0.dtype != weight_scale_1.dtype:
                raise ValueError(
                    "locality domain NVFP4 MoE FC2 weight-scale shards must have "
                    f"identical dtypes, got {weight_scale_0.dtype} and "
                    f"{weight_scale_1.dtype}.")
            if output.dtype != output_dtype:
                raise ValueError(
                    "locality domain NVFP4 MoE FC2 output tensor dtype must match "
                    f"output_dtype, got {output.dtype} and {output_dtype}.")

            runtime = LocalityDomainRuntime(num_partitions=2)
            # Preserve the pre-refactor leaf namespace for cache compatibility.
            tuner_key = (
                "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin")
            op_runner = (
                Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
                    num_experts,
                    top_k,
                    num_local_experts,
                    local_expert_offset,
                    tile_size,
                    output_dtype,
                    scaling_vector_size,
                ))
            inputs = [
                input,
                weight_0,
                input_scale,
                weight_scale_0,
                alpha,
                output,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                token_final_scales,
            ]

            def launch_partition(
                partition_id: int,
                partition_inputs: List[torch.Tensor],
                tactic,
            ) -> None:
                weight = weight_0 if partition_id == 0 else weight_1
                weight_scale = (weight_scale_0
                                if partition_id == 0 else weight_scale_1)
                torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin(
                    input=partition_inputs[0],
                    weight=weight,
                    input_scale=partition_inputs[2],
                    weight_scale=weight_scale,
                    alpha=partition_inputs[4],
                    output=partition_inputs[5],
                    tile_idx_to_group_idx=partition_inputs[6],
                    tile_idx_to_mn_limit=partition_inputs[7],
                    permuted_idx_to_expanded_idx=partition_inputs[8],
                    num_non_exiting_tiles=partition_inputs[9],
                    token_final_scales=partition_inputs[10],
                    num_experts=num_experts,
                    top_k=top_k,
                    num_local_experts=num_local_experts,
                    local_expert_offset=local_expert_offset,
                    tile_size=tile_size,
                    output_dtype=output_dtype,
                    scaling_vector_size=scaling_vector_size,
                    precomputed_tactic=repr(tactic),
                )

            runner, best_tactic = tune_locality_domain_concurrent(
                tuner_key,
                op_runner,
                runtime,
                2,
                launch_partition,
                inputs,
                op_runner.get_tuning_config(),
            )
            # Profiling finalize accumulates into the shared output. Preserve
            # the selective all-to-all memset semantics when restoring the
            # baseline before the actual dual-partition launch.
            if AutoTuner.get().is_tuning_mode:
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=top_k,
                    ep_size=ep_size,
                    enable_alltoall=enable_alltoall,
                )
            runner(inputs, tactic=best_tactic)

        @torch.library.register_fake(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin"
        )
        def _(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale_0: torch.Tensor,
            weight_scale_1: torch.Tensor,
            alpha: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            expanded_idx_to_permuted_idx: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            ep_size: int,
            enable_alltoall: bool = False,
            scaling_vector_size: int = 16,
        ) -> None:
            return None

        @torch.library.register_fake(
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            alpha: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            scaling_vector_size: int = 16,
        ) -> torch.Tensor:
            num_tokens = token_final_scales.size(0)
            n = weight.size(1)
            return torch.empty(num_tokens,
                               n,
                               dtype=output_dtype,
                               device=input.device)

        # ----------------------------------------------------------------
        # Rubin BF16/FP16 Finalize Fusion (FC2 layer: grouped GEMM + scatter-add)
        # ----------------------------------------------------------------
        from ..cute_dsl_kernels.rubin.moe.rubin_contiguous_grouped_gemm_finalize_fusion import \
            Sm107ContiguousGroupedGemmFinalizeFusionKernel

        class Sm107ContiguousGroupedGemmFinalizeFusionRunner(TunableRunner):
            """Rubin (SM107) runner for BF16/FP16 grouped GEMM + finalize fusion (FC2).

            Similar to the blockscaled finalize runner but without scale factors.
            Uses MmaF16BF16Op (K=16) for BFloat16/Float16 inputs.
            """
            kernel_class = Sm107ContiguousGroupedGemmFinalizeFusionKernel
            kernel_cache = dict()
            tuning_config_cache = dict()

            def __init__(self,
                         num_experts: int,
                         top_k: int,
                         num_local_experts: int,
                         local_expert_offset: int,
                         tile_size: int,
                         output_dtype: torch.dtype,
                         input_dtype: Optional[torch.dtype] = None):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.num_local_experts = num_local_experts
                self.local_expert_offset = local_expert_offset
                self.tile_size = tile_size

                assert output_dtype in (torch.bfloat16, torch.float16)
                self.output_dtype = output_dtype
                self.input_dtype = input_dtype
                if input_dtype is not None and input_dtype not in (
                        torch.bfloat16, torch.float16):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} requires BF16 or FP16 input, "
                        f"but got {input_dtype}")

                if (sm_version := get_sm_version()) != 107:
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports SM 107 (Rubin) only, but got SM {sm_version}"
                    )

                if self.tile_size not in (64, 128, 256):
                    raise ValueError(
                        f"{self.__class__.kernel_class.__name__} supports tile_size 64, 128 and 256 only, but got {self.tile_size}"
                    )

            def unique_id(self):
                return (
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    self.tile_size,
                    self.output_dtype,
                    self.input_dtype,
                )

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
                **kwargs,
            ) -> List[Tuple[int, int]]:
                a, b, *_ = inputs
                m, k = a.size(0), a.size(1)
                l, n = b.size(0), b.size(1)  # noqa: E741

                ab_dtype = cutlass.BFloat16 if a.dtype == torch.bfloat16 else cutlass.Float16
                c_dtype = cutlass.BFloat16 if self.output_dtype == torch.bfloat16 else cutlass.Float16

                # BF16/FP16 K-dimension: mma_tiler_k=64, mma_inst_k=16
                mma_tiler_k = 64
                mma_inst_k = 16

                # BF16 (no B-reuse): CTA tile M must equal tile_size.
                # cluster_shape_mn is always (max(1, tile_size//128), 1).
                mma_n_candidates = [128, 256]
                raster_along_m_candidates = [False]

                mma_m = self.tile_size
                cluster_shape_mn = (max(1, self.tile_size // 128), 1)

                valid_tactics = []
                for mma_n, raster_along_m in (itertools.product(
                        mma_n_candidates, raster_along_m_candidates)):

                    mma_tiler = (mma_m, mma_n, mma_tiler_k)
                    mma_inst_shape = (mma_m, mma_n, mma_inst_k)

                    if self.__class__.kernel_class.can_implement(
                            a_dtype=ab_dtype,
                            b_dtype=ab_dtype,
                            c_dtype=c_dtype,
                            mma_inst_shape=mma_inst_shape,
                            mma_tiler=mma_tiler,
                            cluster_shape_mn=cluster_shape_mn,
                            m=m,
                            n=n,
                            k=k,
                            l=l,
                            a_major="k",
                            b_major="k",
                            c_major="n",
                    ):
                        valid_tactics.append((mma_tiler, mma_inst_shape,
                                              cluster_shape_mn, raster_along_m))

                logger.debug(
                    f"CuteDSL Rubin BF16 GroupedGemmFinalize: Found {len(valid_tactics)} valid tactics "
                    f"for M={m}, N={n}, K={k}, L={l}")
                return valid_tactics

            def _bf16_inputs_pre_hook_finalize(
                    self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
                """Pre-hook for BF16 finalize (no scale factors)."""
                a, b, output, tile_idx_to_group_idx, \
                    tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, \
                    num_non_exiting_tiles, token_final_scales = inputs

                helper = GroupedGemmInputsHelper(self.num_experts, self.top_k,
                                                 self.num_local_experts,
                                                 self.local_expert_offset,
                                                 self.tile_size)
                num_tokens = helper.infer_num_tokens(a.size(0))
                num_tokens_per_expert = helper.generate_num_tokens_per_expert(
                    num_tokens, approx_max_load=True)
                token_selected_experts = \
                    helper.generate_token_selected_experts(
                        num_tokens, num_tokens_per_expert)

                token_selected_experts = token_selected_experts.cuda()
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)
                (
                    tile_idx_to_group_idx,
                    tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx,
                    total_num_padded_tokens,
                    num_non_exiting_tiles,
                ) = torch.ops.trtllm.moe_sort(
                    token_selected_experts=token_selected_experts,
                    token_final_scales=token_final_scales,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    local_expert_offset=self.local_expert_offset,
                    local_num_experts=self.num_local_experts,
                    tile_tokens_dim=self.tile_size,
                )
                return (a, b, output, tile_idx_to_group_idx,
                        tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles, token_final_scales)

            def get_tuning_config(self) -> TuningConfig:
                key = self.unique_id()
                if key not in self.__class__.tuning_config_cache:
                    helper = GroupedGemmInputsHelper(self.num_experts,
                                                     self.top_k,
                                                     self.num_local_experts,
                                                     self.local_expert_offset,
                                                     self.tile_size)
                    # BF16 finalize input layout (8 tensors):
                    #   0: a, 1: b, 2: c (output),
                    #   3: tile_idx_to_group_idx, 4: tile_idx_to_mn_limit,
                    #   5: permuted_idx_to_expanded_idx, 6: num_non_exiting_tiles,
                    #   7: token_final_scales
                    self.__class__.tuning_config_cache[key] = TuningConfig(
                        dynamic_tensor_specs=(DynamicTensorSpec(
                            0, 0, helper.gen_tuning_buckets,
                            helper.map_to_tuning_buckets), ),
                        constraint_specs=(
                            ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                            ConstraintSpec(3, 0,
                                           helper.infer_shape_max_num_tiles),
                            ConstraintSpec(4, 0,
                                           helper.infer_shape_max_num_tiles),
                            ConstraintSpec(
                                5, 0,
                                helper.infer_shape_max_num_permuted_tokens),
                            ConstraintSpec(7, 0, helper.infer_shape_num_tokens),
                        ),
                        inputs_pre_hook=self._bf16_inputs_pre_hook_finalize,
                    )
                return self.__class__.tuning_config_cache[key]

            def forward(self, inputs: List[torch.Tensor],
                        tactic: Optional[tuple]) -> torch.Tensor:
                a, b, c, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales = inputs
                assert a.dtype in (torch.bfloat16, torch.float16)
                assert a.dim() == 2
                assert b.dtype == a.dtype
                assert b.dim() == 3

                ab_dtype = cutlass.BFloat16 if a.dtype == torch.bfloat16 else cutlass.Float16
                c_cutlass_dtype = cutlass.BFloat16 if c.dtype == torch.bfloat16 else cutlass.Float16

                m, k = a.size(0), a.size(1)
                l, n = b.size(0), b.size(1)  # noqa: E741
                assert m % self.tile_size == 0
                assert b.size(2) == k

                assert c.dtype == self.output_dtype
                assert c.dim() == 2
                num_tokens = c.size(0)
                assert c.size(1) == n or c.size(1) == n * 2

                locality_domain_id = get_current_locality_domain()
                locality_domain_half_gemm = locality_domain_id is not None
                if locality_domain_half_gemm:
                    assert locality_domain_id in (
                        0,
                        1), f"Invalid locality domain id: {locality_domain_id}"
                    assert c.size(1) == n * 2, \
                        f"[locality domain] BF16 FC2 output must be 2x width: c.size(1)={c.size(1)}, n={n}"
                    c_stride_row_val = cutlass.Int64(c.size(1))
                    c = c[:,
                          locality_domain_id * n:(locality_domain_id + 1) * n]
                else:
                    c_stride_row_val = cutlass.Int64(0)

                num_tiles = m // self.tile_size
                assert tile_idx_to_group_idx.dtype == torch.int32
                assert tile_idx_to_group_idx.size() == (num_tiles, )
                assert tile_idx_to_mn_limit.dtype == torch.int32
                assert tile_idx_to_mn_limit.size() == (num_tiles, )
                assert permuted_idx_to_expanded_idx.dtype == torch.int32
                assert permuted_idx_to_expanded_idx.size() == (m, )
                assert num_non_exiting_tiles.dtype == torch.int32
                assert num_non_exiting_tiles.numel() == 1
                assert token_final_scales.dtype == torch.float32
                assert token_final_scales.dim() == 2
                assert token_final_scales.size() == (num_tokens, self.top_k)

                a_ptr = make_ptr(ab_dtype,
                                 a.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)
                b_ptr = make_ptr(ab_dtype,
                                 b.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)
                tile_idx_to_group_idx_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_group_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                tile_idx_to_mn_limit_ptr = make_ptr(
                    cutlass.Int32, tile_idx_to_mn_limit.data_ptr(),
                    cute.AddressSpace.gmem)
                permuted_idx_to_expanded_idx_ptr = make_ptr(
                    cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(),
                    cute.AddressSpace.gmem)
                num_non_exiting_tiles_ptr = make_ptr(
                    cutlass.Int32, num_non_exiting_tiles.data_ptr(),
                    cute.AddressSpace.gmem)
                token_final_scales_ptr = make_ptr(cutlass.Float32,
                                                  token_final_scales.data_ptr(),
                                                  cute.AddressSpace.gmem)
                c_ptr = make_ptr(c_cutlass_dtype,
                                 c.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=16)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                if isinstance(tactic, tuple):
                    mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m = tactic
                else:
                    # Default tactic for Rubin BF16/FP16
                    mma_tiler = (self.tile_size, 128, 64)
                    mma_inst_shape = (self.tile_size, 128, 16)
                    cluster_shape_mn = (max(1, self.tile_size // 128), 1)
                    raster_along_m = False

                max_active_clusters = get_max_activate_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])
                cache_key = (a.dtype, c.dtype, self.tile_size, self.top_k,
                             mma_tiler, mma_inst_shape, cluster_shape_mn,
                             raster_along_m, locality_domain_half_gemm,
                             max_active_clusters)
                if cache_key not in self.__class__.kernel_cache:
                    gemm = self.__class__.kernel_class(
                        mma_inst_shape=mma_inst_shape,
                        mma_tiler=mma_tiler,
                        cluster_shape_mn=cluster_shape_mn,
                        raster_along_m=raster_along_m,
                        topK=self.top_k,
                    )
                    compiled_gemm = cute.compile(
                        gemm.wrapper,
                        a_ptr,
                        b_ptr,
                        c_ptr,
                        tile_idx_to_group_idx_ptr,
                        tile_idx_to_mn_limit_ptr,
                        permuted_idx_to_expanded_idx_ptr,
                        num_non_exiting_tiles_ptr,
                        token_final_scales_ptr,
                        m,
                        n,
                        k,
                        l,
                        num_tokens,
                        self.top_k,
                        tile_size=self.tile_size,
                        max_active_clusters=max_active_clusters,
                        stream=stream,
                        c_stride_row=c_stride_row_val,
                    )
                    self.__class__.kernel_cache[cache_key] = compiled_gemm
                else:
                    compiled_gemm = self.__class__.kernel_cache[cache_key]

                compiled_gemm(
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    num_non_exiting_tiles_ptr,
                    token_final_scales_ptr,
                    m,
                    n,
                    k,
                    l,
                    num_tokens,
                    self.top_k,
                    stream=stream,
                    c_stride_row=c_stride_row_val,
                )
                return c

        @torch.library.custom_op(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin",
            mutates_args=("output", ),
            device_types="cuda")
        def cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            precomputed_tactic: Optional[str] = None,
        ) -> None:
            tuner = AutoTuner.get()

            runner = Sm107ContiguousGroupedGemmFinalizeFusionRunner(
                num_experts,
                top_k,
                num_local_experts,
                local_expert_offset,
                tile_size,
                output_dtype,
                input_dtype=input.dtype)

            inputs = [
                input, weight, output, tile_idx_to_group_idx,
                tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                num_non_exiting_tiles, token_final_scales
            ]

            if precomputed_tactic is None:
                _, best_tactic = tuner.choose_one(
                    "trtllm::cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin",
                    [runner],
                    runner.get_tuning_config(),
                    inputs,
                )
            else:
                best_tactic = ast.literal_eval(precomputed_tactic)

            runner(inputs, tactic=best_tactic)

        @torch.library.custom_op(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_rubin",
            mutates_args=(),
            device_types="cuda")
        def cute_dsl_bf16_grouped_gemm_finalize_rubin(
            input: torch.Tensor,
            weight: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
        ) -> torch.Tensor:
            num_tokens = token_final_scales.size(0)
            n = weight.size(1)
            output = torch.zeros(num_tokens,
                                 n,
                                 dtype=output_dtype,
                                 device=input.device)
            torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
                input=input,
                weight=weight,
                output=output,
                tile_idx_to_group_idx=tile_idx_to_group_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                token_final_scales=token_final_scales,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                tile_size=tile_size,
                output_dtype=output_dtype,
            )
            return output

        @torch.library.register_fake(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            precomputed_tactic: Optional[str] = None,
        ) -> None:
            return

        @torch.library.custom_op(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin",
            mutates_args=("output", ),
            schema="(Tensor input, Tensor weight_0, Tensor weight_1, "
            "Tensor(a!) output, Tensor tile_idx_to_group_idx, "
            "Tensor tile_idx_to_mn_limit, "
            "Tensor expanded_idx_to_permuted_idx, "
            "Tensor permuted_idx_to_expanded_idx, "
            "Tensor num_non_exiting_tiles, Tensor token_final_scales, "
            "SymInt num_experts, SymInt top_k, SymInt num_local_experts, "
            "SymInt local_expert_offset, SymInt tile_size, "
            "ScalarType output_dtype, SymInt ep_size, "
            "bool enable_alltoall=False) -> ()",
            device_types="cuda")
        def cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            expanded_idx_to_permuted_idx: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            ep_size: int,
            enable_alltoall: bool = False,
        ) -> None:
            """Tune and launch both Rubin locality domain BF16 MoE FC2 partitions.

            The MoE outer runner primes this op before CUDA graph capture.
            Direct callers must likewise invoke it once before capture.
            """
            if weight_0.shape != weight_1.shape:
                raise ValueError(
                    "locality domain BF16 MoE FC2 weight shards must have identical "
                    f"shapes, got {tuple(weight_0.shape)} and "
                    f"{tuple(weight_1.shape)}.")
            if weight_0.dtype != weight_1.dtype:
                raise ValueError(
                    "locality domain BF16 MoE FC2 weight shards must have identical "
                    f"dtypes, got {weight_0.dtype} and {weight_1.dtype}.")
            if output.dtype != output_dtype:
                raise ValueError(
                    "locality domain BF16 MoE FC2 output tensor dtype must match "
                    f"output_dtype, got {output.dtype} and {output_dtype}.")

            runtime = LocalityDomainRuntime(num_partitions=2)
            # Preserve the pre-refactor leaf namespace for cache compatibility.
            tuner_key = (
                "trtllm::cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin")
            op_runner = Sm107ContiguousGroupedGemmFinalizeFusionRunner(
                num_experts,
                top_k,
                num_local_experts,
                local_expert_offset,
                tile_size,
                output_dtype,
                input_dtype=input.dtype,
            )
            inputs = [
                input,
                weight_0,
                output,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                token_final_scales,
            ]

            def launch_partition(
                partition_id: int,
                partition_inputs: List[torch.Tensor],
                tactic,
            ) -> None:
                weight = weight_0 if partition_id == 0 else weight_1
                torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
                    input=partition_inputs[0],
                    weight=weight,
                    output=partition_inputs[2],
                    tile_idx_to_group_idx=partition_inputs[3],
                    tile_idx_to_mn_limit=partition_inputs[4],
                    permuted_idx_to_expanded_idx=partition_inputs[5],
                    num_non_exiting_tiles=partition_inputs[6],
                    token_final_scales=partition_inputs[7],
                    num_experts=num_experts,
                    top_k=top_k,
                    num_local_experts=num_local_experts,
                    local_expert_offset=local_expert_offset,
                    tile_size=tile_size,
                    output_dtype=output_dtype,
                    precomputed_tactic=repr(tactic),
                )

            runner, best_tactic = tune_locality_domain_concurrent(
                tuner_key,
                op_runner,
                runtime,
                2,
                launch_partition,
                inputs,
                op_runner.get_tuning_config(),
            )
            # Restore the same selective zero baseline used by the backend
            # before the real dual-partition finalize launch.
            if AutoTuner.get().is_tuning_mode:
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=output,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=tile_size,
                    top_k=top_k,
                    ep_size=ep_size,
                    enable_alltoall=enable_alltoall,
                )
            runner(inputs, tactic=best_tactic)

        @torch.library.register_fake(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin"
        )
        def _(
            input: torch.Tensor,
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            output: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            expanded_idx_to_permuted_idx: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
            ep_size: int,
            enable_alltoall: bool = False,
        ) -> None:
            return None

        @torch.library.register_fake(
            "trtllm::cute_dsl_bf16_grouped_gemm_finalize_rubin")
        def _(
            input: torch.Tensor,
            weight: torch.Tensor,
            tile_idx_to_group_idx: torch.Tensor,
            tile_idx_to_mn_limit: torch.Tensor,
            permuted_idx_to_expanded_idx: torch.Tensor,
            num_non_exiting_tiles: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            output_dtype: torch.dtype,
        ) -> torch.Tensor:
            num_tokens = token_final_scales.size(0)
            n = weight.size(1)
            return torch.empty(num_tokens,
                               n,
                               dtype=output_dtype,
                               device=input.device)
