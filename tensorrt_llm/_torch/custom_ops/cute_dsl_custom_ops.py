# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.logger import logger

from ..._utils import get_sm_version, is_sm_100f
from ...math_utils import ceil_div, pad_up
from ..autotuner import (AutoTuner, ConstraintSpec, DistributedTuningStrategy,
                         DynamicTensorSpec, OptimizationProfile, TunableRunner,
                         TuningConfig)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import (deep_gemm_gen_tuning_buckets, fp4_scale_infer_shape,
                     fp8_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2, next_positive_power_of_2)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


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

    Input layout (positions 1, 3, 4 are lists for multi-B support):
        0: a                               - tensor, original input activation
        1: b_list                          - list of tensors, weight tensors
        2: a_sf                            - tensor, scale factor for a
        3: b_sf_list                       - list of tensors, scale factors for b
        4: alpha_list                      - list of tensors, per-expert scaling factors
        5: tile_idx_to_group_idx           - tensor, tile to expert mapping
        6: tile_idx_to_mn_limit            - tensor, tile M/N limits
        7: permuted_idx_to_expanded_idx    - tensor, token permutation mapping
        8: num_non_exiting_tiles           - tensor, number of valid tiles
        9: global_sf                       - tensor, global scale factor
    """
    # Override: use permuted_idx_to_expanded_idx for shape inference
    IDX_PERMUTED_IDX_TO_EXPANDED_IDX = 7
    IDX_SHAPE_INFER = IDX_PERMUTED_IDX_TO_EXPANDED_IDX

    def inputs_pre_hook(self, inputs: List) -> List:
        """Pre-hook for gather-based SwiGLU fusion kernel.

        Generates:
            - tile_idx_to_group_idx
            - tile_idx_to_mn_limit
            - permuted_idx_to_expanded_idx (for gather operation)
            - num_non_exiting_tiles

        Input layout (positions 1, 3, 4 are lists):
            0: a                               - tensor
            1: b_list                          - list of tensors
            2: a_sf                            - tensor
            3: b_sf_list                       - list of tensors
            4: alpha_list                      - list of tensors
            5: tile_idx_to_group_idx           - tensor
            6: tile_idx_to_mn_limit            - tensor
            7: permuted_idx_to_expanded_idx    - tensor
            8: num_non_exiting_tiles           - tensor
            9: global_sf                       - tensor
        """
        a, b_list, a_sf, b_sf_list, alpha_list, tile_idx_to_group_idx, \
            tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, \
            num_non_exiting_tiles, global_sf = inputs
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
        return (a, b_list, a_sf, b_sf_list, alpha_list, tile_idx_to_group_idx,
                tile_idx_to_mn_limit, permuted_idx_to_expanded_idx,
                num_non_exiting_tiles, global_sf)


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

    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import \
        BlockScaledContiguousGatherGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm import \
        Sm100BlockScaledContiguousGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import \
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_swiglu_fusion import \
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
    from ..cute_dsl_kernels.blackwell.blockwise_gemm.blockwise_gemm import \
        Sm100BlockwiseGemmKernel
    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import \
        Sm100BlockScaledPersistentDenseGemmKernel
    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_swiglu_fusion import \
        Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel
    from ..cute_dsl_kernels.blackwell.dense_gemm_persistent import \
        PersistentDenseGemmKernel
    from ..cute_dsl_kernels.blackwell.moe_as_dense_gemm.fc1 import \
        Sm100BlockScaledPersistentDenseGemmKernel as DenseGemmSwigluKernel
    from ..cute_dsl_kernels.blackwell.top_k.filtered_top_k_decode_varlen import \
        FilteredTopKKernelVarlenDecode
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import \
        STATE_SIZE as DISTRIBUTED_TOPK_STATE_SIZE
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import \
        SinglePassMultiCTARadixTopKKernel
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import \
        STATE_SIZE as CLUSTER_TOPK_STATE_SIZE
    from ..cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import (
        SinglePassMultiCTARadixTopKClusterKernel, _query_max_cluster_size)
    from ..cute_dsl_kernels.blackwell.utils import make_ptr

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

            # Optimize swap_ab candidates based on M and N alignment
            # swap_ab=False → C is N-major → requires N%8==0 (BF16: 128 bits / 16 bits = 8)
            # swap_ab=True  → C is M-major → requires M%8==0
            m_aligned = (m % 8 == 0)
            n_aligned = (n % 8 == 0)

            if not m_aligned and not n_aligned:
                logger.debug(
                    f"CuteDSL: Neither M={m} nor N={n} meets 16-byte alignment "
                    f"(M%8={m%8}, N%8={n%8}). No valid C layout. Skipping all tactics."
                )
                return []

            # Only test swap_ab values that satisfy alignment
            swap_ab_candidates = []
            if n_aligned:
                swap_ab_candidates.append(False)  # N-major layout
            if m_aligned:
                swap_ab_candidates.append(True)  # M-major layout

            logger.debug(
                f"CuteDSL: M={m}(aligned={m_aligned}), N={n}(aligned={n_aligned}), K={real_k}(aligned=True). "
                f"Testing swap_ab={swap_ab_candidates}")

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
            swap_ab_candidates = [True, False]
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

            Returns:
                torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
            """
            sf_vec_size = 16

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = tactic
            else:
                # fallback to default tactic
                mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = [
                    (128, 128),
                    (1, 1),
                    False,
                    False,
                ]

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
                         use_prefetch, self.use_tvm_ffi)
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
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
        kernel_class = Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def __init__(self, output_dtype: torch.dtype, use_tvm_ffi: bool = True):
            super().__init__()

            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL NVFP4 SwiGLU only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.use_tvm_ffi = use_tvm_ffi

        def unique_id(self):
            return (self.output_dtype, self.use_tvm_ffi)

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

            # SwiGLU output has N/2 columns — check alignment for C
            n_out = n // 2
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

            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
            n_out = n // 2  # SwiGLU halves the N dimension

            # Allocate output tensor with halved N dimension
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

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

            # No swap_ab for SwiGLU — always use A as activation, B as weight
            kernel_m = m
            kernel_n = n  # Full B width (passed to wrapper, which creates B with n columns)
            kernel_sf_m = sf_m
            kernel_sf_n = sf_n

            cache_key = (sf_vec_size, mma_tiler_mn, cluster_shape_mn,
                         use_prefetch, self.use_tvm_ffi)
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
                    stream = cute.runtime.make_fake_stream(
                        use_tvm_ffi_env_stream=True)

                gemm = self.__class__.kernel_class(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    True,  # vectorized_f32
                    use_prefetch,
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Launch kernel
            if self.use_tvm_ffi:
                compiled_gemm(
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
                )
            else:
                compiled_gemm(
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
                )

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

    class CuteDSLNVFP4SwigluFP4OutBlackwellRunner(TunableRunner):
        """Runner for dense GEMM + SwiGLU fusion with FP4 output on Blackwell.

        Same as CuteDSLNVFP4SwigluBlackwellRunner but produces Float4E2M1FN
        output with scale factors (SFC quantization), eliminating the bf16→fp4
        requantization between FC1 and FC2.
        """
        kernel_class = Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel
        kernel_cache = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cold_l2_cache=True,
            distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
        )

        def __init__(self, use_tvm_ffi: bool = True):
            super().__init__()
            self.use_tvm_ffi = use_tvm_ffi

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
            # Same tactic search as BF16 runner but with FP4 C dtype
            a, b, a_sf, b_sf, alpha, global_sf = inputs
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

            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor, global_sf_tensor = inputs
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
            n_out = n // 2  # SwiGLU halves the N dimension

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

            cache_key = (sf_vec_size, mma_tiler_mn, cluster_shape_mn,
                         use_prefetch, self.use_tvm_ffi, 'fp4out')
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
                )
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi else "--opt-level 2",
                )

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Launch kernel
            if self.use_tvm_ffi:
                compiled_gemm(
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
                )
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
                alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                norm_const_cute_tensor = cute.runtime.from_dlpack(
                    global_sf_tensor)

                torch_stream = torch.cuda.current_stream()
                stream = cuda.CUstream(torch_stream.cuda_stream)

                compiled_gemm(
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
                )

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
        m = mat_a.shape[0]
        n = mat_b.shape[-2]
        n_out = n // 2
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
            l = sum(bi.size(0) for bi in b_list)
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
            l, n = b.size(0), b.size(1)
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
                     scaling_vector_size: int = 16,
                     b_tensor_l_sizes: Optional[Tuple[int, ...]] = None):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size

            assert output_dtype == torch.bfloat16
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size
            self.b_tensor_l_sizes = b_tensor_l_sizes

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
                self.b_tensor_l_sizes,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            a, b_list, *_ = inputs
            if not isinstance(b_list, (list, tuple)):
                raise TypeError("weight must be a list of tensors")
            m, k = a.size(0), a.size(1) * 2
            l = sum(bi.size(0) for bi in b_list)
            n = b_list[0].size(1)

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
            a, b_list, a_sf, b_sf_list, alpha_list, c, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales = inputs
            if not isinstance(b_list, (list, tuple)):
                raise TypeError("weight must be a list of tensors")
            if not isinstance(b_sf_list, (list, tuple)):
                raise TypeError("weight_scale must be a list of tensors")
            if not isinstance(alpha_list, (list, tuple)):
                raise TypeError("alpha must be a list of tensors")
            assert len(b_list) == len(b_sf_list) == len(alpha_list)
            b_tensor_l_sizes = tuple(bi.size(0) for bi in b_list)

            b0 = b_list[0]
            b_sf0 = b_sf_list[0]
            alpha0 = alpha_list[0]
            assert a.dtype == torch.float4_e2m1fn_x2
            assert a.dim() == 2
            assert b0.dtype == torch.float4_e2m1fn_x2
            assert b0.dim() == 3
            assert a_sf.dtype == torch.uint8
            assert a_sf.dim() == 1
            assert b_sf0.dtype == torch.uint8
            assert b_sf0.dim() == 3
            assert alpha0.dtype == torch.float32
            assert alpha0.dim() == 1

            m, k = a.size(0), a.size(1) * 2
            sum(bi.size(0) for bi in b_list)
            n = b0.size(1)
            scale_k = k // self.scaling_vector_size
            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert b0.size(2) * 2 == k
            assert a_sf.size(0) == m * scale_k
            for bi, bsfi, ai in zip(b_list, b_sf_list, alpha_list):
                assert bi.size(1) == n
                assert bi.size(2) * 2 == k
                assert bsfi.size(0) == bi.size(0)
                assert bsfi.size(1) == n
                assert bsfi.size(2) == scale_k
                assert ai.size(0) == bi.size(0)

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
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
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
            token_final_scales_ptr = make_ptr(cutlass.Float32,
                                              token_final_scales.data_ptr(),
                                              cute.AddressSpace.gmem)
            c_ptr = make_ptr(cutlass.BFloat16,
                             c.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=16)

            b_ptr = tuple(
                make_ptr(cutlass.Float4E2M1FN,
                         bi.data_ptr(),
                         cute.AddressSpace.gmem,
                         assumed_align=32) for bi in b_list)
            b_sf_ptr = tuple(
                make_ptr(cutlass.Float8E4M3FN,
                         bsfi.data_ptr(),
                         cute.AddressSpace.gmem,
                         assumed_align=16) for bsfi in b_sf_list)
            alpha_ptr = tuple(
                make_ptr(cutlass.Float32, ai.data_ptr(), cute.AddressSpace.gmem)
                for ai in alpha_list)

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
                         cluster_shape_mn, raster_along_m, b_tensor_l_sizes)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    use_blkred=True,
                    raster_along_m=raster_along_m,
                    b_tensor_l_sizes=b_tensor_l_sizes,
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
        weight: List[torch.Tensor],
        input_scale: torch.Tensor,
        weight_scale: List[torch.Tensor],
        alpha: List[torch.Tensor],
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

        b_tensor_l_sizes = tuple(w.size(0)
                                 for w in weight) if len(weight) > 1 else None
        runner = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, output_dtype, scaling_vector_size, b_tensor_l_sizes)

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
        weight: List[torch.Tensor],
        input_scale: torch.Tensor,
        weight_scale: List[torch.Tensor],
        alpha: List[torch.Tensor],
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
        n = weight[0].size(1)
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
        weight: List[torch.Tensor],
        input_scale: torch.Tensor,
        weight_scale: List[torch.Tensor],
        alpha: List[torch.Tensor],
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
        n = weight[0].size(1)
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
                     scaling_vector_size: int = 16):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size
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
            l, n = b.size(0), b.size(1)

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
            l, n = b.size(0), b.size(1)
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
                         cluster_shape_mn)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tuner = AutoTuner.get()

        runner = Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, scaling_vector_size)
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

    class Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner(
            TunableRunner):
        kernel_class = BlockScaledContiguousGatherGroupedGemmKernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        # Maximum number of B tensors supported (must match kernel's MAX_B_TENSORS)
        MAX_B_TENSORS = 4

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     tile_size: int,
                     scaling_vector_size: int = 16,
                     b_tensor_l_sizes: Optional[Tuple[int, ...]] = None):
            """Initialize the runner.

            Args:
                b_tensor_l_sizes: Tuple of L sizes for each B tensor in multi-B mode.
                    None for single-B mode. Used for kernel cache key.
            """
            super().__init__()
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
            self.b_tensor_l_sizes = b_tensor_l_sizes

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
                self.b_tensor_l_sizes,
            )

        def get_valid_tactics(
            self,
            inputs: List,
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            # Tuning uses layout: a, b_list, a_sf, b_sf_list, alpha_list, ...
            a = inputs[0]
            b_list = inputs[1]  # List of B tensors
            permuted_idx_to_expanded_idx = inputs[7]
            # m is the permuted size from permuted_idx_to_expanded_idx, not from a
            m = permuted_idx_to_expanded_idx.size(0)
            k = a.size(1) * 2
            l = sum(bi.size(0) for bi in b_list)
            n = b_list[0].size(1)

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
                # a, b_list, a_sf, b_sf_list, alpha_list, tile_idx, tile_mn_limit, permuted_idx, ...
                # Constraint indices adjusted for list inputs at positions 1, 3, 4
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
            """Forward pass supporting both single tensor and list inputs.

            Input layout (positions 1, 3, 4 are lists for multi-B support):
                0: a                               - tensor
                1: b_list                          - list of tensors
                2: a_sf                            - tensor
                3: b_sf_list                       - list of tensors
                4: alpha_list                      - list of tensors
                5: tile_idx_to_group_idx           - tensor
                6: tile_idx_to_mn_limit            - tensor
                7: permuted_idx_to_expanded_idx    - tensor
                8: num_non_exiting_tiles           - tensor
                9: global_sf                       - tensor
            """
            a, b_list, a_sf, b_sf_list, alpha_list, tile_idx_to_group_idx, \
                tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, \
                num_non_exiting_tiles, global_sf = inputs

            b_tensor_l_sizes = tuple(bi.size(0) for bi in b_list)

            b0 = b_list[0]  # Use first B for shape inference

            # Verify input dtypes and dimensions
            assert a.dtype == torch.float4_e2m1fn_x2
            assert a.dim() == 2
            assert b0.dtype == torch.float4_e2m1fn_x2
            assert b0.dim() == 3
            assert a_sf.dtype == torch.uint8
            assert a_sf.dim() == 2
            assert b_sf_list[0].dtype == torch.uint8
            assert b_sf_list[0].dim() == 3
            assert alpha_list[0].dtype == torch.float32
            assert alpha_list[0].dim() == 1

            # a.size(0) is orig_m (original input size before gather)
            # permuted_idx_to_expanded_idx.size(0) is m (permuted size after gather)
            orig_m, k = a.size(0), a.size(1) * 2
            m = permuted_idx_to_expanded_idx.size(0)
            n = b0.size(1)
            sum(bi.size(0) for bi in b_list)
            scale_k = k // self.scaling_vector_size
            interm_size = n // 2

            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert n % (self.scaling_vector_size * 4 * 2) == 0
            assert b0.size(2) * 2 == k
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

            # Create common pointers
            a_ptr = make_ptr(cutlass.Float4E2M1FN,
                             a.data_ptr(),
                             cute.AddressSpace.gmem,
                             assumed_align=32)
            a_sf_ptr = make_ptr(cutlass.Float8E4M3FN,
                                a_sf.data_ptr(),
                                cute.AddressSpace.gmem,
                                assumed_align=16)
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

            b_ptr = tuple(
                make_ptr(cutlass.Float4E2M1FN,
                         bi.data_ptr(),
                         cute.AddressSpace.gmem,
                         assumed_align=32) for bi in b_list)
            b_sf_ptr = tuple(
                make_ptr(cutlass.Float8E4M3FN,
                         bsfi.data_ptr(),
                         cute.AddressSpace.gmem,
                         assumed_align=16) for bsfi in b_sf_list)
            alpha_ptr = tuple(
                make_ptr(cutlass.Float32, ai.data_ptr(), cute.AddressSpace.gmem)
                for ai in alpha_list)

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
                         b_tensor_l_sizes)

            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
                    topk=self.top_k,
                    raster_along_m=raster_along_m,
                    b_tensor_l_sizes=b_tensor_l_sizes,
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
            ]

            compiled_gemm(*exec_args, stream=stream)

            return c, c_sf

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b(
        input: torch.Tensor,
        weight: List[torch.Tensor],
        input_scale: torch.Tensor,
        weight_scale: List[torch.Tensor],
        alpha: List[torch.Tensor],
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL-based NVFP4 gather grouped GEMM with SwiGLU fusion (multi-B list interface).

        Args:
            weight: List of B tensors. Single-B mode: [b], multi-B mode: [b0, b1, ...].
            weight_scale: List of scale tensors, matching weight.
            alpha: List of alpha tensors, matching weight.
        """
        tuner = AutoTuner.get()

        b_tensor_l_sizes = tuple(w.size(0) for w in weight)

        runner = Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, scaling_vector_size, b_tensor_l_sizes)
        inputs = [
            input, weight, input_scale, weight_scale, alpha,
            tile_idx_to_group_idx, tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )

        # Call forward with inputs list
        output = runner.forward(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b")
    def _fake_multi_b(
        input: torch.Tensor,
        weight: List[torch.Tensor],
        input_scale: torch.Tensor,
        weight_scale: List[torch.Tensor],
        alpha: List[torch.Tensor],
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = permuted_idx_to_expanded_idx.size(0)
        n = weight[0].size(1)
        interm_size = n // 2
        output = torch.empty(m,
                             interm_size // 2,
                             dtype=input.dtype,
                             device=input.device)
        output_scale = torch.empty(m * interm_size // scaling_vector_size,
                                   dtype=input_scale.dtype,
                                   device=input_scale.device)
        return output, output_scale

    @torch.library.custom_op(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell",
        mutates_args=(),
        device_types="cuda")
    def cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL-based NVFP4 gather grouped GEMM with SwiGLU fusion (single-B tensor interface).

        Thin wrapper: wraps single tensors into lists and calls
        cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b.
        """
        return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell_multi_b(
            input,
            [weight],
            input_scale,
            [weight_scale],
            [alpha],
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            global_sf,
            num_experts,
            top_k,
            num_local_experts,
            local_expert_offset,
            tile_size,
            scaling_vector_size,
        )

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell")
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = permuted_idx_to_expanded_idx.size(0)
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
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

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 1, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 2, fp8_scale_infer_shape), ),
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
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
            l = 1  # dense GEMM

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
            l = 1  # dense GEMM
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
        l = 1  # dense GEMM

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
        ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
            """Return valid (mma_tiler_mn, cluster_shape_mn) combinations."""
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
            l = 1  # dense GEMM

            # Define candidates together
            mma_tiler_mn_candidates = [(128, 64), (128, 128), (128, 256)]
            cluster_shape_mn_candidates = [(1, 1), (1, 2), (1, 4)]

            # Map torch dtype to cutlass dtype
            if self.output_dtype not in self._CUTLASS_DTYPE_MAP:
                raise ValueError(
                    f"Unsupported output_dtype {self.output_dtype} for FC2 DenseGEMM runner"
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
                        0, 0, get_last_power_of_2_num_tokens_buckets,
                        last_positive_power_of_2), ),
                    constraint_specs=(
                        ConstraintSpec(2, 0, fp4_scale_infer_shape),
                        ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),
                    ),
                    use_cold_l2_cache=True,
                    tune_max_num_tokens=256,
                    distributed_tuning_strategy=DistributedTuningStrategy.
                    PARALLEL,
                )
            return self.tuning_config_cache[key]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        ) -> torch.Tensor:
            """Execute the dense GEMM FC2.

            Args:
                inputs: [a, b, a_sf, b_sf, alpha_scale]
                tactic: ((mma_m, mma_n), (cluster_m, cluster_n))

            Returns:
                Output tensor
            """
            a, b, a_sf, b_sf, alpha_scale = inputs[:5]

            # Get dimensions
            # a: [m, k//2] (fp4 packed), b: [n, k//2]
            m = a.shape[0]
            k = a.shape[1] * 2  # fp4 packed in k dimension
            n = b.shape[0]
            l = 1  # dense GEMM

            # Default tactic if not provided
            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn = tactic
            else:
                mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 1)

            # Allocate output tensor
            c_dtype = self.output_dtype
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
                         load_balance, large_occupancy).

        Note:
            - Requires Blackwell architecture (SM100+)
            - Maximum tested top_k is 2048 (see kernel documentation for larger values)
            - Supports fp16, bf16, and fp32 dtypes
            - Automatically selects occupancy optimization based on batch size
        """
        kernel_cache = dict()
        buffers = get_memory_buffers()

        @classmethod
        def _compile(cls, dtype, bucketed_num_cols, top_k, next_n, return_val,
                     num_copy_bits, load_balance, large_occupancy):
            """Compile and cache a single-CTA top-k kernel for the given config."""
            key = (
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                load_balance,
                large_occupancy,
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
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, cute.sym_int(), n_cols),
                stride_order=(2, 1, 0),
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

            filtered_topk_func = FilteredTopKKernelVarlenDecode(
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                large_occupancy=large_occupancy,
                num_sms=_get_num_sms(),
            )
            if load_balance:
                g_global_counter_fake = cute.runtime.make_fake_compact_tensor(
                    cutlass.Int32, (1, ), stride_order=(0, ))
            else:
                g_global_counter_fake = None
            compiled_kernel = cute.compile(
                filtered_topk_func,
                input_fake,
                None,  # indices_fake
                buffer_fake,
                g_global_counter_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                enable_persistent_dynamic_scheduling=load_balance,
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
            load_balance: bool = False,
            output_indices: Optional[torch.Tensor] = None,
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
                load_balance,
                large_occupancy,
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

            # Prepare buffer
            # extra buffer: num_rows * buffer_numbers * num_cols * 4 bytes
            # fp32: up to 256 MB (256 * 2 * 262144 * 4)
            # fp16/bf16: up to 128 MB (256 * 1 * 262144 * 4)
            if dtype == cutlass.Float32:
                buffer_numbers = 2
            else:
                buffer_numbers = 1
            buffer_torch = cls.buffers.get_buffer(
                [num_rows, buffer_numbers, bucketed_num_cols],
                torch.int32,
                buffer_name="single_cta_buffer",
                reserve_buffer=reserve)
            buffer_torch = buffer_torch[:, :, :num_cols]
            # Prepare global counter for persistent dynamic scheduling
            if load_balance:
                g_global_counter_torch = cls.buffers.get_buffer(
                    [1],
                    torch.int32,
                    buffer_name="single_cta_g_global_counter",
                    reserve_buffer=reserve)
                g_global_counter_torch.zero_()
            else:
                g_global_counter_torch = None

            # Execute kernel (TVM FFI uses env stream automatically)
            compiled_kernel(
                input_values,
                None,  # indices
                buffer_torch,
                g_global_counter_torch,
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
        load_balance: bool = False,
    ) -> torch.Tensor:
        """CuteDSL-based Top-K selection optimized for Blackwell decode phase.

        Args:
            input_values: Input logits tensor [batch_size * next_n, vocab_size]
            seq_lens: Sequence lengths for each batch [batch_size]
            top_k: Number of top elements to select (max 2048)
            next_n: Number of candidates per sequence (for speculative decoding)
            num_copy_bits: Number of bits for vectorized memory copy (128 or 256)
            load_balance: Enable persistent dynamic scheduling for load balancing

        Returns:
            indices: Top-k indices [batch_size * next_n, top_k]

        Note:
            This function requires Blackwell architecture (SM100+) and CuTE DSL support.
            Maximum supported top_k is 2048.
        """
        # Validate SM version
        sm_version = get_sm_version()
        if sm_version < 100:
            raise ValueError(
                f"CuTE DSL top-k requires Blackwell (SM100+), but got SM {sm_version}. "
                "Use standard top-k implementation for older architectures.")

        # Validate inputs
        if top_k <= 0 or top_k > 2048:
            raise ValueError(
                f"top_k must be in range [1, 2048], got {top_k}. "
                "Maximum supported top_k is 2048 for Blackwell architecture.")

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
            load_balance=load_balance,
        )
        return indices

    @torch.library.register_fake("trtllm::cute_dsl_topk_decode_blackwell")
    def _(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        load_balance: bool = False,
    ):
        num_rows = input_values.shape[0]
        input_values.dtype

        # Create output tensors matching the custom op return signature: (values, indices)
        indices = input_values.new_empty((num_rows, top_k), dtype=torch.int32)
        return indices

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
                     load_balance,
                     large_occupancy,
                     chunk_size_per_cta,
                     num_ctas_per_row,
                     dynamic=False):
            """Compile and cache multi-CTA top-k kernels for the given config."""
            key = (
                dtype,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                load_balance,
                large_occupancy,
                chunk_size_per_cta,
                num_ctas_per_row,
                dynamic,
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
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
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
            )
            compiled_kernel_first = cute.compile(
                filtered_topk_func_first,
                input_fake,
                None,  # indices_fake
                buffer_fake,
                None,  # g_global_counter_fake
                seqlen_fake,
                first_kernel_output_indices_fake,
                first_kernel_output_values_fake,
                stream=fake_stream,
                enable_persistent_dynamic_scheduling=load_balance,
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
            )
            compiled_kernel_second = cute.compile(
                filtered_topk_func_second,
                input_fake,
                indices_fake,
                buffer_fake,
                None,  # g_global_counter_fake
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                enable_persistent_dynamic_scheduling=load_balance,
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
            dynamic: bool = True,
            output_indices: Optional[torch.Tensor] = None,
        ):
            """Execute multi-CTA filtered top-k selection on input logits."""
            torch_dtype = input_values.dtype
            dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
            num_rows, num_cols = input_values.shape

            num_sms = _get_num_sms()
            large_occupancy = num_rows > num_sms
            load_balance = False

            num_ctas_per_row = math.ceil(num_cols / chunk_size_per_cta)
            merge_cols = num_ctas_per_row * top_k

            key = (
                dtype,
                top_k,
                next_n,
                return_val,
                num_copy_bits,
                load_balance,
                large_occupancy,
                chunk_size_per_cta,
                num_ctas_per_row,
                dynamic,
            )
            cls._compile(*key)
            compiled_kernel_first, compiled_kernel_second = \
                cls.kernel_cache[key]
            reserve = torch.cuda.is_current_stream_capturing()

            if dtype == cutlass.Float32:
                buffer_numbers = 2
            else:
                buffer_numbers = 1

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

            # Shared buffer for both kernels (they run sequentially)
            buffer_dim2 = max(chunk_size_per_cta, merge_cols)
            buffer_torch = cls.buffers.get_buffer(
                [num_rows * num_ctas_per_row, buffer_numbers, buffer_dim2],
                torch.int32,
                buffer_name="multi_cta_buffer",
                reserve_buffer=reserve)

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
                None,  # g_global_counter_torch
                seq_lens,
                first_output_indices,
                first_output_values,
            )

            # Execute second kernel: merge partial results
            compiled_kernel_second(
                first_output_values,
                first_output_indices,
                buffer_torch,
                None,  # g_global_counter_torch
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
            top_k: Number of top elements to select (max 2048)
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
        if top_k <= 0 or top_k > 2048:
            raise ValueError(
                f"top_k must be in range [1, 2048], got {top_k}. "
                "Maximum supported top_k is 2048 for Blackwell architecture.")

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
    ) -> None:
        """Unified CuTE DSL Top-K that auto-selects single-CTA or multi-CTA (2-pass multi-CTA) or
        single-pass multi-CTA. When single_pass_multi_cta=True, it selects between single-CTA
        and multi-CTA (1-pass multi-CTA). When single_pass_multi_cta=False, it selects between
        single-CTA and multi-CTA (2-pass multi-CTA).

        Writes results directly into the pre-allocated ``output_indices`` buffer.

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
            top_k: Number of top elements to select (max 2048)
            next_n: Number of candidates per sequence (for speculative decoding)
            num_copy_bits: Number of bits for vectorized memory copy (128 or 256)
            dynamic: Use dynamic multi-CTA scheduling (for 2-pass multi-CTA)
            single_pass_multi_cta: Use single-pass multi-CTA radix top-k
            single_pass_multi_cta_cluster: Force cluster-accelerated variant
                (only effective when single_pass_multi_cta=True)
        """
        num_rows = input_values.shape[0]
        num_tokens = input_values.shape[1]

        if single_pass_multi_cta:
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
                    output_indices=output_indices,
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
                    dynamic=dynamic,
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
                    output_indices=output_indices,
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
    ) -> None:
        return None

    def warmup_cute_dsl_indexer_topk(
        dtype: torch.dtype,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        min_seq_len_log2: int = 10,
        max_seq_len_log2: int = 18,
        single_pass_multi_cta: bool = False,
        single_pass_multi_cta_cluster: bool = False,
    ) -> None:
        """Pre-compile all CuTE DSL top-k kernel variants for every
        power-of-2 bucketed_num_cols in [2^min_seq_len_log2, 2^max_seq_len_log2].

        Because the kernel compilation key uses
        ``bucketed_num_cols = next_positive_power_of_2(num_cols)``, only
        a small number of distinct kernels are needed regardless of the
        actual ``max_seq_len``.  This allows warmup to be called at model
        init time without knowing the runtime ``max_seq_len``.

        Must be called before CUDA Graph capture so that JIT compilation
        does not occur during capture/replay.

        Args:
            dtype: Data type of the logits (e.g. torch.bfloat16).
            top_k: Number of top elements to select.
            next_n: Number of candidates per sequence (speculative decoding).
            num_copy_bits: Vectorized memory copy width (128 or 256).
            min_seq_len_log2: Log2 of minimum bucketed_num_cols (default 10 → 1024).
            max_seq_len_log2: Log2 of maximum bucketed_num_cols (default 18 → 262144).
            single_pass_multi_cta: Use single-pass multi-CTA radix top-k
                dispatch path instead of the legacy two-pass kernels.
            single_pass_multi_cta_cluster: Force cluster-accelerated variant
                (only effective when single_pass_multi_cta=True).
        """
        cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[dtype]
        return_val = False
        chunk_size_per_cta = 16384

        # Multi-CTA vocab thresholds by dtype
        if dtype == torch.float32:
            multi_cta_threshold = 65536
        else:
            multi_cta_threshold = 131072

        # SingleCTA: enumerate all power-of-2 bucketed_num_cols
        for log2_n in range(min_seq_len_log2, max_seq_len_log2 + 1):
            bucketed_num_cols = 1 << log2_n
            for large_occupancy in (False, True):
                CuteDSLTopKDecodeSingleCTARunner._compile(
                    cutlass_dtype,
                    bucketed_num_cols,
                    top_k,
                    next_n,
                    return_val,
                    num_copy_bits,
                    load_balance=False,
                    large_occupancy=large_occupancy,
                )

        if single_pass_multi_cta:
            # Single-pass multi-CTA: enumerate all (chunk_size, ctas_per_group)
            # pairs.  chunk_size is snapped to power-of-2 (+ max_chunk clamp),
            # so the set of possible values is small and deterministic.
            num_sms = _get_num_sms()
            possible_chunks = CuteDSLTopKDecodeSinglePassMultiCTARunner._get_possible_chunk_sizes(
                cutlass_dtype, num_copy_bits)
            max_chunk, vec_size = CuteDSLTopKDecodeSinglePassMultiCTARunner._compute_max_chunk(
                cutlass_dtype, num_copy_bits)
            single_pass_multi_cta_configs = set()
            for cs in possible_chunks:
                for log2_n in range(min_seq_len_log2, max_seq_len_log2 + 1):
                    num_cols = 1 << log2_n
                    ctas = math.ceil(num_cols / cs)
                    if ctas >= 1:
                        single_pass_multi_cta_configs.add((cs, ctas))
            # Also cover FlashInfer-style fallback path (large batch):
            # ctas_per_group = ceil(num_cols / max_chunk), chunk_size aligned
            for log2_n in range(min_seq_len_log2, max_seq_len_log2 + 1):
                num_cols = 1 << log2_n
                ctas = math.ceil(num_cols / max_chunk)
                if ctas >= 1:
                    cs = math.ceil(num_cols / ctas)
                    cs = ((cs + vec_size - 1) // vec_size) * vec_size
                    if cs > max_chunk:
                        cs = max_chunk
                    single_pass_multi_cta_configs.add((cs, ctas))
            for cs, ctas in sorted(single_pass_multi_cta_configs):
                CuteDSLTopKDecodeSinglePassMultiCTARunner._compile(
                    cutlass_dtype, cs, top_k, next_n, num_copy_bits, ctas,
                    num_sms, return_val)

            # Cluster variant: enumerate configs using the cluster runner's
            # _get_chunk_config (which clamps to hw max cluster size).
            cluster_configs = set()
            if single_pass_multi_cta_cluster:
                for log2_n in range(min_seq_len_log2, max_seq_len_log2 + 1):
                    num_cols = 1 << log2_n
                    for nr in [1, 4, 16, 64, 256]:
                        cfg = CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner._get_chunk_config(
                            cutlass_dtype,
                            num_cols,
                            num_copy_bits=num_copy_bits,
                            num_rows=nr)
                        if cfg[0] is not None:
                            cluster_configs.add((cfg[0], cfg[1]))
                for cs, ctas in sorted(cluster_configs):
                    CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner._compile(
                        cutlass_dtype, cs, top_k, next_n, num_copy_bits, ctas,
                        num_sms, return_val)

            multi_cta_info = (
                f"SinglePassMultiCTA ({len(single_pass_multi_cta_configs)} configs"
                f", cluster {len(cluster_configs)} configs)")
        else:
            # 2-pass MultiCTA: enumerate all possible num_ctas_per_row values
            # num_ctas_per_row = ceil(num_cols / chunk_size_per_cta)
            # fp32: num_cols in [65536, 262144] → num_ctas_per_row in [4, 16]
            # fp16/bf16: num_cols in [131072, 262144] → num_ctas_per_row in [8, 16]
            min_ctas = math.ceil(multi_cta_threshold / chunk_size_per_cta)
            max_ctas = math.ceil((1 << max_seq_len_log2) / chunk_size_per_cta)
            for num_ctas_per_row in range(min_ctas, max_ctas + 1):
                for large_occupancy in (False, True):
                    CuteDSLTopKDecodeMultiCTARunner._compile(
                        cutlass_dtype,
                        top_k,
                        next_n,
                        return_val,
                        num_copy_bits,
                        load_balance=False,
                        large_occupancy=large_occupancy,
                        chunk_size_per_cta=chunk_size_per_cta,
                        num_ctas_per_row=num_ctas_per_row,
                        dynamic=True,
                    )
            multi_cta_info = (
                f"MultiCTA num_ctas_per_row=[{min_ctas}..{max_ctas}]")

        logger.info(
            f"Warmed up CuTE DSL indexer top-k kernels: dtype={dtype}, "
            f"SingleCTA bucketed_num_cols=[2^{min_seq_len_log2}..2^{max_seq_len_log2}], "
            f"{multi_cta_info}, top_k={top_k}, next_n={next_n}")

    # ------------------------------------------------------------------ #
    #  CuTE DSL FP8 Paged MQA Logits (Blackwell SM100)                   #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.paged_mqa_logits import FP8MQALogitsKernel

    def _check_fp8_paged_mqa_logits_dtypes(q, kv_fused, weights, context_lens,
                                           block_table, schedule_meta,
                                           epi_dtype, acc_dtype, output_dtype):
        errs = []
        if q.dtype != torch.float8_e4m3fn:
            errs.append(f"q must be float8_e4m3fn, got {q.dtype}")
        if kv_fused.dtype != torch.uint8:
            errs.append(f"kv_fused must be uint8, got {kv_fused.dtype}")
        # TODO: update to (torch.float32, torch.float16) once fp16 weights
        # are validated end-to-end and the in-kernel .half() conversion is removed.
        if weights.dtype != torch.float32:
            errs.append(f"weights must be float32, got {weights.dtype}")
        if context_lens.dtype != torch.int32:
            errs.append(f"context_lens must be int32, got {context_lens.dtype}")
        if block_table.dtype != torch.int32:
            errs.append(f"block_table must be int32, got {block_table.dtype}")
        if schedule_meta.dtype != torch.int32:
            errs.append(
                f"schedule_meta must be int32, got {schedule_meta.dtype}")
        for name, dt in [("epi_dtype", epi_dtype), ("acc_dtype", acc_dtype),
                         ("output_dtype", output_dtype)]:
            if dt not in (torch.float16, torch.float32):
                errs.append(f"{name} must be float16 or float32, got {dt}")
        if errs:
            raise ValueError("FP8 Paged MQA Logits dtype errors:\n  " +
                             "\n  ".join(errs))

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

            kv_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (sym_num_phys_blocks, block_bytes),
                stride_order=(1, 0))

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
            logits = torch.empty(
                (B * next_n, aligned_max_ctx),
                device=q.device,
                dtype=output_dtype,
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
        _check_fp8_paged_mqa_logits_dtypes(q, kv_fused, weights, context_lens,
                                           block_table, schedule_meta,
                                           epi_dtype, acc_dtype, output_dtype)
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

    # ======================================================================
    # BF16 Dense Persistent BMM (CuTe DSL) for Blackwell
    # ======================================================================

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
