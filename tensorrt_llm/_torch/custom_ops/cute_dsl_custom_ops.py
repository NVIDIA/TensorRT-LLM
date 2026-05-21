# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
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
from ..utils import (ActivationType, deep_gemm_gen_tuning_buckets,
                     fp4_scale_infer_shape, fp8_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     is_gated_activation, last_positive_power_of_2,
                     next_positive_power_of_2)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

# Torch schema parsing rejects ``inf`` as a default value.
SWIGLU_LIMIT_DISABLED = -1.0


def _canonicalize_swiglu_limit(swiglu_limit: float) -> float:
    return float("inf") if swiglu_limit < 0 else swiglu_limit


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
    """
    # Override: use permuted_idx_to_expanded_idx for shape inference
    IDX_PERMUTED_IDX_TO_EXPANDED_IDX = 7
    IDX_SHAPE_INFER = IDX_PERMUTED_IDX_TO_EXPANDED_IDX

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pre-hook for gather-based activation fusion kernel.

        Generates:
            - tile_idx_to_group_idx
            - tile_idx_to_mn_limit
            - permuted_idx_to_expanded_idx (for gather operation)
            - num_non_exiting_tiles
        """
        a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, \
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf = inputs
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
        return (a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx,
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
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
                    options=f"--opt-level 2 --enable-tvm-ffi"
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
                     swiglu_limit: float = float("inf")):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size
            self.scaling_vector_size = scaling_vector_size
            self.swiglu_limit = swiglu_limit

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
                self.swiglu_limit,
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
                         cluster_shape_mn, self.swiglu_limit)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
                    swiglu_limit=self.swiglu_limit,
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
        swiglu_limit: float = SWIGLU_LIMIT_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tuner = AutoTuner.get()
        swiglu_limit = _canonicalize_swiglu_limit(swiglu_limit)

        runner = Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, scaling_vector_size, swiglu_limit)
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
        swiglu_limit: float = SWIGLU_LIMIT_DISABLED,
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
                     swiglu_limit: float = float("inf")):
            """Initialize the runner.

            Args:
                activation_type: ``ActivationType`` for the fused epilogue. Only
                    ``Swiglu`` (gated) and ``Relu2`` (non-gated) are supported.
                swiglu_limit: Uniform clamp limit for SwiGLU. ``+inf`` disables clamp.
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
            self.swiglu_limit = swiglu_limit

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
                self.swiglu_limit,
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
                         self.activation_type, self.swiglu_limit)

            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
                    topk=self.top_k,
                    raster_along_m=raster_along_m,
                    activation_type=self.activation_type,
                    swiglu_limit=self.swiglu_limit,
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
        swiglu_limit: float = SWIGLU_LIMIT_DISABLED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CuteDSL-based NVFP4 gather grouped GEMM with activation fusion.

        Supports ``ActivationType.Swiglu`` (gated) and ``ActivationType.Relu2``
        (non-gated) epilogues; other ``ActivationType`` values raise an
        assertion in the runner.
        """
        tuner = AutoTuner.get()
        swiglu_limit = _canonicalize_swiglu_limit(swiglu_limit)

        runner = Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner(
            num_experts,
            top_k,
            num_local_experts,
            local_expert_offset,
            tile_size,
            scaling_vector_size,
            activation_type=ActivationType(activation_type),
            swiglu_limit=swiglu_limit)
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
        swiglu_limit: float = SWIGLU_LIMIT_DISABLED,
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

    # ------------------------------------------------------------------ #
    #  CuTe DSL GVR Top-K Decode                                         #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import \
        GvrTopKKernel as _GvrTopKKernel

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
            """Pick T / V / min_blocks_per_mp tuning knobs shared by
            single-CTA / sort and LB compile paths. Returned keys match
            ``_compile`` / ``_compile_lb`` param names for ``**tuning``
            spreading.
            """
            enable_unroll_4 = True
            enable_phase3_unroll = True
            use_constant_hint = False

            # T=1024 needs 1 CTA/SM grid AND enough per-CTA vec work.
            # Under graph capture, raise the half-prec bar so a small
            # capture-N doesn't force T=1024 on small-N replays
            # (~14-16% regression).
            if max_seq_len is not None and torch_dtype != torch.float32:
                n_thresh_t = 131072
            else:
                n_thresh_t = 65536
            num_threads_per_block = (1024 if
                                     (num_rows <= num_sms
                                      and N_per_cta >= n_thresh_t) else 512)
            # V=256-bit only helps fp32 at large N. Half-prec cvt
            # doubles reg pressure (5-11% loss at K=512/1024). Caller
            # must hand a contiguous (32B-aligned) tensor — torch.empty
            # / row slices satisfy this; column / stride-padded layouts
            # may not.
            use_256bit_load = (torch_dtype == torch.float32
                               and N_per_cta >= 16384)
            if use_256bit_load:
                assert data_ptr % 32 == 0, (
                    f"use_256bit_load=True requires 32B-aligned "
                    f"logits.data_ptr(), got {data_ptr} % 32 = "
                    f"{data_ptr % 32}.")
            # Warp-parallel reduce only pays at 32-warp (T=1024).
            enable_warp_parallel_reduce = num_threads_per_block == 1024

            # min_blocks_per_mp: reg-vs-occupancy 3-tier. Half-prec
            # prefers extra CTA/SM (cvt-ILP fits in 40 regs); fp32
            # wants mb=2 (4-LDG ILP needs ~70 regs).
            vec_bits_host = 256 if use_256bit_load else 128
            vec_w_host = vec_bits_host // (32 if torch_dtype == torch.float32
                                           else 16)
            n_vec_iters = max(1,
                              N_per_cta // (num_threads_per_block * vec_w_host))
            if torch_dtype == torch.float32:
                if n_vec_iters < 4:
                    min_blocks_per_mp = 0
                elif num_rows <= num_sms:
                    min_blocks_per_mp = 1
                elif (num_sms * 2 < num_rows <= num_sms * 3
                      and N_per_cta <= 32768):
                    # mb=3 packs all CTAs in 1 wave; at N>=64K kernel
                    # is bandwidth-bound and mb=2 wins instead.
                    min_blocks_per_mp = 3
                else:
                    min_blocks_per_mp = 2
            else:
                if num_rows > num_sms:
                    min_blocks_per_mp = 3
                elif n_vec_iters < 4:
                    min_blocks_per_mp = 0
                else:
                    min_blocks_per_mp = 1

            return dict(
                enable_unroll_4=enable_unroll_4,
                enable_phase3_unroll=enable_phase3_unroll,
                use_constant_hint=use_constant_hint,
                num_threads_per_block=num_threads_per_block,
                use_256bit_load=use_256bit_load,
                enable_warp_parallel_reduce=enable_warp_parallel_reduce,
                min_blocks_per_mp=min_blocks_per_mp,
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
        ) -> tuple:
            key = (dtype, top_k, next_n, enable_unroll_4, enable_phase3_unroll,
                   use_constant_hint, min_blocks_per_mp, use_256bit_load,
                   num_threads_per_block, enable_warp_parallel_reduce,
                   compress_ratio, return_output_values, cluster_size,
                   seqlen_sorted)
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
                    # B200 SXM5 synth-data tuning, 2026-06-10:
                    #   N < 64K              -> 1 (sync unrecouped)
                    #   N >= 128K, BS <= 4   -> 8 (tiny grid)
                    #   BS * cs <= num_sms   -> cs (single-wave)
                    #   else                 -> 1 (multi-wave loses)
                    if N_row < 65536:
                        cluster_size = 1
                    elif num_rows <= 4 and N_row >= 131072:
                        cluster_size = 8
                    elif num_rows * 4 <= num_sms:
                        cluster_size = 4
                    elif num_rows * 2 <= num_sms:
                        cluster_size = 2
                    else:
                        cluster_size = 1
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
            key = cls._compile(
                cute_dtype,
                top_k,
                next_n,
                compress_ratio=compress_ratio,
                return_output_values=return_output_values,
                cluster_size=cluster_size,
                seqlen_sorted=seqlen_sorted,
                **tuning,
            )
            cls.kernel_cache[key](logits, pre_idx, seq_lens, None,
                                  output_indices, order_row)

    # TODO(dsa.py): wire ``order_row = argsort(seq_lens, descending=True)``
    # (device-side, graph-safe) into the LJF row-reorder branch when
    # ``num_rows >= 2 * num_sms``. Physical meaning: wave-2 must fit a
    # full SM-row's worth of CTAs so the sort has long-vs-short rows to
    # swap. Below that threshold the win is noise / can regress a few
    # percent (B200 N∈{8K,16K,32K} sweep 2026-06-23).
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
                     remove_online_sf_transpose=False):
            """Compile kernel using fake tensors + TVM FFI."""
            key = (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n,
                   num_sms, num_epi_subtiles, epi_dtype, output_dtype,
                   remove_online_sf_transpose)
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
                fake_stream,
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
            Returns:
                logits: [B*next_n, max_context_len] output_dtype
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
            logits = torch.empty(
                (B * next_n, aligned_max_ctx),
                device=q.device,
                dtype=output_dtype,
            )
            logits = logits[:, :max_context_len]

            # Compile if needed (fake tensors, no real data required)
            key = (compute_block_kv, phys_block_kv, H, D, next_n, num_sms,
                   num_epi_subtiles, epi_dtype, output_dtype,
                   remove_online_sf_transpose)
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
                    remove_online_sf_transpose=remove_online_sf_transpose)
            compiled = cls.kernel_cache[key]

            # TVM FFI: pass raw tensors, no dlpack/stream needed
            compiled(kv_flat, q_3d, sf_q_2d, w_2d, logits, block_table,
                     context_lens, schedule_meta, num_phys_blocks, B)
            return logits

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
        return CuteDSLFP4PagedMQALogitsRunner.forward(
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
            remove_online_sf_transpose=remove_online_sf_transpose)

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
    ) -> torch.Tensor:
        B = q.shape[0]
        next_n = q.shape[1]
        return torch.empty(B * next_n,
                           max_context_len,
                           dtype=output_dtype,
                           device=q.device)
