import itertools
import math
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version
from ...math_utils import ceil_div, pad_up
from ..autotuner import (AutoTuner, ConstraintSpec, DistributedTuningStrategy,
                         DynamicTensorSpec, OptimizationProfile, TunableRunner,
                         TuningConfig)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import (fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)

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
        """Pre-hook for gather-based SwiGLU fusion kernel.

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

    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import \
        BlockScaledContiguousGatherGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm import \
        Sm100BlockScaledContiguousGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import \
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
    from ..cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_swiglu_fusion import \
        Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import \
        Sm100BlockScaledPersistentDenseGemmKernel
    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    class CuteDSLNVFP4BlackwellLinear(TunableRunner):
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
                     to_userbuffers: bool = False,
                     use_tvm_ffi: bool = True):
            super().__init__()

            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"CuteDSL NVFP4 only supports bfloat16 output, got {output_dtype}"
                )
            self.output_dtype = output_dtype
            self.to_userbuffers = to_userbuffers
            self.use_tvm_ffi = use_tvm_ffi

        def unique_id(self):
            return (self.output_dtype, self.to_userbuffers, self.use_tvm_ffi)

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
            Performs fp8 blockwise gemm operation using CuTe DSL.

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

            # Allocate output tensor from UserBuffers or regular CUDA memory
            if self.to_userbuffers:
                c_tensor = torch.ops.trtllm.create_userbuffers_tensor(
                    [m, n], self.output_dtype)
            else:
                c_tensor = torch.empty(*(m, n),
                                       dtype=self.output_dtype,
                                       device="cuda")

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
                         use_prefetch)
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
        to_userbuffers: bool = False,
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
            to_userbuffers: Whether to allocate output from UserBuffers pool
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

        runner = CuteDSLNVFP4BlackwellLinear(output_dtype, to_userbuffers,
                                             use_tvm_ffi)
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
        to_userbuffers: bool = False,
        use_tvm_ffi: bool = True,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k]
        shape[-1] = mat_b.shape[-2]
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

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
            m, k = a.size(0), a.size(1) * 2
            l, n = b.size(0), b.size(1)

            mma_tiler_mn_candidates = [(self.tile_size, 128),
                                       (self.tile_size, 256)]
            cluster_shape_mn_candidates = [(self.tile_size // 128, 1),
                                           (self.tile_size // 128, 2)]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
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
            l, n = b.size(0), b.size(1)

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
                    use_blkred=True,
                    raster_along_m=raster_along_m,
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
            )
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
            if tile_size not in [128, 256]:
                raise ValueError(
                    f"Tile size {tile_size} is not supported, it only supports 128 and 256."
                )
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
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, *_ = inputs
            # m is the permuted size from permuted_idx_to_expanded_idx, not from a
            m = permuted_idx_to_expanded_idx.size(0)
            k = a.size(1) * 2
            l, n = b.size(0), b.size(1)

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

        def forward(self, inputs: List[torch.Tensor],
                    tactic: Optional[tuple]) -> torch.Tensor:
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf = inputs
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
            l, n = b.size(0), b.size(1)
            scale_k = k // self.scaling_vector_size
            interm_size = n // 2
            assert m % self.tile_size == 0
            assert k % (self.scaling_vector_size * 4) == 0
            assert n % (self.scaling_vector_size * 4 * 2) == 0
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
                mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
            else:
                mma_tiler_mn = (self.tile_size, 128)
                cluster_shape_mn = (self.tile_size // 128, 1)
                raster_along_m = False
            assert mma_tiler_mn[
                0] == self.tile_size, f"Tactic ({tactic}) is incompatible with tile size ({self.tile_size})"

            cache_key = (self.scaling_vector_size, self.tile_size, self.top_k,
                         mma_tiler_mn, cluster_shape_mn, raster_along_m)
            if cache_key not in self.__class__.kernel_cache:
                gemm = self.__class__.kernel_class(
                    sf_vec_size=self.scaling_vector_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    vectorized_f32=True,
                    topk=self.top_k,
                    raster_along_m=raster_along_m,
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
            )
            return c, c_sf

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
        tuner = AutoTuner.get()

        runner = Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner(
            num_experts, top_k, num_local_experts, local_expert_offset,
            tile_size, scaling_vector_size)
        inputs = [
            input, weight, input_scale, weight_scale, alpha,
            tile_idx_to_group_idx, tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake(
        "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell")
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
