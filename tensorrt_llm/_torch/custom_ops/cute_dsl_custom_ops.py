import itertools
from typing import List, Optional, Tuple

import torch

from ..._utils import get_sm_version
from ...math_utils import pad_up
from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import (fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

if IS_CUTLASS_DSL_AVAILABLE:

    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
        Sm100BlockScaledPersistentDenseGemmKernelWrapper)
    from ..cute_dsl_kernels.blackwell.grouped_blockscaled_gemm_persistent import \
        Sm100BlockScaledPersistentGroupedGemmKernel
    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    class CuteDSLNVFP4BlackwellLinear(TunableRunner):
        kernel_dict = dict()

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
            use_cuda_graph=True,
        )

        def __init__(self, alpha: float, output_dtype: torch.dtype):
            super().__init__()
            self.alpha = alpha
            self.output_dtype = output_dtype
            assert output_dtype == torch.bfloat16

            if get_sm_version() != 100:
                raise ValueError(
                    f"SM version {get_sm_version()} is not supported for {self.__class__.__name__}, it only supports SM 100"
                )

        # rewrite the hash function because the value of self.alpha doesn't affect the tactic.
        def __hash__(self):
            return hash((self.output_dtype, ))

        def __eq__(self, other):
            if not isinstance(other, CuteDSLNVFP4BlackwellLinear):
                return False
            return self.output_dtype == other.output_dtype

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple[int, int]]:
            assert inputs[0].dim() == 2
            assert inputs[1].dim() == 2

            m = inputs[0].shape[0]
            n = inputs[1].shape[0]
            k = inputs[0].shape[1]
            # Note: the input tensor use uint8 to store fp4, so the real_k is k * 2
            real_k = k * 2
            batch_size = 1
            sf_vec_size = 16
            # m,k
            a_major = "k"
            # n, k
            b_major = "k"

            # full shamoo
            mma_tiler_mn_candidates = [
                (256, 128),
                (128, 128),
                (128, 256),
                (256, 256),
                (256, 64),
                (128, 64),
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

            valid_tactics = []
            for swap_ab in swap_ab_candidates:
                for mma_tiler_mn in mma_tiler_mn_candidates:
                    for cluster_shape_mn in cluster_shape_mn_candidates:
                        if swap_ab:
                            c_major = "m"
                            kernel_m = n
                            kernel_n = m
                        else:
                            c_major = "n"
                            kernel_m = m
                            kernel_n = n

                        if Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
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
                            valid_tactics.append(
                                (mma_tiler_mn, cluster_shape_mn, swap_ab))

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
                    inputs[5]: Output dtype, expected to be torch.bfloat16.
                tactic: Tiling and cluster strategy, typically a tuple (mma_tiler_mn, cluster_shape_mn).

            Returns:
                torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
            """
            sf_vec_size = 16

            if isinstance(tactic, tuple):
                mma_tiler_mn, cluster_shape_mn, swap_ab = tactic
            else:
                # fallback to default tactic
                mma_tiler_mn, cluster_shape_mn, swap_ab = [
                    (128, 128),
                    (1, 1),
                    False,
                ]

            a_tensor, b_tensor, a_sf_tensor, b_sf_tensor = inputs
            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
            c_tensor = torch.empty(*(m, n),
                                   dtype=self.output_dtype,
                                   device="cuda")

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)

            real_k = k * 2
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // sf_vec_size, 4)
            sf_n = pad_up(n, 128)

            # the scaling tensor is 1D. we need to make sure it has been padded to the correct shape
            assert a_sf_tensor.shape == (sf_m * sf_k, )
            assert b_sf_tensor.shape == (sf_n * sf_k, )

            a_ptr = self.make_cute_dsl_global_pointer(a_tensor,
                                                      cutlass.Float4E2M1FN, 32)
            b_ptr = self.make_cute_dsl_global_pointer(b_tensor,
                                                      cutlass.Float4E2M1FN, 32)
            a_sf_ptr = self.make_cute_dsl_global_pointer(
                a_sf_tensor, cutlass.Float8E4M3FN, 16)
            b_sf_ptr = self.make_cute_dsl_global_pointer(
                b_sf_tensor, cutlass.Float8E4M3FN, 16)
            c_ptr = self.make_cute_dsl_global_pointer(c_tensor,
                                                      cutlass.BFloat16, 16)

            # get stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            CACHE_KEY = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
            )
            if swap_ab:
                kernel_a_ptr = b_ptr
                kernel_a_sf_ptr = b_sf_ptr
                kernel_b_ptr = a_ptr
                kernel_b_sf_ptr = a_sf_ptr
                kernel_m = n
                kernel_n = m
                kernel_sf_m = sf_n
                kernel_sf_n = sf_m
            else:
                kernel_a_ptr = a_ptr
                kernel_a_sf_ptr = a_sf_ptr
                kernel_b_ptr = b_ptr
                kernel_b_sf_ptr = b_sf_ptr
                kernel_m = m
                kernel_n = n
                kernel_sf_m = sf_m
                kernel_sf_n = sf_n

            if CACHE_KEY not in CuteDSLNVFP4BlackwellLinear.kernel_dict:
                gemm = Sm100BlockScaledPersistentDenseGemmKernelWrapper(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                )
                # Compute max active clusters on current device
                hardware_info = cutlass.utils.HardwareInfo()
                max_active_clusters = hardware_info.get_max_active_clusters(
                    cluster_shape_mn[0] * cluster_shape_mn[1])

                compiled_gemm = cute.compile(
                    gemm,
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    1,
                    kernel_a_ptr,
                    kernel_b_ptr,
                    kernel_a_sf_ptr,
                    kernel_b_sf_ptr,
                    c_ptr,
                    self.alpha,
                    max_active_clusters,
                    stream,
                    swap_ab,
                )

                CuteDSLNVFP4BlackwellLinear.kernel_dict[
                    CACHE_KEY] = compiled_gemm
            else:
                compiled_gemm = CuteDSLNVFP4BlackwellLinear.kernel_dict[
                    CACHE_KEY]

            # launch gemm kernel
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
                self.alpha,
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
        alpha: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:

        tuner = AutoTuner.get()

        cute_dsl_nvfp4_gemm_blackwell_runner = CuteDSLNVFP4BlackwellLinear(
            alpha, output_dtype)
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_gemm_blackwell",
            [cute_dsl_nvfp4_gemm_blackwell_runner],
            CuteDSLNVFP4BlackwellLinear.tuning_config,
            [input, weight, input_scale, weight_scale],
        )
        return cute_dsl_nvfp4_gemm_blackwell_runner(
            inputs=[input, weight, input_scale, weight_scale],
            tactic=best_tactic,
        )

    @torch.library.register_fake("trtllm::cute_dsl_nvfp4_gemm_blackwell")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: float,
        output_dtype: torch.dtype,
    ):
        # [m, k]
        shape = list(mat_a.shape)
        # [n, k]
        shape[-1] = mat_b.shape[-2]
        # output is fixed as bf16
        ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
        return ret

    class GroupedGemmInputsHelper:

        def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                     local_expert_offset: int, tile_size: int):
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size

        def get_max_num_tiles(self, num_tokens: int) -> int:
            num_expanded_tokens = num_tokens * self.top_k
            if num_expanded_tokens <= self.num_local_experts:
                return num_expanded_tokens
            return (
                num_expanded_tokens +
                (self.tile_size - 1) * self.num_local_experts) // self.tile_size

        def get_max_num_permuted_tokens(self, num_tokens: int) -> int:
            return self.get_max_num_tiles(num_tokens) * self.tile_size

        def infer_num_tokens(self, max_num_permuted_tokens: int) -> int:
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

        def infer_tile_idx_to_group_idx_shape(
                self, input_shapes: List[torch.Size]) -> int:
            return input_shapes[0][0] // self.tile_size

        def inputs_pre_hook(self,
                            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles = inputs
            num_tokens = self.infer_num_tokens(a.size(0))
            average_num_tokens_per_expert = num_tokens * self.top_k / self.num_experts
            balance = 0
            tile_idx_to_group_idx_list = []
            for i in range(self.num_local_experts):
                balance += average_num_tokens_per_expert
                if balance <= 1e-3:
                    continue
                curr_num_tokens = int(balance) + 1
                curr_num_tiles = (curr_num_tokens + self.tile_size -
                                  1) // self.tile_size
                tile_idx_to_group_idx_list.extend([i] * curr_num_tiles)
                balance -= curr_num_tokens

            num_non_exiting_tiles_val = len(tile_idx_to_group_idx_list)
            assert 0 < num_non_exiting_tiles_val <= tile_idx_to_group_idx.size(
                0)

            tile_idx_to_group_idx_list.extend(
                [int(-1e9)] *
                (tile_idx_to_group_idx.size(0) - num_non_exiting_tiles_val))
            tile_idx_to_group_idx = torch.tensor(
                tile_idx_to_group_idx_list,
                dtype=tile_idx_to_group_idx.dtype,
                device=tile_idx_to_group_idx.device)
            num_non_exiting_tiles = torch.tensor(
                [num_non_exiting_tiles_val],
                dtype=num_non_exiting_tiles.dtype,
                device=num_non_exiting_tiles.device)
            return a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, num_non_exiting_tiles

    class Sm100BlockScaledPersistentGroupedGemmRunner(TunableRunner):
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

            if get_sm_version() != 100:
                raise ValueError(
                    f"SM version {get_sm_version()} is not supported for {self.__class__.__name__}, it only supports SM 100"
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

            # TODO: Add full shmoo
            mma_tiler_mn_candidates = [(128, 128), (128, 256)]
            cluster_shape_mn_candidates = [(1, 1), (1, 2)]

            valid_tactics = []
            for mma_tiler_mn, cluster_shape_mn in itertools.product(
                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
                if Sm100BlockScaledPersistentGroupedGemmKernel.can_implement(
                        ab_dtype=cutlass.Float4E2M1FN,
                        sf_dtype=cutlass.Float8E4M3FN,
                        sf_vec_size=self.scaling_vector_size,
                        acc_dtype=cutlass.Float32,
                        c_dtype=cutlass.BFloat16,
                        use_2cta_instrs=False,
                        mma_tiler_mn=mma_tiler_mn,
                        cluster_shape_mn=cluster_shape_mn,
                        m=m,
                        n=n,
                        k=k,
                        l=l,
                        a_major="k",
                        b_major="k",
                        c_major="n",
                        m_aligned=self.tile_size,
                ):
                    valid_tactics.append((mma_tiler_mn, cluster_shape_mn))

            assert len(valid_tactics) > 0
            return valid_tactics

        def get_tuning_config(self) -> TuningConfig:
            key = hash(self)
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
                        ConstraintSpec(
                            5, 0, helper.infer_tile_idx_to_group_idx_shape)),
                    inputs_pre_hook=helper.inputs_pre_hook,
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
            assert num_non_exiting_tiles.size() == (1, )

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
                mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 1)

            cache_key = (self.scaling_vector_size, self.tile_size, mma_tiler_mn,
                         cluster_shape_mn)
            if cache_key not in self.__class__.kernel_cache:
                gemm = Sm100BlockScaledPersistentGroupedGemmKernel(
                    sf_vec_size=self.scaling_vector_size,
                    acc_dtype=cutlass.Float32,
                    use_2cta_instrs=False,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
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
                    num_non_exiting_tiles_ptr,
                    m,
                    n,
                    k,
                    l,
                    tile_size=self.tile_size,
                    scaling_vector_size=self.scaling_vector_size,
                    max_active_clusters=16,
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

        runner = Sm100BlockScaledPersistentGroupedGemmRunner(
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
    ):
        m = input.size(0)
        n = weight.size(1)
        return torch.empty(m, n, dtype=output_dtype, device=input.device)

    class FusedMoEInputsHelper:

        def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                     local_expert_offset: int):
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset

        def infer_token_selected_experts_shape(
                self, input_shapes: List[torch.Size]) -> int:
            return input_shapes[0][0]

        def infer_token_final_scales_shape(
                self, input_shapes: List[torch.Size]) -> int:
            return input_shapes[0][0]

        def inputs_pre_hook(self,
                            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            x, x_sf, token_selected_experts, token_final_scales, *others = inputs
            num_tokens = token_selected_experts.size(0)
            new_token_final_scales, new_token_selected_experts = torch.randn(
                num_tokens,
                self.num_experts,
                device=token_selected_experts.device).topk(self.top_k, dim=-1)
            new_token_selected_experts = new_token_selected_experts.to(
                token_selected_experts.dtype)
            new_token_final_scales = new_token_final_scales.softmax(dim=-1).to(
                token_final_scales.dtype)
            return x, x_sf, new_token_selected_experts, new_token_final_scales, *others

    class Sm100BlockScaledFusedMoERunner(TunableRunner):
        tuning_config_cache = dict()

        def __init__(self,
                     num_experts: int,
                     top_k: int,
                     num_local_experts: int,
                     local_expert_offset: int,
                     output_dtype: torch.dtype,
                     scaling_vector_size: int = 16):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset

            assert output_dtype == torch.bfloat16
            self.output_dtype = output_dtype
            self.scaling_vector_size = scaling_vector_size

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[int]:
            return [128]

        def get_tuning_config(self) -> TuningConfig:
            key = hash(self)
            if key not in self.__class__.tuning_config_cache:
                helper = FusedMoEInputsHelper(self.num_experts, self.top_k,
                                              self.num_local_experts,
                                              self.local_expert_offset)
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets,
                        last_positive_power_of_2), ),
                    constraint_specs=(
                        ConstraintSpec(1, 0, fp4_scale_infer_shape),
                        ConstraintSpec(
                            2, 0, helper.infer_token_selected_experts_shape),
                        ConstraintSpec(3, 0,
                                       helper.infer_token_final_scales_shape)),
                    inputs_pre_hook=helper.inputs_pre_hook,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(self, inputs: List[torch.Tensor],
                    tactic: Optional[int]) -> torch.Tensor:
            if isinstance(tactic, int):
                tile_size = tactic
            else:
                tile_size = 128

            x, x_sf, token_selected_experts, token_final_scales, gemm1_weight, gemm1_weight_scale, gemm1_alpha, gemm2_input_global_scale, gemm2_weight, gemm2_weight_scale, gemm2_alpha = inputs
            tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                num_experts=self.num_experts,
                top_k=self.top_k,
                local_expert_offset=self.local_expert_offset,
                local_num_experts=self.num_local_experts,
                tile_tokens_dim=tile_size,
            )
            x, x_sf = torch.ops.trtllm.moe_permute(
                input=x,
                input_sf=x_sf,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
                top_k=self.top_k,
            )
            x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=gemm1_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=gemm1_weight_scale.view(torch.uint8),
                alpha=gemm1_alpha,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_offset,
                tile_size=tile_size,
                output_dtype=self.output_dtype,
            )
            x, x_sf = torch.ops.trtllm.moe_swiglu_nvfp4_quantize(
                input=x,
                global_sf=gemm2_input_global_scale,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
            )
            x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=gemm2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=gemm2_weight_scale.view(torch.uint8),
                alpha=gemm2_alpha,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_offset,
                tile_size=tile_size,
                output_dtype=self.output_dtype,
            )
            x = torch.ops.trtllm.moe_unpermute(
                permuted_input=x,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
            )
            return x

    @torch.library.custom_op("trtllm::cute_dsl_nvfp4_fused_moe_blackwell",
                             mutates_args=(),
                             device_types="cuda")
    def cute_dsl_nvfp4_fused_moe_blackwell(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        gemm1_weight: torch.Tensor,
        gemm1_weight_scale: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm2_input_global_scale: torch.Tensor,
        gemm2_weight: torch.Tensor,
        gemm2_weight_scale: torch.Tensor,
        gemm2_alpha: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        tuner = AutoTuner.get()
        runner = Sm100BlockScaledFusedMoERunner(num_experts, top_k,
                                                num_local_experts,
                                                local_expert_offset,
                                                output_dtype,
                                                scaling_vector_size)
        inputs = [
            input, input_scale, token_selected_experts, token_final_scales,
            gemm1_weight, gemm1_weight_scale, gemm1_alpha,
            gemm2_input_global_scale, gemm2_weight, gemm2_weight_scale,
            gemm2_alpha
        ]

        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_nvfp4_fused_moe_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        output = runner(inputs, tactic=best_tactic)
        return output

    @torch.library.register_fake("trtllm::cute_dsl_nvfp4_fused_moe_blackwell")
    def _(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        gemm1_weight: torch.Tensor,
        gemm1_weight_scale: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm2_input_global_scale: torch.Tensor,
        gemm2_weight: torch.Tensor,
        gemm2_weight_scale: torch.Tensor,
        gemm2_alpha: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ):
        m, k = input.size(0), input.size(1) * 2
        return torch.empty(m, k, dtype=output_dtype, device=input.device)
