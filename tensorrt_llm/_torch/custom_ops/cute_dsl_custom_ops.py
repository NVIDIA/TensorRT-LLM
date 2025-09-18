from typing import List, Tuple

import torch
import triton  # type: ignore[import]

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import (fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)

if IS_CUTLASS_DSL_AVAILABLE:

    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
        Sm100BlockScaledPersistentDenseGemmKernelWrapper)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    class CuteDSLNVFP4BlackwellLinear(TunableRunner):
        kernel_dict = dict()

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0, 0, get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2), ),
            constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
        )

        def __init__(self, alpha: float, output_dtype: torch.dtype):
            super().__init__()
            self.alpha = alpha
            self.output_dtype = output_dtype
            assert output_dtype == torch.bfloat16

            if get_sm_version() != 100:
                raise ValueError(
                    f"SM version {get_sm_version()} is not supported for CuteDSLNVFP4BlackwellLinear, it only supports SM 100"
                )

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

            gemm_wrapper_func = Sm100BlockScaledPersistentDenseGemmKernelWrapper
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
                gemm = gemm_wrapper_func(
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
