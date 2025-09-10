import sys
from typing import List, Tuple

try:
    if sys.version_info >= (3, 12):
        HAS_CUTLASS_DSL = True
        import cutlass
        import cutlass.cute as cute

        from tensorrt_llm._torch.cute_dsl_ops.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelWrapper)

        from .cute_dsl_kernels.blackwell.utils import make_ptr
    else:
        HAS_CUTLASS_DSL = False
except ImportError:
    HAS_CUTLASS_DSL = False

import torch
import triton  # type: ignore[import]

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from ..autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                         OptimizationProfile, TunableRunner, TuningConfig)
from ..utils import (fp4_scale_infer_shape,
                     get_last_power_of_2_num_tokens_buckets,
                     last_positive_power_of_2)


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class CuteDSLNVFP4BlackwellLinear(TunableRunner):
    kernel_dict = dict()

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            0, 0, get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2), ),
        constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape), ),
    )

    def __init__(self):
        super().__init__()

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
        # m,k
        a_major = "k"
        # n, k
        b_major = "k"
        # m, n
        c_major = "n"
        sf_vec_size = 16

        # full shamoo
        mma_tiler_mn_candi = [(256, 128), (128, 128), (128, 256), (256, 256),
                              (256, 64), (128, 64)]
        cluster_shape_mn_candi = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2),
                                  (2, 4), (4, 1), (4, 2), (4, 4)]
        return [
            (mma_tiler_mn, cluster_shape_mn)
            for mma_tiler_mn in mma_tiler_mn_candi
            for cluster_shape_mn in cluster_shape_mn_candi
            if Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                cutlass.Float4E2M1FN,  # ab_dtype,
                cutlass.Float8E4M3FN,  # sf_dtype
                sf_vec_size,  # sf_vec_size,
                cutlass.BFloat16,  # c_dtype,
                mma_tiler_mn,
                cluster_shape_mn,
                m,
                n,
                real_k,
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
        """Performs fp8 blockwise (deepgemm like) operation using CuTe DSL.
        :param a (inputs[0]): Input tensor of shape (m, k)
        :type a: torch.Tensor, type: fp4
        :param b (inputs[1]): Weight tensor of shape (n, k)
        :type b: torch.Tensor, type: fp4
        :param a_sf (inputs[2]): Input scale tensor of shape (k//16, m).
        :type a_sf: torch.Tensor, type: fp8
        :param b_sf (inputs[3]): Weight scale tensor of shape (n, k//16)
        :type b_sf: torch.Tensor, type: fp8
        :return: Output tensor of shape (m, n)
        :rtype: torch.Tensor, type: bf16
        """
        sf_vec_size = 16

        if isinstance(tactic, tuple):
            mma_tiler_mn, cluster_shape_mn = tactic
        else:
            # fallback to default tactic
            mma_tiler_mn, cluster_shape_mn = [
                (128, 128),
                (1, 1),
            ]

        a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha, output_dtype = inputs
        assert output_dtype == torch.bfloat16
        m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]
        c_tensor = torch.empty(*(m, n), dtype=output_dtype, device="cuda")

        real_k = k * 2
        sf_m = pad_up(m, 128)
        sf_k = pad_up(real_k // sf_vec_size, 4)
        sf_n = pad_up(n, 128)

        a_ptr = make_ptr(
            # cutlass type
            cutlass.Float4E2M1FN,
            a_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            b_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            a_sf_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            b_sf_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            cutlass.BFloat16,
            c_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )

        # get stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        gemm_wrapper_func = Sm100BlockScaledPersistentDenseGemmKernelWrapper
        cache_key = (
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
        )
        if cache_key not in CuteDSLNVFP4BlackwellLinear.kernel_dict:
            gemm = gemm_wrapper_func(
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
            )
            # Compute max active clusters on current device
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1])
            hardware_info.get_l2_cache_size_in_bytes()

            compiled_gemm = cute.compile(
                gemm,
                m,
                n,
                real_k,
                sf_m // 128,
                sf_n // 128,
                sf_k // 4,
                1,
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                alpha,
                max_active_clusters,
                stream,
            )

            CuteDSLNVFP4BlackwellLinear.kernel_dict[cache_key] = compiled_gemm
        else:
            compiled_gemm = CuteDSLNVFP4BlackwellLinear.kernel_dict[cache_key]

        # launch gemm kernel
        compiled_gemm(m, n, real_k, sf_m // 128, sf_n // 128, sf_k // 4, a_ptr,
                      b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha, stream)
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

    if not HAS_CUTLASS_DSL:
        raise RuntimeError("nvidia-cutlass-dsl 4.1.0 requires Python >=3.12")

    tuner = AutoTuner.get()

    cute_dsl_nvfp4_gemm_blackwell_runner = CuteDSLNVFP4BlackwellLinear()
    _, best_tactic = tuner.choose_one(
        "trtllm::cute_dsl_nvfp4_gemm_blackwell::gemm",
        [cute_dsl_nvfp4_gemm_blackwell_runner],
        CuteDSLNVFP4BlackwellLinear.tuning_config,
        [input, weight, input_scale, weight_scale, alpha, output_dtype],
    )
    return cute_dsl_nvfp4_gemm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale, alpha, output_dtype],
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
    shape = [i for i in mat_a.shape]
    # [n, k]
    shape[-1] = mat_b.shape[-2]
    # output is fixed as bf16
    ret = mat_a.new_empty(shape, dtype=torch.bfloat16)
    return ret
