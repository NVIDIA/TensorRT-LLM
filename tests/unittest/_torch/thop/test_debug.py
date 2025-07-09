# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from typing import List, Tuple

import cutlass
import cutlass.cute as cute
import torch
from _torch.helpers import per_block_cast_to_fp8
from cuda import cuda
from cutlass.cute.runtime import from_dlpack

# import tensorrt_llm
# import ctypes

sys.path.append(
    '/home/lmin/scratch/trt-dkg/cutlass_ir/compiler/python/examples/blackwell')
from blockwise_gemm import BlockwiseGemmKernel

# handle = ctypes.CDLL("/home/lmin/scratch/trtlm-study/tekit-release-0.19/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so",
#                      mode=ctypes.RTLD_GLOBAL,
#                      winmode=None)

kernel_dict = {}


def cute_dsl_fp8_gemm_blackwell_wrapper(
    inputs: List[torch.Tensor],
    # acc_dtype: torch.dtype = torch.bfloat16,
    use_2cta_instrs: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    use_tma_store: bool = True,
    tactic: int = -1,
) -> torch.Tensor:
    """Performs linear operation using cute-dsl with autotuning.

        :param a: Input tensor of shape (M, K)
        :type a: torch.Tensor, type: fp8
        :param b: Weight tensor of shape (N, K)
        :type b: torch.Tensor, type: fp8
        :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
            P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
        :type a_sf: torch.Tensor, type: fp32
        :param b_sf: Weight scale tensor of shape (w_n, w_k)
        :type b_sf: torch.Tensor, type: fp32

        :return: Output tensor of shape (M, N)
        :rtype: torch.Tensor, type: bf16
        """
    a, b, a_sf, b_sf = inputs
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    w_n, w_k = b_sf.shape[0], b_sf.shape[1]
    c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
    # print(f"limin: m = {m}, n = {n}, k = {k}, w_n = {w_n}, w_k = {w_k}")
    # print("limin: a.dtype = ", a.dtype)
    # print("limin: b.dtype = ", b.dtype)
    # print("limin: a_sf.dtype = ", a_sf.dtype)
    # print("limin: b_sf.dtype = ", b_sf.dtype)
    print(f"limin: a.shape = {a.shape}, a.stride = {a.stride()}")
    print(f"limin: b.shape = {b.shape}, b.stride = {b.stride()}")
    print(f"limin: a_sf.shape = {a_sf.shape}, a_sf.stride = {a_sf.stride()}")
    print(f"limin: b_sf.shape = {b_sf.shape}, b_sf.stride = {b_sf.stride()}")

    # torch_tensor -> cute.tensor
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
    b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
    c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))

    weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))

    # m_padded = pad_up(m, 4)
    # input_scale_tmp = a_sf[0:m_padded * w_k]
    # # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    # input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    # # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    # input_scale_tmp = input_scale_tmp[:w_k, :m_padded].contiguous().permute(
    #     1, 0)
    # # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    # input_scale_tmp = input_scale_tmp.as_strided(
    #     (m_padded, w_k, 1), (1, m_padded, m_padded * w_k))
    # # print(f"limin: 3, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    # [xx, m]
    input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                    (1, m, w_k * m))
    print(
        f"limin: input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}"
    )
    print(
        f"limin: use_2cta_instrs = {use_2cta_instrs}, mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, use_tma_store = {use_tma_store}"
    )

    mA = from_dlpack(a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB = from_dlpack(b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mC = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mA.element_type = cutlass.Float8E4M3FN
    mB.element_type = cutlass.Float8E4M3FN

    # TODO: mSFA is column major
    mSFA = from_dlpack(input_scale_tmp,
                       assumed_align=16).mark_layout_dynamic(leading_dim=0)
    mSFB = from_dlpack(weight_scale_tmp,
                       assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
    # print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
    # print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
    # print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
    # print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")
    # get stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    cache_key = (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store)
    if cache_key not in kernel_dict:
        # gemm = HopperBlockwiseGemmKernel(
        #     # acc_dtype,  # acc_dtype,
        #     cutlass.Float32,
        #     tile_shape_mnk=tile_shape_mnk,
        #     cluster_shape_mnk=cluster_shape_mnk,
        # )
        # gemm = BlockwiseGemmKernel(
        #     # acc_dtype,  # acc_dtype,
        #     cutlass.Float32,
        #     use_2cta_instrs=use_2cta_instrs,
        #     mma_tiler_mn=mma_tiler_mn,
        #     cluster_shape_mn=cluster_shape_mn,
        #     use_tma_store=use_tma_store,
        # )
        gemm = BlockwiseGemmKernel(
            # acc_dtype,  # acc_dtype,
            cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_tma_store=True,
        )
        # gemm = BlockwiseGemmKernel(
        #     acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store
        # )
        # Compute max active clusters on current device
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1])
        # max_active_clusters = 148
        # compiled_gemm = gemm
        compiled_gemm = cute.compile(
            gemm,
            mA,
            mB,
            mC,
            mSFA,
            mSFB,
            max_active_clusters,
            stream,
        )
        kernel_dict[cache_key] = compiled_gemm
    else:
        compiled_gemm = kernel_dict[cache_key]

    # launch gemm kernel
    compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)

    return c


def test_cute_dsl_fp8_block_scale_gemm_blackwell(dtype=torch.bfloat16,
                                                 m=64,
                                                 k=7168,
                                                 n=2112):

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    print(f"limin: torch.ops.trtllm = {dir(torch.ops.trtllm)}")
    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)
    # act_a_fp8 = a.to(torch.float8_e4m3fn)
    # act_b_fp8 = b.to(torch.float8_e4m3fn)
    # act_a_sf = torch.ones((k // 128, m), dtype=torch.float32, device='cuda')
    # act_b_sf = torch.ones((n // 128, k // 128), dtype=torch.float32, device='cuda')

    print("call act_a_fp8.shape", act_a_fp8.shape)
    print("call act_b_fp8.shape", act_b_fp8.shape)
    print("call act_a_sf.shape", act_a_sf.shape)
    print("call act_b_sf.shape", act_b_sf.shape)

    output_expected = a @ b.t()
    # diff = calc_diff(output, output_expected)
    # print("limin: diff = ", diff)
    # assert diff < 1e-3
    # torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)

    # with autotune():
    #     our_out = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(act_a_fp8, act_b_fp8,
    #                                                  act_a_sf, act_b_sf)
    # print("after autotune")
    # from tensorrt_llm._torch.autotuner import AutoTuner
    # for k, v in AutoTuner.get().profiling_cache.items():
    #     print(f"Autotuner profiling cache: {k} = {v}")
    print("inference after autotune")
    our_out = cute_dsl_fp8_gemm_blackwell_wrapper(
        [act_a_fp8, act_b_fp8, act_a_sf, act_b_sf],
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_tma_store=True)
    print("our_out.shape", our_out.shape)
    print("output_expected.shape", output_expected.shape)
    print("our_out", our_out)
    print("output_expected", output_expected)


test_cute_dsl_fp8_block_scale_gemm_blackwell(dtype=torch.bfloat16,
                                             m=64,
                                             k=7168,
                                             n=2112)
