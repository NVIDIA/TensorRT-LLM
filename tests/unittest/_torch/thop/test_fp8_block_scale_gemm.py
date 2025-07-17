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
import os
import subprocess
import sys

import nvtx
import pytest
import torch
from _torch.helpers import calc_diff, per_block_cast_to_fp8
from utils.util import getSMVersion

from tensorrt_llm._torch.autotuner import autotune

from time import time


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 89,
    reason="The test is for Blackwell and Ada only. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_block_scale_gemm(dtype, m, k, n):
    if getSMVersion() == 89 and k == 7168 and n == 2112:
        pytest.skip("https://nvbugs/5328184")

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output_expected = a @ b.t()

    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                     act_a_sf, act_b_sf)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


# @pytest.mark.skipif(
#     getSMVersion() != 100,
#     reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
# )
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
    # [(7168, 2112)]
    # [(512, 32768)]
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
    # [64]
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_cute_dsl_fp8_block_scale_gemm(dtype, m, k, n):

    # print("limin: torch.ops.trtllm = ", torch.ops.trtllm, dir(torch.ops.trtllm))
    # for op in dir(torch.ops.trtllm):
    #     print(op)
    # print("limin end print ops")

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)

    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    # # torch.float8_e4m3fn
    # # print("limin: act_a_fp8.dtype = ", act_a_fp8.dtype)
    # act_a_fp8 = a.to(torch.float8_e4m3fn)
    # act_b_fp8 = b.to(torch.float8_e4m3fn)
    # act_a_sf = torch.ones((k // 128, m), dtype=torch.float32, device='cuda')
    # act_b_sf = torch.ones((n // 128, k // 128), dtype=torch.float32, device='cuda')

    print("call act_a_fp8.shape", act_a_fp8.shape)
    print("call act_b_fp8.shape", act_b_fp8.shape)
    print("call act_a_sf.shape", act_a_sf.shape)
    print("call act_b_sf.shape", act_b_sf.shape)
    act_a_fp8_origin = act_a_fp8.view(torch.float8_e4m3fn)
    act_b_fp8_origin = act_b_fp8.view(torch.float8_e4m3fn)
    for i in range(10):
        output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8_origin, act_b_fp8_origin,
                                                         act_a_sf, act_b_sf)
    a_time = 0
    for i in range(10):
        t1 = time()
        with nvtx.annotate("fp8_block_scaling_gemm", color="red"):
            output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8_origin, act_b_fp8_origin,
                                                             act_a_sf, act_b_sf)
        t2 = time()
        print(f"    limin: aot time = {(t2 - t1)*1000000} us")
        a_time += (t2 - t1)
    print(f"limin: aot host overhead time = {(a_time / 10)*1000000} us")

    output_expected = a @ b.t()
    diff = calc_diff(output, output_expected)
    print("limin: diff = ", diff)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)

    cute_dsl_fp_gemm_func = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell if getSMVersion(
    ) == 100 else torch.ops.trtllm.cute_dsl_fp8_gemm
    
    """
    with autotune():
        our_out = cute_dsl_fp_gemm_func(act_a_fp8, act_b_fp8, act_a_sf,
                                        act_b_sf)
    print("after autotune")
    from tensorrt_llm._torch.autotuner import AutoTuner
    for k, v in AutoTuner.get().profiling_cache.items():
        print(f"Autotuner profiling cache: {k} = {v}")
    print("inference after autotune")
    host_time = 0
    for i in range(10):
        # with nvtx.annotate("cute_dsl_fp_gemm_func", color="red"):
        t1 = time()
        with nvtx.annotate("cute_dsl_fp_gemm_func", color="red"):
            our_out = cute_dsl_fp_gemm_func(act_a_fp8, act_b_fp8, act_a_sf,
                                            act_b_sf)
        t2 = time()
        host_time += (t2 - t1)
        print(f"limin: time = {(t2 - t1)*1000000} us")
    print(f"limin: host_time = {host_time / 10 * 1000000} us")
    """

    # diff = calc_diff(our_out, output_expected)
    # print("limin: our_out, output_expected, diff = ", diff)
    # assert diff < 1e-3
    # torch.testing.assert_close(our_out, output_expected, atol=1e-3, rtol=1e-3)

    # our_ref = cute_dsl_fp8_linear_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
    # diff = calc_diff(our_ref, output_expected)
    # print("limin: diff = ", diff)
    # assert diff < 1e-3
    # torch.testing.assert_close(our_ref, output_expected, atol=1e-3, rtol=1e-3)

    # our_out = cute_dsl_fp8_linear(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
    # our_ref = cute_dsl_fp8_linear_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
    # diff = calc_diff(our_out, our_ref)
    # print("limin: diff = ", diff)
    # assert diff < 1e-3
    # torch.testing.assert_close(our_out, our_ref, atol=1e-3, rtol=1e-3)

    # ds_ref = run_ds_bs_gemm_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
    # diff = calc_diff(ds_ref, output_expected)
    # print("limin: ds_ref, output_expected, diff = ", diff)
    # assert diff < 1e-3
    # torch.testing.assert_close(ds_ref, output_expected, atol=1e-3, rtol=1e-3)


def test_cute_dsl_fp8_block_scale_bmm(dtype=torch.bfloat16,
                                      m=1024,
                                      k=256,
                                      n=512,
                                      l=16):

    # print("limin: torch.ops.trtllm = ", torch.ops.trtllm, dir(torch.ops.trtllm))
    # for op in dir(torch.ops.trtllm):
    #     print(op)
    # print("limin end print ops")
    import math

    torch.random.manual_seed(0)
    a = torch.randn((l, m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((l, n, k), device='cuda', dtype=dtype) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
        a)

    b_fp8 = torch.empty(l, n, k, dtype=torch.float8_e4m3fn, device="cuda")
    b_scale = torch.empty(l,
                          math.ceil(n / 128),
                          math.ceil(k / 128),
                          dtype=torch.float32,
                          device="cuda")
    for i in range(l):
        cur_b, cur_b_scale = per_block_cast_to_fp8(b[i, :, :])
        b_fp8[i, :, :] = cur_b
        b_scale[i, :, :] = cur_b_scale

    output_expected = torch.einsum("lmk,lnk->lmn", a, b)

    cute_dsl_fp_gemm_func = torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell

    output = torch.empty((l, m, n), device='cuda', dtype=torch.bfloat16)

    # from tensorrt_llm._torch.autotuner import autotune
    with autotune():
        cute_dsl_fp_gemm_func(act_a_fp8, b_fp8, act_a_sf, b_scale, output)
    print("after autotune")
    from tensorrt_llm._torch.autotuner import AutoTuner
    for k, v in AutoTuner.get().profiling_cache.items():
        print(f"Autotuner profiling cache: {k} = {v}")
    print("inference after autotune")
    cute_dsl_fp_gemm_func(act_a_fp8, b_fp8, act_a_sf, b_scale, output)

    diff = calc_diff(output, output_expected)
    print("limin: our_out, output_expected, diff = ", diff)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


# from functools import lru_cache
# from typing import List, Optional, Tuple

# from _torch.helpers import calc_diff, per_block_cast_to_fp8

# # import cutlass.torch as cutlass_torch
# import pytest
# import torch
# from cuda import cuda

# import cutlass
# import cutlass.cute as cute
# from cutlass.cute.runtime import from_dlpack

# import sys
# sys.path.append(
#     '/home/lmin/scratch/trt-dkg/cutlass_ir/compiler/python/examples/blackwell')
# from blockwise_gemm import BlockwiseGemmKernel
# # # from .cute_dsl_blackwell_kernel.blockwise_gemm import BlockwiseGemmKernel

# from utils.util import getSMVersion

# kernel_dict = {}
# def cute_dsl_fp8_gemm_blackwell_wrapper(
#         inputs: List[torch.Tensor],
#         # acc_dtype: torch.dtype = torch.bfloat16,
#         use_2cta_instrs: bool = False,
#         mma_tiler_mn: Tuple[int, int] = (128, 128),
#         cluster_shape_mn: Tuple[int, int] = (1, 1),
#         use_tma_store: bool = True,
#         tactic: int = -1,
#     ) -> torch.Tensor:
#         """Performs linear operation using cute-dsl with autotuning.

#         :param a: Input tensor of shape (M, K)
#         :type a: torch.Tensor, type: fp8
#         :param b: Weight tensor of shape (N, K)
#         :type b: torch.Tensor, type: fp8
#         :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
#             P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
#         :type a_sf: torch.Tensor, type: fp32
#         :param b_sf: Weight scale tensor of shape (w_n, w_k)
#         :type b_sf: torch.Tensor, type: fp32

#         :return: Output tensor of shape (M, N)
#         :rtype: torch.Tensor, type: bf16
#         """
#         a, b, a_sf, b_sf = inputs
#         m, n, k = a.shape[0], b.shape[0], a.shape[1]
#         w_n, w_k = b_sf.shape[0], b_sf.shape[1]
#         c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
#         # print(f"limin: m = {m}, n = {n}, k = {k}, w_n = {w_n}, w_k = {w_k}")
#         # print("limin: a.dtype = ", a.dtype)
#         # print("limin: b.dtype = ", b.dtype)
#         # print("limin: a_sf.dtype = ", a_sf.dtype)
#         # print("limin: b_sf.dtype = ", b_sf.dtype)
#         print(f"limin: a.shape = {a.shape}, a.stride = {a.stride()}")
#         print(f"limin: b.shape = {b.shape}, b.stride = {b.stride()}")
#         print(f"limin: a_sf.shape = {a_sf.shape}, a_sf.stride = {a_sf.stride()}")
#         print(f"limin: b_sf.shape = {b_sf.shape}, b_sf.stride = {b_sf.stride()}")

#         # torch_tensor -> cute.tensor
#         a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
#         b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
#         c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))

#         weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))

#         # m_padded = pad_up(m, 4)
#         # input_scale_tmp = a_sf[0:m_padded * w_k]
#         # # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
#         # input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
#         # # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
#         # input_scale_tmp = input_scale_tmp[:w_k, :m_padded].contiguous().permute(
#         #     1, 0)
#         # # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
#         # input_scale_tmp = input_scale_tmp.as_strided(
#         #     (m_padded, w_k, 1), (1, m_padded, m_padded * w_k))
#         # # print(f"limin: 3, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
#         # [xx, m]
#         input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1), (1, m, w_k * m))
#         print(f"limin: input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
#         print(f"limin: use_2cta_instrs = {use_2cta_instrs}, mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, use_tma_store = {use_tma_store}")

#         mA = from_dlpack(a_tmp,
#                          assumed_align=16).mark_layout_dynamic(leading_dim=1)
#         mB = from_dlpack(b_tmp,
#                          assumed_align=16).mark_layout_dynamic(leading_dim=1)
#         mC = from_dlpack(c_tmp,
#                          assumed_align=16).mark_layout_dynamic(leading_dim=1)
#         mA.element_type = cutlass.Float8E4M3FN
#         mB.element_type = cutlass.Float8E4M3FN

#         # TODO: mSFA is column major
#         mSFA = from_dlpack(input_scale_tmp,
#                            assumed_align=16).mark_layout_dynamic(leading_dim=0)
#         mSFB = from_dlpack(weight_scale_tmp,
#                            assumed_align=16).mark_layout_dynamic(leading_dim=1)

#         # print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
#         # print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
#         # print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
#         # print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
#         # print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")
#         # get stream
#         torch_stream = torch.cuda.current_stream()
#         stream = cuda.CUstream(torch_stream.cuda_stream)

#         cache_key = (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store)
#         if cache_key not in kernel_dict:
#             # gemm = HopperBlockwiseGemmKernel(
#             #     # acc_dtype,  # acc_dtype,
#             #     cutlass.Float32,
#             #     tile_shape_mnk=tile_shape_mnk,
#             #     cluster_shape_mnk=cluster_shape_mnk,
#             # )
#             # gemm = BlockwiseGemmKernel(
#             #     # acc_dtype,  # acc_dtype,
#             #     cutlass.Float32,
#             #     use_2cta_instrs=use_2cta_instrs,
#             #     mma_tiler_mn=mma_tiler_mn,
#             #     cluster_shape_mn=cluster_shape_mn,
#             #     use_tma_store=use_tma_store,
#             # )
#             gemm = BlockwiseGemmKernel(
#                 # acc_dtype,  # acc_dtype,
#                 cutlass.Float32,
#                 use_2cta_instrs=False,
#                 mma_tiler_mn=(128, 128),
#                 cluster_shape_mn=(1, 1),
#                 use_tma_store=True,
#             )
#             # gemm = BlockwiseGemmKernel(
#             #     acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store
#             # )
#             # Compute max active clusters on current device
#             hardware_info = cutlass.utils.HardwareInfo()
#             max_active_clusters = hardware_info.get_max_active_clusters(
#                 cluster_shape_mn[0] * cluster_shape_mn[1]
#             )
#             # max_active_clusters = 148
#             # compiled_gemm = gemm
#             compiled_gemm = cute.compile(
#                 gemm,
#                 mA,
#                 mB,
#                 mC,
#                 mSFA,
#                 mSFB,
#                 max_active_clusters,
#                 stream,
#             )
#             kernel_dict[cache_key] = compiled_gemm
#         else:
#             compiled_gemm = kernel_dict[cache_key]

#         # launch gemm kernel
#         compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)

#         return c

# # @pytest.mark.skipif(
# #     getSMVersion() != 100,
# #     reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
# # )
# @pytest.mark.parametrize(
#     "k, n",
#     # [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
#     #  (2048, 7168), (1024, 1024)],
#     [(7168, 2112)]

# )
# @pytest.mark.parametrize(
#     "m",
#     # [7, 64, 128, 4096],
#     [64]
# )
# @pytest.mark.parametrize(
#     "dtype",
#     [torch.bfloat16],
# )
# def test_cute_dsl_fp8_block_scale_gemm_blackwell(dtype, m, k, n):
#     torch.random.manual_seed(0)
#     a = torch.randn((m, k), device='cuda', dtype=dtype) / k
#     b = torch.randn((n, k), device='cuda', dtype=dtype) / k

#     act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
#     act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)
#     # act_a_fp8 = a.to(torch.float8_e4m3fn)
#     # act_b_fp8 = b.to(torch.float8_e4m3fn)
#     # act_a_sf = torch.ones((k // 128, m), dtype=torch.float32, device='cuda')
#     # act_b_sf = torch.ones((n // 128, k // 128), dtype=torch.float32, device='cuda')

#     print("call act_a_fp8.shape", act_a_fp8.shape)
#     print("call act_b_fp8.shape", act_b_fp8.shape)
#     print("call act_a_sf.shape", act_a_sf.shape)
#     print("call act_b_sf.shape", act_b_sf.shape)
#     # output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
#     #                                                  act_a_sf, act_b_sf)

#     output_expected = a @ b.t()
#     # diff = calc_diff(output, output_expected)
#     # print("limin: diff = ", diff)
#     # assert diff < 1e-3
#     # torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)

#     # with autotune():
#     #     our_out = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(act_a_fp8, act_b_fp8,
#     #                                                  act_a_sf, act_b_sf)
#     # print("after autotune")
#     # from tensorrt_llm._torch.autotuner import AutoTuner
#     # for k, v in AutoTuner.get().profiling_cache.items():
#     #     print(f"Autotuner profiling cache: {k} = {v}")
#     print("inference after autotune")
#     our_out = cute_dsl_fp8_gemm_blackwell_wrapper([act_a_fp8, act_b_fp8, act_a_sf, act_b_sf],
#                                                   use_2cta_instrs=False,
#                                                   mma_tiler_mn=(128, 128),
#                                                   cluster_shape_mn=(1, 1),
#                                                   use_tma_store=True)

#     diff = calc_diff(our_out, output_expected)
#     print("limin: our_out, output_expected, diff = ", diff)
#     assert diff < 1e-3
#     torch.testing.assert_close(our_out, output_expected, atol=1e-3, rtol=1e-3)

#     # our_ref = cute_dsl_fp8_linear_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
#     # diff = calc_diff(our_ref, output_expected)
#     # print("limin: diff = ", diff)
#     # assert diff < 1e-3
#     # torch.testing.assert_close(our_ref, output_expected, atol=1e-3, rtol=1e-3)

#     # our_out = cute_dsl_fp8_linear(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
#     # our_ref = cute_dsl_fp8_linear_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
#     # diff = calc_diff(our_out, our_ref)
#     # print("limin: diff = ", diff)
#     # assert diff < 1e-3
#     # torch.testing.assert_close(our_out, our_ref, atol=1e-3, rtol=1e-3)

#     # ds_ref = run_ds_bs_gemm_ref(act_a_fp8, act_b_fp8, act_a_sf, act_b_sf)
#     # diff = calc_diff(ds_ref, output_expected)
#     # print("limin: ds_ref, output_expected, diff = ", diff)
#     # assert diff < 1e-3
#     # torch.testing.assert_close(ds_ref, output_expected, atol=1e-3, rtol=1e-3)


def run_ds_bs_gemm_ref(a, b, a_scale, b_scale, tile_size=128):
    m, k, n = a.shape[0], a.shape[1], b.shape[0]
    w_k = b_scale.shape[1]
    m_padded = (m + 3) // 4 * 4
    input_scale_tmp = a_scale.reshape(-1, m_padded)
    print(
        f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}"
    )
    input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous()
    print(
        f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}"
    )
    out = fp8_block_scaling_gemm_reference(a, b, input_scale_tmp,
                                           b_scale).cuda().to(torch.bfloat16)


def deepSeekFp8ComputeGemmReference(mM, mN, mK, valsC, dqSfsC, valsA, dqSfsA,
                                    valsB, dqSfsB, quantizeOutput, tileSize):
    for mi in range(mM):
        for ni in range(0, mN, tileSize):
            acc = torch.zeros(tileSize, dtype=torch.float32)
            for nj in range(tileSize):
                nk = ni + nj
                for ki in range(0, mK, tileSize):
                    '''
                    tmp = 0.0
                    for kj in range(tileSize):
                        kk = ki + kj
                        a = valsA[mi, kk]
                        b = valsB[nk, kk]
                        tmp += a * b
                    '''
                    tmp = valsA[mi, ki:ki + tileSize] @ valsB[nk,
                                                              ki:ki + tileSize]
                    dpSfA = dqSfsA[ki // tileSize, mi]
                    dpSfB = dqSfsB[ni // tileSize, ki // tileSize]
                    acc[nj] += (dpSfA * dpSfB) * tmp
            aMax = -float("inf")
            for nj in range(tileSize):
                aMax = max(aMax, abs(acc[nj]))
            E4m3MaxVal = 448
            if dqSfsC is not None:
                dqSfsC[ni // tileSize, mi] = aMax / E4m3MaxVal
            for nj in range(tileSize):
                val = acc[nj]
                if quantizeOutput:
                    val = val / aMax * E4m3MaxVal
                valsC[mi, ni + nj] = val


def fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size=128):
    m, k = a.shape
    n = b.shape[0]
    assert b.shape[1] == k
    assert k % tile_size == 0
    assert n % tile_size == 0
    assert a_scale.shape == (k // tile_size, m)
    assert b_scale.shape == (n // tile_size, k // tile_size)
    c = torch.zeros((m, n), dtype=torch.float32)

    a = a.to(torch.float32).cpu()
    b = b.to(torch.float32).cpu()
    a_scale = a_scale.cpu()
    b_scale = b_scale.cpu()
    deepSeekFp8ComputeGemmReference(m, n, k, c, None, a, a_scale, b, b_scale,
                                    False, tile_size)
    return c


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." %
    getSMVersion(),
)
def test_fp8_blockscale_gemm_reference():
    torch.random.manual_seed(0)

    m, k, n = 3, 6, 4
    tile_size = 2
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32)
    a_scale = torch.ones((k // tile_size, m), dtype=torch.float32)
    b_scale = torch.ones((n // tile_size, k // tile_size), dtype=torch.float32)
    c = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size)
    torch.testing.assert_close(c, a @ b.t(), atol=1e-1, rtol=1e-2)

    m, k, n = 4, 4, 4
    tile_size = 2
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32)
    a_scale = torch.randint(1, 8, (k // tile_size, m), dtype=torch.float32)
    b_scale = torch.randint(1,
                            8, (n // tile_size, k // tile_size),
                            dtype=torch.float32)
    c = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size)
    c_expected = torch.zeros_like(c)
    for i in range(m):
        for j in range(n):
            for kk in range(k):
                a_current_scale = a_scale[kk // tile_size, i]
                b_current_scale = b_scale[j // tile_size, kk // tile_size]
                c_expected[i, j] += a[i, kk] * b[
                    j, kk] * a_current_scale * b_current_scale
    torch.testing.assert_close(c, c_expected, atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_blockscale_gemm_trtllmgen(dtype):
    torch.random.manual_seed(0)

    m, k, n = 128, 512, 512
    tile_size = 128
    if dtype == torch.float8_e4m3fn:
        a = torch.randn((m, k), device='cuda',
                        dtype=torch.float32).to(torch.float8_e4m3fn)
        a_scale = 2 * torch.randn(
            (k // tile_size, m), device='cuda').to(torch.float)

    else:
        a = torch.randn((m, k), device='cuda', dtype=dtype)
        a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
        a_scale = a_scale.view(-1, a.shape[0])

    b = torch.randn((n, k), device='cuda',
                    dtype=torch.float32).to(torch.float8_e4m3fn)
    b_scale = 2 * torch.randn(
        (n // tile_size, k // tile_size), device='cuda').to(torch.float)

    c_expected = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale,
                                                  tile_size)
    c_actual = torch.ops.trtllm.fp8_block_scaling_gemm(a, b, a_scale, b_scale)
    torch.testing.assert_close(c_actual.cpu().to(torch.float32),
                               c_expected,
                               atol=1e-1,
                               rtol=1e-2)


def run_test_in_subprocess(env, test_file):
    # Create a copy of the current environment
    process_env = os.environ.copy()

    # Update with the new environment variables
    process_env.update(env)

    # Run the test in a subprocess
    result = subprocess.run([sys.executable, '-m', 'pytest', test_file, '-v'],
                            capture_output=True,
                            text=True,
                            env=process_env)

    # Print the output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Return the exit code
    return result.returncode


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="The test is for Hopper only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("env", [
    {},
    {
        'TRTLLM_DG_JIT_USE_NVCC': '1'
    },
    {
        'TRTLLM_DG_ENABLED': '0'
    },
])
def test_deep_gemm_in_subprocess(env):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Specify the target test file in the same directory
    test_file = os.path.join(current_dir, "deep_gemm_tests.py")

    exit_code = run_test_in_subprocess(env, test_file)
    assert exit_code == 0, f"Test for env {env} failed with exit code {exit_code}"
