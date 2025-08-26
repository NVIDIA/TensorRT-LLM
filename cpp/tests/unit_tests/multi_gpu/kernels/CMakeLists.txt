# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

if(NOT ENABLE_MULTI_DEVICE EQUAL 0)
  add_gtest(allReduceKernelTest allReduce/allReduceKernelTest.cu)
  add_gtest(allReduceFusionTest allReduce/allReduceFusionTest.cu)
  add_gtest(gemmAllReduceTest allReduce/gemmAllReduceTest.cu)
  if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
    target_link_libraries(gemmAllReduceTest PRIVATE ar_gemm_src)
    target_compile_definitions(gemmAllReduceTest
                               PRIVATE USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  endif()
endif()
