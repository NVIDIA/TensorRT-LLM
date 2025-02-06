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

add_gtest(banRepeatNGramsKernelsTest banRepeatNGramsKernelsTest.cpp)
add_gtest(decodingKernelsTest decodingKernelTest.cpp)
add_gtest(logitsBitmaskTest logitsBitmaskTest.cpp)
add_gtest(mixtureOfExpertsTest mixtureOfExpertsTest.cu)
add_gtest(ropeTest ropeTest.cu)
add_gtest(shiftKCacheKernelTest shiftKCacheKernelTest.cu)
add_gtest(smoothQuantKernelTest smoothQuant/smoothQuantKernelTest.cpp)
add_gtest(stopCriteriaKernelsTest stopCriteriaKernelsTest.cpp)
add_gtest(weightOnlyKernelTest weightOnly/weightOnlyKernelTest.cpp)

add_gtest(cudaCoreGemmKernelTest cudaCoreGemm/cudaCoreGemmKernelTest.cpp)

if(NOT ENABLE_MULTI_DEVICE EQUAL 0)
  add_gtest(allReduceKernelTest allReduce/allReduceKernelTest.cu)
  add_gtest(gemmAllReduceTest allReduce/gemmAllReduceTest.cu)
endif()

add_gtest(
  gemmSwigluRunnerTest
  fused_gated_gemm/gemmSwigluRunnerTest.cu
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/kernels/fused_gated_gemm/gemm_swiglu_e4m3.cu
  NO_GTEST_MAIN)
add_gtest(gemmSwigluKernelTestSm90Fp8
          fused_gated_gemm/gemmSwigluKernelTestSm90Fp8.cu NO_GTEST_MAIN
          NO_TLLM_LINKAGE)

foreach(target_name gemmSwigluRunnerTest;gemmSwigluKernelTestSm90Fp8)
  set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # Note - we deliberately do not include 90a PTX (even when 9.0+PTX is
  # specified). This is because sm_90a has arch conditional instructions that
  # are not forward compatible. As a result, it does not make sense to embed PTX
  # into the binary anyway.
  if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     OR "90-real" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     OR "90-real" IN_LIST CMAKE_CUDA_ARCHITECTURES_NATIVE)

    message(STATUS "MANUALLY APPENDING FLAG TO COMPILE FOR SM_90a.")
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a
              -res-usage>)

    # Hopper kernels require cuda lib for TMA APIs
    target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver)

    # No kernels should be parsed, unless hopper is specified. This is a build
    # time improvement
    target_compile_definitions(${target_name} PRIVATE COMPILE_HOPPER_TMA_GEMMS)
    target_compile_definitions(${target_name}
                               PRIVATE COMPILE_HOPPER_TMA_GROUPED_GEMMS)
  endif()

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()
endforeach()

set(SAMPLING_KERNEL_TEST_SRC
    sampling/samplingTest.cpp sampling/samplingTopKTest.cpp
    sampling/samplingTopPTest.cpp sampling/samplingAirTopPTest.cpp
    sampling/samplingPenaltyTest.cpp sampling/samplingUtilsTest.cu)

add_gtest(samplingKernelsTest "${SAMPLING_KERNEL_TEST_SRC}")
