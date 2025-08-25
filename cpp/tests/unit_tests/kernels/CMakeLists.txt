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

macro(remove_compile_definition TARGET_NAME DEFINITION)
  get_target_property(DEFS ${TARGET_NAME} COMPILE_DEFINITIONS)
  if(DEFS)
    list(REMOVE_ITEM DEFS ${DEFINITION})
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_DEFINITIONS
                                                    "${DEFS}")
  endif()
endmacro()

add_gtest(mixtureOfExpertsTest mixtureOfExpertsTest.cu)

# If we are using oss cutlass, build an explicit internal test
if(USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(mixtureOfExpertsTest
                             PUBLIC USING_OSS_CUTLASS_MOE_GEMM)

  add_gtest(mixtureOfExpertsInternalTest mixtureOfExpertsTest.cu)
  remove_compile_definition(mixtureOfExpertsInternalTest
                            USING_OSS_CUTLASS_MOE_GEMM)
endif()

add_gtest(ropeTest ropeTest.cu)
add_gtest(shiftKCacheKernelTest shiftKCacheKernelTest.cu)
add_gtest(smoothQuantKernelTest smoothQuant/smoothQuantKernelTest.cpp)
add_gtest(stopCriteriaKernelsTest stopCriteriaKernelsTest.cpp)
add_gtest(weightOnlyKernelTest weightOnly/weightOnlyKernelTest.cpp)
add_gtest(mlaPreprocessTest mlaPreprocessTest.cu)
add_gtest(sparseAttentionKernelsTest sparseAttentionKernelsTest.cpp)

add_gtest(cudaCoreGemmKernelTest cudaCoreGemm/cudaCoreGemmKernelTest.cpp)

add_gtest(mlaChunkedPrefillTest mlaChunkedPrefillTest.cu)

add_gtest(fusedMoeCommKernelTest fusedMoeCommKernelTest.cpp)

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

  if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
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

set(ROUTING_KERNEL_TEST_SRC
    routing/routingTest.cpp routing/routingLlama4Test.cpp
    routing/routingRenormalizeTest.cpp routing/routingDeepSeekTest.cpp)
add_gtest(routingKernelsTest "${ROUTING_KERNEL_TEST_SRC}")

add_gtest(moeLoadBalanceKernelTest moeLoadBalanceKernelTest.cpp)

add_gtest(eaglePackDataTest eaglePackDataTest.cpp)
