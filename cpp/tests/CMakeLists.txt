# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY
    https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# On Windows major version is appended to nvinfer libs.
if(WIN32)
  set(ONNX_PARSER_LIB_NAME nvonnxparser_10)
else()
  set(ONNX_PARSER_LIB_NAME nvonnxparser)
endif()
find_library_create_target(nvonnxparser ${ONNX_PARSER_LIB_NAME} SHARED
                           ${TRT_OUT_DIR} ${TRT_LIB_DIR})

include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include ${3RDPARTY_DIR}/cutlass/include
  ${3RDPARTY_DIR}/cutlass/tools/util/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(google-tests)

function(add_gtest test_name test_src)
  set(options NO_GTEST_MAIN NO_TLLM_LINKAGE)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  add_executable(${test_name} ${test_src})

  target_link_libraries(${test_name} PUBLIC gmock_main nvonnxparser)
  if(NOT ARGS_NO_GTEST_MAIN)
    target_link_libraries(${test_name} PUBLIC gtest_main)
  endif()
  if(NOT ARGS_NO_TLLM_LINKAGE)
    target_link_libraries(${test_name} PUBLIC ${SHARED_TARGET}
                                              nvinfer_plugin_tensorrt_llm)
  endif()

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  gtest_discover_tests(
    ${test_name}
    PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=LAZY" DISCOVERY_MODE
               PRE_TEST # WAR for DLL discovery on windows.
               DISCOVERY_TIMEOUT 30) # Longer timeout needed because discovery
                                     # can be slow on Windows
  add_dependencies(google-tests ${test_name})
endfunction()

add_gtest(loraManagerTest runtime/loraManagerTest.cpp)
add_gtest(loraUtilsTest runtime/loraUtilsTest.cpp)
add_gtest(loraCacheTest runtime/loraCacheTest.cpp)
add_gtest(workerPoolTest runtime/workerPoolTest.cpp)
add_gtest(transposeKVKernelTest runtime/transposeKVKernelTest.cpp)
add_gtest(gptDecoderTest runtime/gptDecoderTest.cpp)
add_gtest(gptDecoderBatchTest runtime/gptDecoderBatchTest.cpp)
add_gtest(gptSessionTest runtime/gptSessionTest.cpp)
add_gtest(allocatorTest common/allocatorTest.cpp)
add_gtest(memoryUtilsTest common/memoryUtilsTest.cu)
if(ENABLE_MULTI_DEVICE EQUAL 1)
  add_gtest(mpiUtilsTest common/mpiUtilsTest.cpp)
endif()
add_gtest(quantizationTest common/quantizationTest.cpp)
add_gtest(stringUtilsTest common/stringUtilsTest.cpp)
add_gtest(tllmExceptionTest common/tllmExceptionTest.cpp)
add_gtest(tensorTest common/tensorTest.cpp)
add_gtest(stlUtilsTest common/stlUtilsTest.cpp)
add_gtest(tllmRuntimeTest runtime/tllmRuntimeTest.cpp)
add_gtest(tllmBuffersTest runtime/tllmBuffersTest.cpp)
add_gtest(bufferManagerTest runtime/bufferManagerTest.cpp)
add_gtest(runtimeKernelTest runtime/runtimeKernelTest.cpp)
add_gtest(samplingTest runtime/samplingTest.cpp)
add_gtest(samplingConfigTest runtime/samplingConfigTest.cpp)
add_gtest(iTensorTest runtime/iTensorTest.cpp)
add_gtest(worldConfigTest runtime/worldConfigTest.cpp)
add_gtest(medusaModuleTest runtime/medusaModuleTest.cpp)
add_gtest(mixtureOfExpertsTest kernels/mixtureOfExpertsTest.cu)
add_gtest(ropeTest kernels/ropeTest.cu)
if(${BUILD_PYT})
  add_gtest(torchTest runtime/torchTest.cpp)
  target_link_libraries(torchTest PUBLIC ${TORCH_LIBRARIES})
endif()
set(SAMPLING_KERNEL_TEST_SRC
    kernels/sampling/samplingTest.cpp
    kernels/sampling/samplingTopKTest.cpp
    kernels/sampling/samplingTopPTest.cpp
    kernels/sampling/samplingAirTopPTest.cpp
    kernels/sampling/samplingPenaltyTest.cpp
    kernels/sampling/samplingUtilsTest.cu)
add_gtest(samplingKernelsTest "${SAMPLING_KERNEL_TEST_SRC}")
add_gtest(weightOnlyKernelTest kernels/weightOnly/weightOnlyKernelTest.cpp)
add_gtest(smoothQuantKernelTest kernels/smoothQuant/smoothQuantKernelTest.cpp)
add_gtest(fp8GemmKernelTest kernels/fp8Gemm/fp8GemmKernelTest.cpp)
if(NOT ENABLE_MULTI_DEVICE EQUAL 0)
  add_gtest(allReduceKernelTest kernels/allReduce/allReduceKernelTest.cu)
endif()
add_gtest(decodingKernelsTest kernels/decodingKernelTest.cpp)
add_gtest(banRepeatNGramsKernelsTest kernels/banRepeatNGramsKernelsTest.cpp)
add_gtest(stopCriteriaKernelsTest kernels/stopCriteriaKernelsTest.cpp)
add_gtest(shiftKCacheKernelTest kernels/shiftKCacheKernelTest.cu)
set(SAMPLING_LAYER_TEST_SRC
    layers/baseSamplingLayerTest.cpp layers/samplingLayerTest.cpp
    layers/topKSamplingLayerTest.cpp layers/topPSamplingLayerTest.cpp)
add_gtest(samplingLayerTest "${SAMPLING_LAYER_TEST_SRC}")
add_gtest(dynamicDecodeLayerTest layers/dynamicDecodeLayerTest.cpp)
add_gtest(medusaDecodeLayerTest layers/medusaDecodeLayerTest.cpp)
set(LOOKAHEAD_POOLMANAGER_TEST_SRC layers/randomLlm.cpp
                                   layers/lookaheadPoolManagerTest.cpp)
add_gtest(lookaheadPoolManagerTest "${LOOKAHEAD_POOLMANAGER_TEST_SRC}")
set(LOOKAHEAD_ALGORITHM_TEST_SRC layers/randomLlm.cpp
                                 layers/lookaheadAlgorithmTest.cpp)
add_gtest(lookaheadAlgorithmTest "${LOOKAHEAD_ALGORITHM_TEST_SRC}")
set(LOOKAHEAD_RANDOMLLM_TEST_SRC layers/randomLlm.cpp
                                 layers/lookaheadRandomLlmTest.cpp)
add_gtest(lookaheadRandomLlmTest "${LOOKAHEAD_RANDOMLLM_TEST_SRC}")
add_gtest(explicitDraftTokensLayerTest layers/explicitDraftTokensLayerTest.cpp)
set(LOOKAHEAD_DECODING_TEST_SRC layers/randomLlm.cpp
                                layers/lookaheadDecodingLayerTest.cpp)
add_gtest(lookaheadDecodingLayerTest "${LOOKAHEAD_DECODING_TEST_SRC}")

add_gtest(
  gemmSwigluRunnerTest
  kernels/fused_gated_gemm/gemmSwigluRunnerTest.cu
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/kernels/fused_gated_gemm/gemm_swiglu_e4m3.cu
  NO_GTEST_MAIN)
add_gtest(gemmSwigluKernelTestSm90Fp8
          kernels/fused_gated_gemm/gemmSwigluKernelTestSm90Fp8.cu NO_GTEST_MAIN
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
  endif()

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()
endforeach()

if(BUILD_BATCH_MANAGER)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batch_manager)
    add_subdirectory(batch_manager)
  endif()
endif()

if(BUILD_EXECUTOR)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/executor)
    add_subdirectory(executor)
  endif()
endif()
