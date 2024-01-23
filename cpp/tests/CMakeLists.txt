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
# the License. GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

find_library_create_target(nvonnxparser nvonnxparser SHARED ${TRT_OUT_DIR}
                           ${TRT_LIB_DIR})

include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(google-tests)

function(add_gtest test_name test_src)
  add_executable(${test_name} ${test_src})

  if(NOT WIN32) # Linux
    target_link_libraries(
      ${test_name} PUBLIC ${SHARED_TARGET} gtest_main gmock_main nvonnxparser
                          nvinfer_plugin_tensorrt_llm)
  else()
    # Use STATIC_TARGET on Windows because MSVC is picky about duplicate symbols
    # if the shared and static libs both get linked
    target_link_libraries(
      ${test_name} PUBLIC ${STATIC_TARGET} gtest_main gmock_main nvonnxparser
                          nvinfer_plugin_tensorrt_llm)
  endif()

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  gtest_discover_tests(
    ${test_name} PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=LAZY"
                            DISCOVERY_TIMEOUT 30) # Longer timeout needed
                                                  # because discovery can be
                                                  # slow on Windows
  add_dependencies(google-tests ${test_name})
endfunction()

add_gtest(loraManagerTest runtime/loraManagerTest.cpp)
add_gtest(loraUtilsTest runtime/loraUtilsTest.cpp)
add_gtest(attentionKernelTest runtime/transposeKVKernelTest.cpp)
add_gtest(gptDecoderTest runtime/gptDecoderTest.cpp)
add_gtest(gptDecoderBatchTest runtime/gptDecoderBatchTest.cpp)
add_gtest(gptSessionTest runtime/gptSessionTest.cpp)
add_gtest(allocatorTest common/allocatorTest.cpp)
add_gtest(memoryUtilsTest common/memoryUtilsTest.cu)
add_gtest(mpiUtilsTest common/mpiUtilsTest.cpp)
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
add_gtest(iTensorTest runtime/iTensorTest.cpp)
add_gtest(worldConfigTest runtime/worldConfigTest.cpp)
add_gtest(mixtureOfExpertsTest kernels/mixtureOfExpertsTest.cu)
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
add_gtest(decodingKernelsTest kernels/decodingKernelTest.cpp)
add_gtest(stopCriteriaKernelsTest kernels/stopCriteriaKernelsTest.cpp)
add_gtest(shiftKCacheKernelTest kernels/shiftKCacheKernelTest.cu)
set(SAMPLING_LAYER_TEST_SRC
    layers/samplingLayerTest.cpp layers/topKSamplingLayerTest.cpp
    layers/topPSamplingLayerTest.cpp)
add_gtest(samplingLayerTest "${SAMPLING_LAYER_TEST_SRC}")

if(BUILD_BATCH_MANAGER)
  add_subdirectory(batch_manager)
endif()
