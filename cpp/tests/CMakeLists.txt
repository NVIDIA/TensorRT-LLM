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
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
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
  ${PROJECT_SOURCE_DIR}/include
  ${3RDPARTY_DIR}/cutlass/include
  ${3RDPARTY_DIR}/cutlass/tools/util/include
  ${PROJECT_SOURCE_DIR}/tests/batch_manager
  ${PROJECT_SOURCE_DIR}/tests/utils)

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
    if(WIN32)
      target_link_libraries(${test_name} PRIVATE context_attention_src)
    endif()
  endif()
  if(ENABLE_MULTI_DEVICE)
    target_compile_definitions(${test_name} PUBLIC ENABLE_MULTI_DEVICE)
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

add_subdirectory(unit_tests)

add_gtest(mpiUtilsTest runtime/mpiUtilsTest.cpp)

add_gtest(loraManagerTest runtime/loraManagerTest.cpp)
add_gtest(loraCacheTest runtime/loraCacheTest.cpp)
add_gtest(gptDecoderTest runtime/gptDecoderTest.cpp)
add_gtest(gptDecoderBatchedTest runtime/gptDecoderBatchedTest.cpp)
add_gtest(gptSessionTest runtime/gptSessionTest.cpp)
target_link_libraries(gptSessionTest PRIVATE modelSpecStatic)
add_gtest(medusaModuleTest runtime/medusaModuleTest.cpp)

add_gtest(sanitizerTest runtime/sanitizerTest.cpp)

add_gtest(eaglePackDataTest kernels/eaglePackDataTest.cpp)

add_gtest(medusaDecodeLayerTest layers/medusaDecodeLayerTest.cpp)

add_gtest(eagleLayerTest layers/eagleLayerTest.cpp)

add_subdirectory(utils)

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
