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

  target_link_libraries(${test_name} PUBLIC gmock_main TensorRT::OnnxParser)
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

add_subdirectory(utils)
add_subdirectory(unit_tests)
add_subdirectory(e2e_tests)
