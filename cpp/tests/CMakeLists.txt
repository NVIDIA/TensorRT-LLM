# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
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
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(cutlass_source_dir ${CMAKE_BINARY_DIR}/_deps/cutlass-src)
include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include
  ${cutlass_source_dir}/include
  ${cutlass_source_dir}/tools/util/include
  ${PROJECT_SOURCE_DIR}/tests/batch_manager)

# Tests previously inherited the MPI include dirs transitively through the
# removed TensorRT plugin target's PUBLIC includes.
if(ENABLE_MULTI_DEVICE)
  include_directories(${MPI_C_INCLUDE_DIRS})
endif()

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(google-tests)
foreach(owner common runtime executor batch-manager kernel)
  add_custom_target(${owner}-tests)
endforeach()

function(add_tllm_gtest test_name)
  set(options FULL_STACK NO_GTEST_MAIN REQUIRES_GPU)
  set(one_value_args TIMEOUT OWNER)
  set(multi_value_args SOURCES LINK_LIBRARIES LABELS)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "${options}" "${one_value_args}"
                        "${multi_value_args}")

  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "${test_name}: invalid or incomplete test arguments")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "${test_name}: SOURCES is required")
  endif()
  if(NOT ARG_FULL_STACK AND NOT ARG_LINK_LIBRARIES)
    message(
      FATAL_ERROR
        "${test_name}: specify LINK_LIBRARIES or explicitly use FULL_STACK")
  endif()
  if(ARG_OWNER AND NOT TARGET ${ARG_OWNER}-tests)
    message(FATAL_ERROR "Unknown test owner: ${ARG_OWNER}")
  endif()

  add_executable(${test_name} EXCLUDE_FROM_ALL ${ARG_SOURCES})
  target_link_libraries(${test_name} PRIVATE gmock)
  if(NOT ARG_NO_GTEST_MAIN)
    target_link_libraries(${test_name} PRIVATE gtest_main)
  endif()
  if(ARG_FULL_STACK)
    target_link_libraries(${test_name} PRIVATE ${SHARED_TARGET})
    if(WIN32)
      target_link_libraries(${test_name} PRIVATE context_attention_src)
    endif()
    if(ENABLE_MULTI_DEVICE)
      target_compile_definitions(${test_name} PRIVATE ENABLE_MULTI_DEVICE)
    endif()
  endif()
  target_link_libraries(${test_name} PRIVATE ${ARG_LINK_LIBRARIES})
  target_compile_features(${test_name} PRIVATE cxx_std_20)
  target_compile_definitions(${test_name}
                             PRIVATE TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

  set(test_properties ENVIRONMENT "CUDA_MODULE_LOADING=LAZY" LABELS
                      "${ARG_LABELS}")
  if(ARG_REQUIRES_GPU)
    list(APPEND test_properties RESOURCE_LOCK gpu)
  endif()
  if(ARG_TIMEOUT)
    list(APPEND test_properties TIMEOUT "${ARG_TIMEOUT}")
  endif()
  gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 30
                       PROPERTIES ${test_properties})

  add_dependencies(google-tests ${test_name})
  if(ARG_OWNER)
    add_dependencies(${ARG_OWNER}-tests ${test_name})
  endif()
endfunction()

function(add_gtest test_name test_src)
  set(options NO_GTEST_MAIN NO_TLLM_LINKAGE)
  set(oneValueArgs TIMEOUT)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  add_executable(${test_name} ${test_src})

  target_link_libraries(${test_name} PUBLIC gmock_main)
  if(NOT ARGS_NO_GTEST_MAIN)
    target_link_libraries(${test_name} PUBLIC gtest_main)
  endif()
  if(NOT ARGS_NO_TLLM_LINKAGE)
    target_link_libraries(${test_name} PUBLIC ${SHARED_TARGET})
    if(WIN32)
      target_link_libraries(${test_name} PRIVATE context_attention_src)
    endif()
  endif()
  if(ENABLE_MULTI_DEVICE)
    target_compile_definitions(${test_name} PUBLIC ENABLE_MULTI_DEVICE)
  endif()

  target_compile_features(${test_name} PRIVATE cxx_std_20)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

  set(test_properties ENVIRONMENT "CUDA_MODULE_LOADING=LAZY")
  if(ARGS_TIMEOUT)
    list(APPEND test_properties TIMEOUT "${ARGS_TIMEOUT}")
  endif()

  gtest_discover_tests(
    ${test_name}
    PROPERTIES ${test_properties}
               DISCOVERY_MODE
               PRE_TEST # WAR for DLL discovery on windows.
               DISCOVERY_TIMEOUT
               30) # Longer timeout needed because discovery can be slow on
                   # Windows
  add_dependencies(google-tests ${test_name})
endfunction()

add_subdirectory(unit_tests)
