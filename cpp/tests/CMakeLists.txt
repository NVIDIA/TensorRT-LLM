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

option(TRTLLM_BUILD_CPP_TEST_BUNDLE "Package runnable C++ tests for CI" OFF)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
set(test_working_directory)
if(TRTLLM_BUILD_CPP_TEST_BUNDLE)
  # BOLT consumes wheel libraries; test binaries retain symbols without the
  # extra static relocation records needed by the optimizer.
  get_directory_property(test_link_options LINK_OPTIONS)
  list(REMOVE_ITEM test_link_options "-Wl,--emit-relocs")
  set_property(DIRECTORY PROPERTY LINK_OPTIONS "${test_link_options}")
  set(TOP_LEVEL_DIR ".")
  get_filename_component(cpp_test_source_root "${PROJECT_SOURCE_DIR}/.."
                         ABSOLUTE)
  set(test_working_directory WORKING_DIRECTORY "${cpp_test_source_root}")
endif()

add_custom_target(google-tests)
foreach(owner common runtime executor batch-manager kernel)
  add_custom_target(${owner}-tests)
endforeach()

function(tllm_assert_test_is_component_local target)
  get_target_property(link_libraries ${target} LINK_LIBRARIES)
  # Reject expressions mentioning the full target in any configuration branch.
  string(REGEX REPLACE "[$<>:,]" ";" link_tokens "${link_libraries}")
  if("${SHARED_TARGET}" IN_LIST link_tokens)
    message(
      FATAL_ERROR
        "${target} links the full TensorRT-LLM library; use FULL_STACK explicitly"
    )
  endif()
  foreach(dep IN LISTS link_libraries)
    if(NOT TARGET ${dep})
      continue()
    endif()
    get_target_property(dep_interface ${dep} INTERFACE_LINK_LIBRARIES)
    string(REGEX REPLACE "[$<>:,]" ";" dep_interface_tokens "${dep_interface}")
    if("${SHARED_TARGET}" IN_LIST dep_interface_tokens)
      message(
        FATAL_ERROR
          "${target} links ${dep}, which interface-links the full TensorRT-LLM library; use FULL_STACK explicitly"
      )
    endif()
  endforeach()
endfunction()

function(add_tllm_gtest test_name)
  set(options FULL_STACK NO_GTEST_MAIN)
  set(one_value_args TIMEOUT OWNER TEST_PREFIX)
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
  if(NOT ARG_FULL_STACK)
    cmake_language(
      EVAL
      CODE
      "cmake_language(DEFER CALL tllm_assert_test_is_component_local [[${test_name}]])"
    )
  endif()
  target_compile_features(${test_name} PRIVATE cxx_std_20)
  target_compile_definitions(${test_name}
                             PRIVATE TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

  set(test_properties ENVIRONMENT "CUDA_MODULE_LOADING=LAZY")
  if(ARG_TIMEOUT)
    list(APPEND test_properties TIMEOUT "${ARG_TIMEOUT}")
  endif()
  set(discovery_args)
  if(NOT "${ARG_TEST_PREFIX}" STREQUAL "")
    list(APPEND discovery_args TEST_PREFIX "${ARG_TEST_PREFIX}")
  endif()
  gtest_discover_tests(
    ${test_name}
    ${discovery_args}
    ${test_working_directory}
    DISCOVERY_MODE
    PRE_TEST
    DISCOVERY_TIMEOUT
    30
    PROPERTIES LABELS "${ARG_LABELS}" ${test_properties})

  add_dependencies(google-tests ${test_name})
  if(ARG_OWNER)
    add_dependencies(${ARG_OWNER}-tests ${test_name})
  endif()
endfunction()

function(add_gtest test_name test_src)
  set(options NO_GTEST_MAIN NO_TLLM_LINKAGE)
  set(oneValueArgs TIMEOUT TEST_PREFIX)
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

  set(discovery_args)
  if(NOT "${ARGS_TEST_PREFIX}" STREQUAL "")
    list(APPEND discovery_args TEST_PREFIX "${ARGS_TEST_PREFIX}")
  endif()
  gtest_discover_tests(
    ${test_name} ${discovery_args} ${test_working_directory}
    PROPERTIES ${test_properties}
               DISCOVERY_MODE
               PRE_TEST # WAR for DLL discovery on windows.
               DISCOVERY_TIMEOUT
               30) # Longer timeout needed because discovery can be slow on
                   # Windows
  add_dependencies(google-tests ${test_name})
endfunction()

add_subdirectory(unit_tests)

# Directory groups mirror recursive CTest discovery, including legacy and
# full-stack tests that are not part of an OWNER aggregate.
function(add_tllm_test_directory_group directory)
  file(RELATIVE_PATH group "${PROJECT_SOURCE_DIR}/tests/unit_tests"
       "${directory}")
  string(REPLACE "/" "-" group "${group}")
  set(group_target "cpp-tests-${group}")
  add_custom_target(${group_target})
  get_property(
    targets
    DIRECTORY "${directory}"
    PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if(type STREQUAL "EXECUTABLE")
      add_dependencies(${group_target} ${target})
    endif()
    if(TRTLLM_BUILD_CPP_TEST_BUNDLE AND (type STREQUAL "EXECUTABLE"
                                         OR type STREQUAL "SHARED_LIBRARY"))
      get_target_property(binary_dir ${target} BINARY_DIR)
      file(RELATIVE_PATH destination "${CMAKE_BINARY_DIR}" "${binary_dir}")
      file(RELATIVE_PATH library_path "${binary_dir}" "${CMAKE_BINARY_DIR}/lib")
      set_target_properties(
        ${target}
        PROPERTIES
          INSTALL_RPATH
          "$ORIGIN;$ORIGIN/${library_path};$ORIGIN/${library_path}/nixl;$ORIGIN/${library_path}/ucx"
          INSTALL_RPATH_USE_LINK_PATH TRUE)
      target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
      install(
        TARGETS ${target}
        RUNTIME DESTINATION "${destination}"
                COMPONENT cpp_test_bundle
                EXCLUDE_FROM_ALL
        LIBRARY DESTINATION "${destination}"
                COMPONENT cpp_test_bundle
                EXCLUDE_FROM_ALL)
    endif()
  endforeach()
  get_property(
    children
    DIRECTORY "${directory}"
    PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS children)
    add_tllm_test_directory_group("${child}")
    add_dependencies(${group_target} ${TLLM_TEST_DIRECTORY_TARGET})
  endforeach()
  set(TLLM_TEST_DIRECTORY_TARGET
      ${group_target}
      PARENT_SCOPE)
endfunction()

get_property(
  test_directories
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests"
  PROPERTY SUBDIRECTORIES)
foreach(directory IN LISTS test_directories)
  add_tllm_test_directory_group("${directory}")
endforeach()

if(TRTLLM_BUILD_CPP_TEST_BUNDLE)
  configure_file(InstallCppTestBundle.cmake.in InstallCppTestBundle.cmake @ONLY)
  install(
    SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallCppTestBundle.cmake"
    COMPONENT cpp_test_bundle
    EXCLUDE_FROM_ALL)
endif()
