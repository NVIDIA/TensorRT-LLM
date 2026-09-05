# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

# Declared centrally in 3rdparty/fetch_content.json so the CI GitHub mirror and
# source revision apply consistently to every FetchContent consumer.
FetchContent_MakeAvailable(nccl_extensions_download)
set(NCCL_EXTENSIONS_SOURCE_DIR ${nccl_extensions_download_SOURCE_DIR})

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    [=[from importlib.metadata import version; from packaging.version import Version; assert Version(version("nccl4py")) >= Version("0.3.1")]=]
  RESULT_VARIABLE NCCL4PY_VERSION_RESULT)
if(NOT NCCL4PY_VERSION_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "nccl4py>=0.3.1 must be installed in Python_EXECUTABLE before building nccl-extensions."
  )
endif()

# NCCL_HOME is the container-supplied NCCL that TRT-LLM discovered. Never clone
# or build the upstream NCCL submodule here.
get_filename_component(NCCL_EXTENSIONS_NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)
get_filename_component(NCCL_EXTENSIONS_NCCL_HOME
                       "${NCCL_EXTENSIONS_NCCL_LIB_DIR}" DIRECTORY)
set(NCCL_HOME
    "${NCCL_EXTENSIONS_NCCL_HOME}"
    CACHE PATH "NCCL container installation used by nccl-extensions" FORCE)
set(NCCL_EP_BUILDDIR
    "${CMAKE_CURRENT_BINARY_DIR}/nccl_ep_build"
    CACHE PATH "NCCL EP artifact output" FORCE)
# NCCL-EP requires SM90+, while TRT-LLM release wheels also include pre-Hopper
# cubins. Restrict only the NCCL-EP targets, preserving the architecture set
# used by every other TRT-LLM target.
set(NCCL_EP_CUDA_ARCHITECTURES)
foreach(NCCL_EP_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX MATCH "^[0-9]+" NCCL_EP_ARCH_NUMBER "${NCCL_EP_ARCH}")
  if(NCCL_EP_ARCH_NUMBER GREATER_EQUAL 90)
    list(APPEND NCCL_EP_CUDA_ARCHITECTURES "${NCCL_EP_ARCH}")
  endif()
endforeach()
if(NOT NCCL_EP_CUDA_ARCHITECTURES)
  message(
    FATAL_ERROR
      "NCCL-EP requires SM90+, but CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}"
  )
endif()
message(
  STATUS "NCCL-EP target CUDA_ARCHITECTURES: ${NCCL_EP_CUDA_ARCHITECTURES}")

add_subdirectory("${NCCL_EXTENSIONS_SOURCE_DIR}/nccl_ep"
                 "${CMAKE_CURRENT_BINARY_DIR}/nccl_ep_native")
set_property(TARGET nccl_ep_shared nccl_ep_static
             PROPERTY CUDA_ARCHITECTURES "${NCCL_EP_CUDA_ARCHITECTURES}")

get_filename_component(NCCL_EXTENSIONS_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}"
                       DIRECTORY)
get_filename_component(NCCL_EXTENSIONS_CUDA_HOME
                       "${NCCL_EXTENSIONS_CUDA_BIN_DIR}" DIRECTORY)

set(NCCL_EXTENSIONS_WHEEL_STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
set(NCCL_EXTENSIONS_WHEEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist")
set(NCCL_EXTENSIONS_WHEEL_STAMP
    "${NCCL_EXTENSIONS_WHEEL_DIR}/.nccl_extensions_wheel_built")
file(GLOB_RECURSE NCCL_EXTENSIONS_PYTHON_SOURCES CONFIGURE_DEPENDS
     "${NCCL_EXTENSIONS_SOURCE_DIR}/python/*")

add_custom_command(
  OUTPUT "${NCCL_EXTENSIONS_WHEEL_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory
          "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory "${NCCL_EXTENSIONS_SOURCE_DIR}/python"
    "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}/nccl/ep/lib"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:nccl_ep_shared>"
          "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}/nccl/ep/lib/libnccl_ep.so"
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    "${NCCL_EP_BUILDDIR}/include/nccl_ep.h"
    "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}/nccl/ep/include/nccl_ep.h"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory "${NCCL_EP_BUILDDIR}/include/nccl_ep"
    "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}/nccl/ep/include/nccl_ep"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${NCCL_EXTENSIONS_WHEEL_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${NCCL_EXTENSIONS_WHEEL_DIR}"
  COMMAND
    ${CMAKE_COMMAND} -E env "CUDA_HOME=${NCCL_EXTENSIONS_CUDA_HOME}"
    "${Python_EXECUTABLE}" -m pip wheel --no-deps --wheel-dir
    "${NCCL_EXTENSIONS_WHEEL_DIR}" "${NCCL_EXTENSIONS_WHEEL_STAGING_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${NCCL_EXTENSIONS_WHEEL_STAMP}"
  DEPENDS nccl_ep_shared ${NCCL_EXTENSIONS_PYTHON_SOURCES}
  VERBATIM)
add_custom_target(nccl_extensions_wheel
                  DEPENDS "${NCCL_EXTENSIONS_WHEEL_STAMP}")
