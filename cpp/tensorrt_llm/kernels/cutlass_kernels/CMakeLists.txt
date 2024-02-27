#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
#

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

# This can happen when not building for Torch
if(NOT Python3_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
endif()

execute_process(
  WORKING_DIRECTORY ${3RDPARTY_DIR}/cutlass/python/
  COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user
  RESULT_VARIABLE _CUTLASS_LIBRARY_SUCCESS)

if(NOT _CUTLASS_LIBRARY_SUCCESS MATCHES 0)
  message(
    FATAL_ERROR
      "Failed to set up the CUTLASS library due to ${_CUTLASS_LIBRARY_SUCCESS}."
  )
endif()

set_directory_properties(
  PROPERTIES CMAKE_CONFIGURE_DEPENDS
             ${CMAKE_CURRENT_SOURCE_DIR}/python/generate_kernels.py)

execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
  COMMAND ${Python3_EXECUTABLE} generate_kernels.py -o
          ${CMAKE_CURRENT_BINARY_DIR}
  RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

if(NOT _KERNEL_GEN_SUCCESS MATCHES 0)
  message(
    FATAL_ERROR
      "Failed to generate CUTLASS kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
  )
endif()

file(GLOB_RECURSE CU_INSTANTIATIONS ${CMAKE_CURRENT_BINARY_DIR}/*.cu)

add_library(cutlass_src_pre_hopper STATIC ${SRC_CPP} ${SRC_CU})
set_property(TARGET cutlass_src_pre_hopper PROPERTY POSITION_INDEPENDENT_CODE
                                                    ON)
set_property(TARGET cutlass_src_pre_hopper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                    ON)

add_library(cutlass_src_hopper STATIC ${CU_INSTANTIATIONS})
set_property(TARGET cutlass_src_hopper PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET cutlass_src_hopper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_dependencies(cutlass_src_hopper cutlass_src_pre_hopper)

# Note - we deliberately do not include 90a PTX (even when 9.0+PTX is
# specified). This is because sm_90a has arch conditional instructions that are
# not forward compatible. As a result, it does not make sense to embed PTX into
# the binary anyway.
if("9.0" IN_LIST TORCH_CUDA_ARCH_LIST
   OR "9.0+PTX" IN_LIST TORCH_CUDA_ARCH_LIST
   OR TORCH_CUDA_ARCH_LIST STREQUAL "Auto")
  message(STATUS "MANUALLY APPENDING FLAG TO COMPILE FOR SM_90a.")
  target_compile_options(
    cutlass_src_pre_hopper
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)
  target_compile_options(
    cutlass_src_hopper
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)

  # Hopper kernels require cuda lib for TMA APIs
  target_link_libraries(cutlass_src_pre_hopper PRIVATE CUDA::cuda_driver)
  target_link_libraries(cutlass_src_hopper PRIVATE CUDA::cuda_driver)

  # No kernels should be parsed, unless hopper is specified. This is a build
  # time improvement
  target_compile_definitions(cutlass_src_pre_hopper
                             PRIVATE COMPILE_HOPPER_MIXED_INPUT_GEMMS)
  target_compile_definitions(cutlass_src_hopper
                             PRIVATE COMPILE_HOPPER_MIXED_INPUT_GEMMS)
endif()

# Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
# changed in GCC 4.6 This note appears for kernels using TMA and clutters the
# compilation output.
if(NOT WIN32)
  target_compile_options(
    cutlass_src_pre_hopper
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  target_compile_options(
    cutlass_src_hopper
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
endif()
