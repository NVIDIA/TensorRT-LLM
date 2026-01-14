#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

set(cutlass_source_dir ${CMAKE_BINARY_DIR}/_deps/cutlass-src)
# Point to the targets include directory which contains both cuda.h and cccl/
# Prefer CUDAToolkit from parent; otherwise derive from CMAKE_CUDA_COMPILER.
if(DEFINED CUDAToolkit_INCLUDE_DIRS AND CUDAToolkit_INCLUDE_DIRS)
  list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_TARGETS_INCLUDE_DIR)
else()
  get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
  get_filename_component(CUDA_TOOLKIT_ROOT ${CUDA_BIN_PATH} DIRECTORY)
  set(CUDA_TARGETS_INCLUDE_DIR
      "${CUDA_TOOLKIT_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
endif()
file(CREATE_LINK ${CUDA_TARGETS_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda
     SYMBOLIC)
file(CREATE_LINK ${cutlass_source_dir} ${CMAKE_CURRENT_BINARY_DIR}/cutlass
     SYMBOLIC)
# Create parent directory for symbolic link
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/trtllm)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_fmha_export/trtllm/dev
     ${CMAKE_CURRENT_BINARY_DIR}/trtllm/dev SYMBOLIC)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/cuda_ptx
     ${CMAKE_CURRENT_BINARY_DIR}/cuda_ptx SYMBOLIC)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_fmha_export/KernelParams.h
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_fmha_export/KernelParamsDecl.h
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

filter_source_cuda_architectures(
  SOURCE_LIST SRC_CPP
  ARCHS 100 103 100f
  TARGET trtllm_gen_fmha_interface)
target_include_directories(
  trtllm_gen_fmha_interface
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_fmha_export)

add_library(trtllm_gen_fmha OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET trtllm_gen_fmha PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_gen_fmha PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(trtllm_gen_fmha PUBLIC trtllm_gen_fmha_interface)
target_include_directories(
  trtllm_gen_fmha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_fmha_export)
set(TRTLLM_FMHA_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(
  trtllm_gen_fmha
  PRIVATE TRTLLM_FMHA_BUILD_DIR="${TRTLLM_FMHA_BUILD_DIR}" TLLM_PUBLIC_RELEASE
          TLLM_GEN_EXPORT_INTERFACE TLLM_FMHA_TRTLLM_COMPAT)
target_compile_definitions(
  trtllm_gen_fmha_interface INTERFACE TLLM_GEN_EXPORT_INTERFACE
                                      TLLM_FMHA_TRTLLM_COMPAT)

# Link the TrtLlmGen FMHA static library
set(TRTLLM_GEN_FMHA_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/${TARGET_ARCH}/libTrtLlmGenFmhaLib.a)
if(NOT EXISTS ${TRTLLM_GEN_FMHA_LIB})
  message(
    FATAL_ERROR
      "TrtLlmGen FMHA library not found: ${TRTLLM_GEN_FMHA_LIB}. "
      "Please ensure the pre-built archive exists under lib/${TARGET_ARCH}/.")
endif()
target_link_libraries(trtllm_gen_fmha PUBLIC ${TRTLLM_GEN_FMHA_LIB})

# Link the TrtLlmGen core library (contains GenLog::getInstance()
# implementation)
set(TRTLLM_GEN_CORE_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/${TARGET_ARCH}/libTrtLlmGen.a)
if(NOT EXISTS ${TRTLLM_GEN_CORE_LIB})
  message(
    FATAL_ERROR
      "TrtLlmGen core library not found: ${TRTLLM_GEN_CORE_LIB}. "
      "Please ensure the pre-built archive exists under lib/${TARGET_ARCH}/.")
endif()
target_link_libraries(trtllm_gen_fmha PUBLIC ${TRTLLM_GEN_CORE_LIB})
