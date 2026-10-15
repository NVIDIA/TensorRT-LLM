#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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
if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(tllm_common_headers INTERFACE)
add_library(tllm::common_headers ALIAS tllm_common_headers)
target_include_directories(tllm_common_headers INTERFACE ${API_INCLUDE_DIR}
                                                         ${PROJECT_SOURCE_DIR})

add_library(tllm_common_error STATIC assert.cpp stringUtils.cpp
                                     tllmException.cpp)
add_library(tllm::common_error ALIAS tllm_common_error)
target_link_libraries(
  tllm_common_error
  PUBLIC tllm::common_headers
  PRIVATE ${CMAKE_DL_LIBS})
# stringUtils.h exposes CUDA half types, but needs no CUDA runtime linkage.
target_include_directories(tllm_common_error SYSTEM
                           PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(tllm_common_error PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_logging STATIC logger.cpp timestampUtils.cpp)
add_library(tllm::common_logging ALIAS tllm_common_logging)
target_link_libraries(
  tllm_common_logging
  PUBLIC tllm::common_error
  PRIVATE Threads::Threads CUDA::cudart)
set_target_properties(tllm_common_logging PROPERTIES POSITION_INDEPENDENT_CODE
                                                     ON)

add_library(tllm_common_cuda STATIC cudaDriverWrapper.cpp memoryUtils.cu
                                    nvmlWrapper.cpp)
add_library(tllm::common_cuda ALIAS tllm_common_cuda)
target_link_libraries(
  tllm_common_cuda
  PUBLIC tllm::common_logging CUDA::cudart
  PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(tllm_common_cuda PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(tllm_common_cuda 89)

add_library(
  common_src OBJECT
  attentionOp.cpp
  cublasMMWrapper.cpp
  cudaFp8Utils.cu
  cudaProfilerUtils.cpp
  envUtils.cpp
  ipUtils.cpp
  mcastDevMemUtils.cpp
  ncclUtils.cpp
  opUtils.cpp
  safetensors.cpp
  sageQuant.cu)
add_cuda_architectures(common_src 89)
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(common_src PUBLIC trtllm_gen_fmha_interface)

if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(common_src PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()
