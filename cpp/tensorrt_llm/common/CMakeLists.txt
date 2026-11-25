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

add_library(tllm_common_ip STATIC ipUtils.cpp)
add_library(tllm::common_ip ALIAS tllm_common_ip)
target_link_libraries(
  tllm_common_ip
  PUBLIC tllm::common_headers
  PRIVATE tllm::common_logging)
target_compile_features(tllm_common_ip PUBLIC cxx_std_20)
set_target_properties(tllm_common_ip PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_profiler STATIC cudaProfilerUtils.cpp)
add_library(tllm::common_profiler ALIAS tllm_common_profiler)
target_link_libraries(tllm_common_profiler PUBLIC tllm::common_logging)
target_compile_features(tllm_common_profiler PUBLIC cxx_std_20)
set_target_properties(tllm_common_profiler PROPERTIES POSITION_INDEPENDENT_CODE
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

add_library(tllm_common_native_memory STATIC nativeMemory.cpp)
add_library(tllm::common_native_memory ALIAS tllm_common_native_memory)
target_link_libraries(
  tllm_common_native_memory
  PUBLIC tllm::common_headers
  PRIVATE CUDA::cudart)
target_compile_features(tllm_common_native_memory PUBLIC cxx_std_20)
set_target_properties(tllm_common_native_memory
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_environment STATIC envUtils.cpp)
add_library(tllm::common_environment ALIAS tllm_common_environment)
target_link_libraries(tllm_common_environment PUBLIC tllm::common_cuda)
set_target_properties(tllm_common_environment
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_cublas STATIC cublasMMWrapper.cpp)
add_library(tllm::common_cublas ALIAS tllm_common_cublas)
target_link_libraries(tllm_common_cublas PUBLIC tllm::common_cuda ${CUBLAS_LIB}
                                                ${CUBLASLT_LIB})
target_compile_features(tllm_common_cublas PUBLIC cxx_std_20)
set_target_properties(tllm_common_cublas PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)
if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(tllm_common_cublas PUBLIC ENABLE_CUBLASLT_FP4_GEMM)
endif()

add_library(tllm_common_nccl STATIC ncclUtils.cpp)
add_library(tllm::common_nccl ALIAS tllm_common_nccl)
target_link_libraries(tllm_common_nccl PUBLIC tllm::common_cuda)
target_compile_features(tllm_common_nccl PUBLIC cxx_std_20)
target_compile_definitions(
  tllm_common_nccl PUBLIC ENABLE_MULTI_DEVICE=$<BOOL:${ENABLE_MULTI_DEVICE}>)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(
    tllm_common_nccl
    PUBLIC ${NCCL_LIB} MPI::MPI_C torch_cpu c10
    PRIVATE tllm::runtime_distributed)
endif()
set_target_properties(tllm_common_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_nccl_comm STATIC ncclCommUtils.cpp)
add_library(tllm::common_nccl_comm ALIAS tllm_common_nccl_comm)
target_link_libraries(
  tllm_common_nccl_comm
  PUBLIC tllm::common_nccl
  PRIVATE tllm::runtime_distributed tllm::runtime_ipc_nvls
          tllm::common_native_memory)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(tllm_common_nccl_comm PRIVATE pg_utils)
endif()
target_compile_features(tllm_common_nccl_comm PUBLIC cxx_std_20)
set_target_properties(tllm_common_nccl_comm PROPERTIES POSITION_INDEPENDENT_CODE
                                                       ON)

add_library(tllm_common_mcast_registry STATIC mcastDevMemUtils.cpp)
add_library(tllm::common_mcast_registry ALIAS tllm_common_mcast_registry)
target_link_libraries(tllm_common_mcast_registry PUBLIC tllm::common_headers)
target_compile_features(tllm_common_mcast_registry PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_mcast_registry PROPERTIES CXX_EXTENSIONS OFF
                                        POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_safetensors STATIC safetensors.cpp)
add_library(tllm::common_safetensors ALIAS tllm_common_safetensors)
target_link_libraries(tllm_common_safetensors PUBLIC tllm::common_logging)
target_include_directories(tllm_common_safetensors SYSTEM
                           PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/include)
target_compile_features(tllm_common_safetensors PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_safetensors PROPERTIES CXX_EXTENSIONS OFF
                                     POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_op_utils STATIC opUtils.cpp)
add_library(tllm::common_op_utils ALIAS tllm_common_op_utils)
target_link_libraries(
  tllm_common_op_utils
  PUBLIC tllm::common_cublas CUDA::cuda_driver
  PRIVATE Threads::Threads)
target_compile_definitions(
  tllm_common_op_utils
  PUBLIC ENABLE_MULTI_DEVICE=$<BOOL:${ENABLE_MULTI_DEVICE}>)
if(ENABLE_MULTI_DEVICE)
  target_include_directories(tllm_common_op_utils SYSTEM
                             PUBLIC ${NCCL_INCLUDE_DIR})
endif()
if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(tllm_common_op_utils
                             PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()
target_compile_features(tllm_common_op_utils PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_op_utils PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE
                                                     ON)

add_library(tllm_common_fp8 STATIC cudaFp8Utils.cu)
add_library(tllm::common_fp8 ALIAS tllm_common_fp8)
target_link_libraries(
  tllm_common_fp8
  PUBLIC tllm::common_cuda
  PRIVATE tllm::common_environment)
target_compile_features(tllm_common_fp8 PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_fp8
  PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(tllm_common_cuda_architectures STATIC cudaArchitectures.cpp)
add_library(tllm::common_cuda_architectures ALIAS
            tllm_common_cuda_architectures)
target_link_libraries(
  tllm_common_cuda_architectures
  PUBLIC tllm::common_headers
  PRIVATE tllm::common_cuda)
target_compile_features(tllm_common_cuda_architectures PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_cuda_architectures PROPERTIES CXX_EXTENSIONS OFF
                                            POSITION_INDEPENDENT_CODE ON)
list(JOIN CMAKE_CUDA_ARCHITECTURES_ORIG "," TRTLLM_BUILT_CUDA_ARCHITECTURES)
list(JOIN CMAKE_CUDA_ARCHITECTURES_WITH_KERNELS ","
     TRTLLM_SUPPORTED_CUDA_ARCHITECTURES)
target_compile_definitions(
  tllm_common_cuda_architectures
  PRIVATE
    TRTLLM_BUILT_CUDA_ARCHITECTURES=${TRTLLM_BUILT_CUDA_ARCHITECTURES}
    TRTLLM_SUPPORTED_CUDA_ARCHITECTURES=${TRTLLM_SUPPORTED_CUDA_ARCHITECTURES})

add_library(common_src OBJECT sageQuant.cu)
add_cuda_architectures(common_src 89)
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(
  common_src PUBLIC trtllm_gen_fmha_interface tllm::common_cublas
                    tllm::common_native_memory tllm::common_nccl_comm)
