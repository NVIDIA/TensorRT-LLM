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
include(FetchContent)

set(SRCS
    bufferManagerNvls.cpp
    utils/runtimeUtils.cpp
    utils/debugUtils.cu
    loraUtils.cpp
    loraModule.cpp
    loraCache.cpp
    ipcUtils.cpp
    mcastDeviceMemory.cpp
    moeLoadBalancer/hostAccessibleDeviceAllocator.cpp
    moeLoadBalancer/moeLoadBalancer.cpp
    ncclCommunicator.cpp)

include_directories(${API_INCLUDE_DIR}/tensorrt_llm/runtime)

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

set(RUNTIME_MEMORY_SRCS cudaMemPool.cpp memoryCounters.cpp)
set(RUNTIME_BUFFERS_SRCS bufferManager.cpp iBuffer.cpp iTensor.cpp
                         tllmBuffers.cpp virtualMemory.cpp)
set(RUNTIME_DISTRIBUTED_SRCS utils/mpiUtils.cpp worldConfig.cpp)

add_library(tllm_runtime_topology STATIC moeLoadBalancer/topologyDetector.cpp)
add_library(tllm::runtime_topology ALIAS tllm_runtime_topology)
target_link_libraries(tllm_runtime_topology PUBLIC tllm::common_cuda
                                                   Threads::Threads)
if(NOT WIN32)
  target_link_libraries(tllm_runtime_topology PUBLIC NUMA::NUMA)
endif()

add_library(tllm_runtime_memory STATIC ${RUNTIME_MEMORY_SRCS})
add_library(tllm::runtime_memory ALIAS tllm_runtime_memory)
target_link_libraries(
  tllm_runtime_memory
  PUBLIC tllm::common_logging CUDA::cudart
  PRIVATE Threads::Threads)

add_library(tllm_runtime_buffers STATIC ${RUNTIME_BUFFERS_SRCS})
add_library(tllm::runtime_buffers ALIAS tllm_runtime_buffers)
target_link_libraries(
  tllm_runtime_buffers PUBLIC tllm::runtime_memory tllm::common_cuda
                              CUDA::cuda_driver)

add_library(tllm_runtime_distributed STATIC ${RUNTIME_DISTRIBUTED_SRCS})
add_library(tllm::runtime_distributed ALIAS tllm_runtime_distributed)
target_link_libraries(tllm_runtime_distributed
                      PUBLIC tllm::common_logging CUDA::cudart Threads::Threads)
target_compile_definitions(
  tllm_runtime_distributed
  PUBLIC ENABLE_MULTI_DEVICE=$<BOOL:${ENABLE_MULTI_DEVICE}>)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(tllm_runtime_distributed PUBLIC MPI::MPI_C)
endif()

add_library(ipc_nvls_memory_src STATIC ipcNvlsMemory.cu ipcSocket.cpp
                                       mcastGroupComm.cpp)
add_library(tllm::runtime_ipc_nvls ALIAS ipc_nvls_memory_src)
target_link_libraries(
  ipc_nvls_memory_src PUBLIC tllm::common_cuda tllm::runtime_distributed
                             CUDA::cuda_driver)
target_compile_features(ipc_nvls_memory_src PUBLIC cxx_std_20)
set_target_properties(
  ipc_nvls_memory_src
  PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(ipc_nvls_memory_src PRIVATE ${NCCL_LIB})
endif()
if(ENABLE_NVSHMEM)
  target_link_libraries(ipc_nvls_memory_src PRIVATE nvshmem::nvshmem_host
                                                    nvshmem::nvshmem_device)
endif()

add_library(tllm_runtime_worker_pool STATIC workerPool.cpp)
add_library(tllm::runtime_worker_pool ALIAS tllm_runtime_worker_pool)
target_link_libraries(
  tllm_runtime_worker_pool
  PUBLIC tllm::common_headers Threads::Threads
  PRIVATE tllm::common_logging CUDA::cudart)

add_library(tllm_runtime_numpy STATIC utils/numpyUtils.cpp)
add_library(tllm::runtime_numpy ALIAS tllm_runtime_numpy)
target_link_libraries(tllm_runtime_numpy PUBLIC tllm::runtime_buffers)
target_compile_features(tllm_runtime_numpy PUBLIC cxx_std_20)
set_target_properties(tllm_runtime_numpy PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)

add_library(tllm_runtime_gdr STATIC moeLoadBalancer/gdrwrap.cpp)
add_library(tllm::runtime_gdr ALIAS tllm_runtime_gdr)
target_link_libraries(
  tllm_runtime_gdr
  PUBLIC tllm::common_logging CUDA::cudart
  PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_compile_features(tllm_runtime_gdr PUBLIC cxx_std_20)
set_target_properties(tllm_runtime_gdr PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_runtime_locality STATIC
            locality_domain/locality_domain_utils.cpp)
add_library(tllm::runtime_locality ALIAS tllm_runtime_locality)
target_link_libraries(
  tllm_runtime_locality
  PUBLIC tllm::common_cuda CUDA::cuda_driver
  PRIVATE Threads::Threads)
target_compile_features(tllm_runtime_locality PUBLIC cxx_std_20)
set_target_properties(tllm_runtime_locality PROPERTIES POSITION_INDEPENDENT_CODE
                                                       ON)

foreach(component
        tllm_runtime_memory tllm_runtime_buffers tllm_runtime_distributed
        tllm_runtime_topology tllm_runtime_worker_pool)
  target_compile_features(${component} PUBLIC cxx_std_20)
  set_target_properties(${component} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(runtime_src 89)
target_link_libraries(runtime_src PUBLIC ipc_nvls_memory_src
                                         tllm::runtime_topology)

target_include_directories(runtime_src PRIVATE ${MPI_C_INCLUDE_DIRS})

if(ENABLE_MULTI_DEVICE)
  target_link_libraries(runtime_src PUBLIC ${NCCL_LIB})
endif()

if(NOT WIN32)
  target_link_libraries(runtime_src PUBLIC NUMA::NUMA)
endif()

# Add utils subdirectory for pg_utils module
add_subdirectory(utils)
