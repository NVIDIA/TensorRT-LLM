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
    mcastGroupComm.cpp
    utils/numpyUtils.cpp
    utils/runtimeUtils.cpp
    utils/debugUtils.cu
    locality_domain/locality_domain_utils.cpp
    loraUtils.cpp
    loraModule.cpp
    loraCache.cpp
    ipcUtils.cpp
    ipcSocket.cpp
    mcastDeviceMemory.cpp
    moeLoadBalancer/gdrwrap.cpp
    moeLoadBalancer/hostAccessibleDeviceAllocator.cpp
    moeLoadBalancer/moeLoadBalancer.cpp
    moeLoadBalancer/topologyDetector.cpp
    ncclCommunicator.cpp
    workerPool.cpp)

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

foreach(component tllm_runtime_memory tllm_runtime_buffers
                  tllm_runtime_distributed)
  target_compile_features(${component} PUBLIC cxx_std_20)
  set_target_properties(${component} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(runtime_src 89)
target_link_libraries(runtime_src PUBLIC ipc_nvls_memory_src)

target_include_directories(runtime_src PRIVATE ${MPI_C_INCLUDE_DIRS})

if(ENABLE_MULTI_DEVICE)
  target_link_libraries(runtime_src PUBLIC ${NCCL_LIB})
endif()

if(NOT WIN32)
  target_link_libraries(runtime_src PUBLIC NUMA::NUMA)
endif()

# Add utils subdirectory for pg_utils module
add_subdirectory(utils)
