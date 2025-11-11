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
include(FetchContent)

set(SRCS
    utils/mpiUtils.cpp
    utils/numpyUtils.cpp
    utils/runtimeUtils.cpp
    utils/debugUtils.cu
    utils/speculativeChoicesUtils.cpp
    bufferManager.cpp
    cudaMemPool.cpp
    decodingLayerWorkspace.cpp
    eagleBuffers.cpp
    explicitDraftTokensBuffers.cpp
    lookaheadBuffers.cpp
    layerProfiler.cpp
    loraManager.cpp
    loraUtils.cpp
    loraModule.cpp
    loraCache.cpp
    decodingOutput.cpp
    decoderState.cpp
    gptDecoder.cpp
    gptDecoderBatched.cpp
    gptJsonConfig.cpp
    iBuffer.cpp
    iTensor.cpp
    ipcUtils.cpp
    ipcSocket.cpp
    ipcNvlsMemory.cu
    mcastDeviceMemory.cpp
    memoryCounters.cpp
    moeLoadBalancer/gdrwrap.cpp
    moeLoadBalancer/hostAccessibleDeviceAllocator.cpp
    moeLoadBalancer/moeLoadBalancer.cpp
    moeLoadBalancer/topologyDetector.cpp
    ncclCommunicator.cpp
    promptTuningParams.cpp
    runtimeKernels.cu
    tllmBuffers.cpp
    tllmRuntime.cpp
    tllmStreamReaders.cpp
    tllmLogger.cpp
    workerPool.cpp
    worldConfig.cpp
    virtualMemory.cpp)

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

add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(runtime_src 89)

target_include_directories(runtime_src PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(runtime_src PUBLIC ${CUDA_NVML_LIB})

if(ENABLE_MULTI_DEVICE)
  target_link_libraries(runtime_src PUBLIC ${NCCL_LIB})
endif()

if(NOT WIN32)
  find_package(libnuma QUIET CONFIG)

  if(NOT libnuma_FOUND)
    message(
      STATUS "libnuma not found via Conan, falling back to system libnuma")
    find_path(NUMA_INCLUDE_DIR numa.h)
    find_library(NUMA_LIBRARY numa)

    if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
      add_library(libnuma::libnuma UNKNOWN IMPORTED)
      set_target_properties(
        libnuma::libnuma
        PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "NUMA library not found, please install libnuma-dev")
    endif()
  else()
    message(STATUS "libnuma found.")
  endif()
  target_link_libraries(runtime_src PUBLIC libnuma::libnuma)
  target_link_options(runtime_src PUBLIC ${CONAN_LIBNUMA_LINK_OPTIONS})
endif()

# Add utils subdirectory for pg_utils module
add_subdirectory(utils)
