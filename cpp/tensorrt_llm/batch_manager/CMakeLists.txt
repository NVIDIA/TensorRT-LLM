# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
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
set(BATCH_MANAGER_TARGET_NAME tensorrt_llm_batch_manager)
set(BATCH_MANAGER_STATIC_TARGET ${BATCH_MANAGER_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    cacheFormatter.cpp
    cacheTransceiver.cpp
    cacheTransferLayer.cpp
    contextTransferCoordinator.cpp
    dataTransceiver.cpp
    kvCacheManagerV2Utils.cpp
    kvCacheManagerV2Utils.cu
    mlaCacheFormatter.cpp
    peftCacheManager.cpp
    rnnCacheFormatter.cpp
    rnnCacheTransBuffer.cpp)

set(xgrammar_source_dir ${CMAKE_BINARY_DIR}/_deps/xgrammar-src)
file(GLOB_RECURSE XGRAMMAR_SRCS "${xgrammar_source_dir}/cpp/*.cc")
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX
     "${xgrammar_source_dir}/cpp/nanobind/.*\\.cc")

add_library(tensorrt_llm_xgrammar_static STATIC ${XGRAMMAR_SRCS})
set_target_properties(
  tensorrt_llm_xgrammar_static
  PROPERTIES CXX_STANDARD "20" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  tensorrt_llm_xgrammar_static SYSTEM
  PRIVATE ${xgrammar_source_dir}/3rdparty/picojson
          ${xgrammar_source_dir}/3rdparty/dlpack/include
          ${xgrammar_source_dir}/include)

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

# Include kv_cache_manager_v2 C++ sources.
include(${CMAKE_CURRENT_SOURCE_DIR}/kv_cache_manager_v2/CMakeLists.txt)

# Include the KVCM cold-page compression codec bridge sources.
include(${CMAKE_CURRENT_SOURCE_DIR}/kv_cache_compression/CMakeLists.txt)

add_library(tllm_kv_cache_host_memory STATIC ${KV_CACHE_HOST_MEMORY_SRCS})
add_library(tllm::kv_cache_host_memory ALIAS tllm_kv_cache_host_memory)
target_include_directories(tllm_kv_cache_host_memory
                           PUBLIC ${KV_CACHE_MANAGER_V2_INCLUDE_DIR})
target_link_libraries(
  tllm_kv_cache_host_memory PUBLIC tllm::common_logging CUDA::cuda_driver
                                   Threads::Threads)

add_library(tllm_kv_cache STATIC ${KV_CACHE_MANAGER_V2_SRCS}
                                 ${KV_CACHE_COMPRESSION_SRCS})
add_library(tllm::kv_cache ALIAS tllm_kv_cache)
target_include_directories(
  tllm_kv_cache PUBLIC ${KV_CACHE_MANAGER_V2_INCLUDE_DIR} ${SHA256_DIR})
target_link_libraries(tllm_kv_cache PUBLIC tllm::kv_cache_host_memory
                                           tllm::common_cuda CUDA::cuda_driver)

add_library(
  tllm_kv_cache_legacy STATIC
  baseTransBuffer.cpp
  blockKey.cpp
  cacheTransBuffer.cpp
  evictionPolicy.cpp
  kvCacheManager.cpp
  kvCacheEventManager.cpp
  kvCacheTransferManager.cpp
  llmRequest.cpp)
add_library(tllm::kv_cache_legacy ALIAS tllm_kv_cache_legacy)
target_link_libraries(
  tllm_kv_cache_legacy
  PUBLIC tllm::runtime_buffers tllm::runtime_distributed tllm::executor_values
  PRIVATE tllm::common_environment tllm::executor_transfer_agent
          tllm::kernel_kv_cache_copy)

add_library(
  tllm_batch_scheduler STATIC
  agentTree.cpp capacityScheduler.cpp contextProgress.cpp
  microBatchScheduler.cpp noOpPeftCacheManager.cpp)
add_library(tllm::batch_scheduler ALIAS tllm_batch_scheduler)
target_link_libraries(tllm_batch_scheduler PUBLIC tllm::kv_cache_legacy)
target_include_directories(tllm_batch_scheduler
                           PRIVATE ${maybe_nvtx_includedir})

foreach(component tllm_kv_cache_host_memory tllm_kv_cache tllm_kv_cache_legacy
                  tllm_batch_scheduler)
  target_compile_features(${component} PUBLIC cxx_std_20)
  set_target_properties(
    ${component} PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endforeach()
target_compile_options(
  tllm_kv_cache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_cuda_architectures(tllm_kv_cache 89)

add_library(${BATCH_MANAGER_STATIC_TARGET} STATIC ${SRCS})
target_link_libraries(${BATCH_MANAGER_STATIC_TARGET}
                      PUBLIC tllm::kv_cache tllm::batch_scheduler)
target_include_directories(
  ${BATCH_MANAGER_STATIC_TARGET}
  PUBLIC ${KV_CACHE_MANAGER_V2_INCLUDE_DIR}
         ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/sha256)

set_target_properties(
  ${BATCH_MANAGER_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "20" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${BATCH_MANAGER_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${BATCH_MANAGER_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

if(ENABLE_UCX)
  find_package(ucx REQUIRED)
  find_package(ucxx REQUIRED)
  target_include_directories(
    ${BATCH_MANAGER_STATIC_TARGET}
    PRIVATE $<TARGET_PROPERTY:ucxx::ucxx,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(${BATCH_MANAGER_STATIC_TARGET} PUBLIC)

endif()

find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)
target_link_libraries(
  ${BATCH_MANAGER_STATIC_TARGET}
  PUBLIC ${TORCH_PYTHON_LIB} Python3::Python pg_utils
  PRIVATE tensorrt_llm_xgrammar_static)
