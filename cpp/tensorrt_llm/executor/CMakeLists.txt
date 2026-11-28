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

set(EXECUTOR_TYPE_SRCS
    cacheTransceiverConfig.cpp
    decodingConfig.cpp
    dynamicBatchConfig.cpp
    guidedDecodingConfig.cpp
    guidedDecodingParams.cpp
    kvCacheConfig.cpp
    kvCacheRetentionConfig.cpp
    outputConfig.cpp
    peftCacheConfig.cpp
    samplingConfig.cpp
    schedulerConfig.cpp
    types.cpp)

add_library(tllm_executor_types STATIC ${EXECUTOR_TYPE_SRCS})
add_library(tllm::executor_types ALIAS tllm_executor_types)
target_link_libraries(
  tllm_executor_types
  PUBLIC tllm::common_error
  PRIVATE tllm::common_logging)
target_compile_features(tllm_executor_types PUBLIC cxx_std_20)
set_target_properties(tllm_executor_types PROPERTIES POSITION_INDEPENDENT_CODE
                                                     ON)

add_library(tllm_executor_json_serialization STATIC jsonSerialization.cpp)
add_library(tllm::executor_json_serialization ALIAS
            tllm_executor_json_serialization)
target_link_libraries(tllm_executor_json_serialization
                      PUBLIC tllm::common_headers)
target_include_directories(
  tllm_executor_json_serialization SYSTEM
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/include)
target_compile_features(tllm_executor_json_serialization PUBLIC cxx_std_20)
set_target_properties(
  tllm_executor_json_serialization PROPERTIES CXX_EXTENSIONS OFF
                                              POSITION_INDEPENDENT_CODE ON)

add_library(tllm_executor_serialization STATIC samplingConfigSerialization.cpp)
add_library(tllm::executor_serialization ALIAS tllm_executor_serialization)
target_link_libraries(
  tllm_executor_serialization
  PUBLIC tllm::executor_types
  PRIVATE tllm::common_error)
set_target_properties(tllm_executor_serialization
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_executor_socket_serialization STATIC
            socketStateSerialization.cpp)
add_library(tllm::executor_socket_serialization ALIAS
            tllm_executor_socket_serialization)
target_link_libraries(tllm_executor_socket_serialization
                      PUBLIC tllm::common_error)

add_library(
  tllm_executor_values STATIC
  contextPhaseParams.cpp
  loraConfig.cpp
  mropeConfig.cpp
  multimodalInput.cpp
  request.cpp
  response.cpp
  serialization.cpp
  tensor.cpp)
add_library(tllm::executor_values ALIAS tllm_executor_values)
target_link_libraries(
  tllm_executor_values
  PUBLIC tllm::executor_serialization tllm::executor_socket_serialization
         tllm::runtime_buffers)

add_library(tllm_executor_cache_routing STATIC
            cache_transmission/cacheRouting.cpp)
add_library(tllm::executor_cache_routing ALIAS tllm_executor_cache_routing)
target_link_libraries(
  tllm_executor_cache_routing
  PUBLIC tllm::common_logging
  PRIVATE tllm::runtime_distributed)

add_library(tllm_executor_transfer_agent STATIC
            cache_transmission/transferAgent.cpp)
add_library(tllm::executor_transfer_agent ALIAS tllm_executor_transfer_agent)
target_link_libraries(
  tllm_executor_transfer_agent
  PUBLIC tllm::common_logging CUDA::cuda_driver
  PRIVATE ${CMAKE_DL_LIBS})

add_library(tllm_executor_memory_desc STATIC memoryDesc.cpp)
add_library(tllm::executor_memory_desc ALIAS tllm_executor_memory_desc)
target_link_libraries(
  tllm_executor_memory_desc
  PUBLIC tllm::common_error
  PRIVATE tllm::executor_values tllm::runtime_distributed)

add_library(tllm_executor_agent_name STATIC
            cache_transmission/agent_utils/agentName.cpp)
add_library(tllm::executor_agent_name ALIAS tllm_executor_agent_name)
target_link_libraries(
  tllm_executor_agent_name
  PUBLIC tllm::common_headers
  PRIVATE Threads::Threads)

add_library(tllm_executor_agent_connection STATIC
            cache_transmission/agent_utils/connection.cpp)
add_library(tllm::executor_agent_connection ALIAS
            tllm_executor_agent_connection)
target_link_libraries(
  tllm_executor_agent_connection
  PUBLIC tllm::executor_transfer_agent tllm::executor_memory_desc
         tllm::batch_request_info tllm::kv_cache_legacy
  PRIVATE tllm::executor_agent_name tllm::executor_cache_routing
          tllm::common_environment pg_utils ${Python3_LIBRARIES})

add_library(
  tllm_executor_cache_transfer_cuda STATIC
  cache_transmission/cacheSplitConcat.cu
  cache_transmission/rnnCacheSplitConcat.cu)
add_library(tllm::executor_cache_transfer_cuda ALIAS
            tllm_executor_cache_transfer_cuda)
target_link_libraries(
  tllm_executor_cache_transfer_cuda
  PUBLIC tllm::runtime_buffers tllm::executor_cache_routing
  PRIVATE tllm::common_environment tllm::executor_values)
set_target_properties(
  tllm_executor_cache_transfer_cuda
  PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(tllm_executor_mpi_connection STATIC
            cache_transmission/mpi_utils/connection.cpp)
add_library(tllm::executor_mpi_connection ALIAS tllm_executor_mpi_connection)
target_link_libraries(tllm_executor_mpi_connection
                      PUBLIC tllm::runtime_distributed tllm::executor_values)

foreach(
  component
  tllm_executor_values
  tllm_executor_transfer_agent
  tllm_executor_agent_name
  tllm_executor_memory_desc
  tllm_executor_cache_routing
  tllm_executor_socket_serialization
  tllm_executor_agent_connection
  tllm_executor_cache_transfer_cuda
  tllm_executor_mpi_connection)
  target_compile_features(${component} PUBLIC cxx_std_20)
  set_target_properties(${component} PROPERTIES CXX_EXTENSIONS OFF
                                                POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(tllm_executor_kv_cache_events STATIC
            executorKVCacheEventManager.cpp)
add_library(tllm::executor_kv_cache_events ALIAS tllm_executor_kv_cache_events)
target_link_libraries(
  tllm_executor_kv_cache_events
  PUBLIC tllm::common_headers
  PRIVATE tllm::kv_cache_legacy)
target_include_directories(tllm_executor_kv_cache_events SYSTEM
                           PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_features(tllm_executor_kv_cache_events PUBLIC cxx_std_20)
set_target_properties(
  tllm_executor_kv_cache_events PROPERTIES CXX_EXTENSIONS OFF
                                           POSITION_INDEPENDENT_CODE ON)

add_subdirectory(cache_transmission/ucx_utils)
add_subdirectory(cache_transmission/mooncake_utils)
add_subdirectory(cache_transmission/nixl_utils)
