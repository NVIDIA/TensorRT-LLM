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
set(EXECUTOR_TARGET_NAME tensorrt_llm_executor)
set(EXECUTOR_STATIC_TARGET ${EXECUTOR_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    cache_transmission/agent_utils/connection.cpp
    cache_transmission/cacheSplitConcat.cu
    cache_transmission/mpi_utils/connection.cpp
    cache_transmission/rnnCacheSplitConcat.cu
    cache_transmission/transferAgent.cpp
    contextPhaseParams.cpp
    executorKVCacheEventManager.cpp
    jsonSerialization.cpp
    loraConfig.cpp
    mropeConfig.cpp
    multimodalInput.cpp
    promptTuningConfig.cpp
    request.cpp
    requestUtils.cpp
    requestWithId.cpp
    response.cpp
    serialization.cpp
    tensor.cpp)

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
    extendedRuntimePerfKnobConfig.cpp
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

add_library(tllm_executor_serialization STATIC samplingConfigSerialization.cpp)
add_library(tllm::executor_serialization ALIAS tllm_executor_serialization)
target_link_libraries(
  tllm_executor_serialization
  PUBLIC tllm::executor_types
  PRIVATE tllm::common_error)
set_target_properties(tllm_executor_serialization
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(${EXECUTOR_STATIC_TARGET} STATIC ${SRCS})
set_target_properties(
  ${EXECUTOR_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "20" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${EXECUTOR_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${EXECUTOR_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

add_subdirectory(cache_transmission/ucx_utils)
add_subdirectory(cache_transmission/mooncake_utils)
add_subdirectory(cache_transmission/nixl_utils)
