# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
set(EXECUTOR_TARGET_NAME tensorrt_llm_executor)
set(EXECUTOR_STATIC_TARGET ${EXECUTOR_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_NIXL)
  set(NIXL_ROOT "/opt/nvidia/nvda_nixl")
  # Include the FindNIXL module
  find_package(NIXL REQUIRED)
  # Check if all required packages were found
  if(NOT NIXL_FOUND)
    message(
      FATAL_ERROR
        "NIXL not found. for building executor. Please install NIXL or set NIXL_ROOT"
    )
  endif()
endif()

# keep this list sorted alphabetically
set(SRCS
    cache_transmission/mpi_utils/connection.cpp
    cache_transmission/cacheConcatenate.cu
    contextPhaseParams.cpp
    debugConfig.cpp
    decodingConfig.cpp
    executor.cpp
    executorConfig.cpp
    executorImpl.cpp
    executorKVCacheEventManager.cpp
    extendedRuntimePerfKnobConfig.cpp
    guidedDecodingConfig.cpp
    guidedDecodingParams.cpp
    jsonSerialization.cpp
    kvCacheConfig.cpp
    kvCacheRetentionConfig.cpp
    logitsPostProcessorConfig.cpp
    loraConfig.cpp
    orchestratorConfig.cpp
    outputConfig.cpp
    parallelConfig.cpp
    peftCacheConfig.cpp
    promptTuningConfig.cpp
    mropeConfig.cpp
    request.cpp
    requestUtils.cpp
    requestWithId.cpp
    response.cpp
    samplingConfig.cpp
    dynamicBatchConfig.cpp
    dynamicBatchTuner.cpp
    schedulerConfig.cpp
    serialization.cpp
    speculativeDecodingConfig.cpp
    tensor.cpp
    types.cpp
    requestUtils.cpp
    contextPhaseParams.cpp
    disaggServerUtil.cpp)

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

add_library(${EXECUTOR_STATIC_TARGET} STATIC ${SRCS})
set_target_properties(
  ${EXECUTOR_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${EXECUTOR_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${EXECUTOR_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

add_subdirectory(cache_transmission/ucx_utils)

if(ENABLE_NIXL)

  add_executable(nixl_example cache_transmission/nixl_utils/nixl_example.cpp)
  target_include_directories(nixl_example PRIVATE ${NIXL_INCLUDE_DIR})
  target_link_libraries(nixl_example PRIVATE ${EXECUTOR_STATIC_TARGET})
  target_link_libraries(
    nixl_example
    PRIVATE ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY} ${SERDES_LIBRARY}
            ${UCX_BACKEND_LIBRARY} ${UCX_UTILS_LIBRARY} ${GDS_BACKEND_LIBRARY})
  # Add include directories
  target_include_directories(${EXECUTOR_STATIC_TARGET}
                             PRIVATE ${NIXL_INCLUDE_DIR})

  # Link against all NIXL libraries
  target_link_libraries(
    ${EXECUTOR_STATIC_TARGET}
    PRIVATE ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY} ${SERDES_LIBRARY}
            ${UCX_BACKEND_LIBRARY} ${UCX_UTILS_LIBRARY} ${GDS_BACKEND_LIBRARY})

  # Add rpath for runtime library loading
  if(NOT WIN32)
    set_target_properties(
      ${EXECUTOR_STATIC_TARGET}
      PROPERTIES BUILD_RPATH "${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}"
                 INSTALL_RPATH "${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}"
                 INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()
