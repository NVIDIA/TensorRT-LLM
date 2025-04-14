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
set(BATCH_MANAGER_TARGET_NAME tensorrt_llm_batch_manager)
set(BATCH_MANAGER_STATIC_TARGET ${BATCH_MANAGER_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add NIXL configuration
set(NIXL_ROOT "/opt/nvidia/nvda_nixl")

# Find NIXL
find_path(NIXL_INCLUDE_DIR nixl.h HINTS ${NIXL_ROOT}/include
                                        /opt/nvidia/nvda_nixl/include)

# Find all required NIXL libraries
find_library(NIXL_LIBRARY nixl HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
                                     /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
find_library(
  NIXL_BUILD_LIBRARY nixl_build
  HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
        /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
find_library(
  SERDES_LIBRARY serdes HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
                              /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
find_library(
  UCX_BACKEND_LIBRARY ucx_backend
  HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
        /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
find_library(
  UCX_UTILS_LIBRARY ucx_utils HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
                                    /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
find_library(
  GDS_BACKEND_LIBRARY gds_backend
  HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu
        /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)

# Check if all required libraries were found
if(NOT NIXL_LIBRARY
   OR NOT NIXL_BUILD_LIBRARY
   OR NOT SERDES_LIBRARY
   OR NOT UCX_BACKEND_LIBRARY
   OR NOT UCX_UTILS_LIBRARY
   OR NOT GDS_BACKEND_LIBRARY)
  message(FATAL_ERROR "One or more NIXL libraries not found")
endif()

message(STATUS "Found NIXL libraries:")
message(STATUS "  NIXL: ${NIXL_LIBRARY}")
message(STATUS "  NIXL Build: ${NIXL_BUILD_LIBRARY}")
message(STATUS "  SERDES: ${SERDES_LIBRARY}")
message(STATUS "  UCX Backend: ${UCX_BACKEND_LIBRARY}")
message(STATUS "  UCX Utils: ${UCX_UTILS_LIBRARY}")
message(STATUS "  GDS Backend: ${GDS_BACKEND_LIBRARY}")

# keep this list sorted alphabetically
set(SRCS
    allocateKvCache.cpp
    assignReqSeqSlots.cpp
    cacheFormatter.cpp
    mlaCacheFormatter.cpp
    cacheTransceiver.cpp
    capacityScheduler.cpp
    createNewDecoderRequests.cpp
    contextProgress.cpp
    dataTransceiver.cpp
    dataTransceiverImpl.cpp
    decoderBuffers.cpp
    encoderBuffers.cpp
    generateRequestOptions.cpp
    guidedDecoder.cpp
    handleContextLogits.cpp
    handleGenerationLogits.cpp
    kvCacheManager.cpp
    kvCacheEventManager.cpp
    kvCacheTransferManager.cpp
    llmRequest.cpp
    logitsPostProcessor.cpp
    loraBuffers.cpp
    makeDecodingBatchInputOutput.cpp
    medusaBuffers.cpp
    microBatchScheduler.cpp
    pauseRequests.cpp
    peftCacheManager.cpp
    promptTuningBuffers.cpp
    rnnStateBuffers.cpp
    rnnStateManager.cpp
    runtimeBuffers.cpp
    sequenceSlotManager.cpp
    transformerBuffers.cpp
    trtEncoderModel.cpp
    trtGptModelInflightBatching.cpp
    trtGptModelV1.cpp
    updateDecoderBuffers.cpp
    utils/debugUtils.cpp
    utils/inflightBatchingUtils.cpp
    utils/logitsThread.cpp
    utils/staticThreadPool.cpp
    evictionPolicy.cpp)

file(GLOB_RECURSE XGRAMMAR_SRCS "${3RDPARTY_DIR}/xgrammar/cpp/*.cc")
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX
     "${3RDPARTY_DIR}/xgrammar/cpp/pybind/.*\\.cc")
list(APPEND SRCS ${XGRAMMAR_SRCS})

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

add_library(${BATCH_MANAGER_STATIC_TARGET} STATIC ${SRCS})
target_include_directories(
  ${BATCH_MANAGER_STATIC_TARGET}
  PUBLIC ${3RDPARTY_DIR}/xgrammar/3rdparty/picojson
         ${3RDPARTY_DIR}/xgrammar/3rdparty/dlpack/include
         ${3RDPARTY_DIR}/xgrammar/include)

set_target_properties(
  ${BATCH_MANAGER_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${BATCH_MANAGER_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${BATCH_MANAGER_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

# Add include directories
target_include_directories(${BATCH_MANAGER_STATIC_TARGET}
                           PRIVATE ${NIXL_INCLUDE_DIR})

# Link against all NIXL libraries
target_link_libraries(
  ${BATCH_MANAGER_STATIC_TARGET}
  PRIVATE ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY} ${SERDES_LIBRARY}
          ${UCX_BACKEND_LIBRARY} ${UCX_UTILS_LIBRARY} ${GDS_BACKEND_LIBRARY})

# Add rpath for runtime library loading
if(NOT WIN32)
  set_target_properties(
    ${BATCH_MANAGER_STATIC_TARGET}
    PROPERTIES BUILD_RPATH "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
               INSTALL_RPATH "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
               INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(ENABLE_UCX)
  find_package(ucx REQUIRED)
  find_package(ucxx REQUIRED)
  target_include_directories(
    ${BATCH_MANAGER_STATIC_TARGET}
    PRIVATE $<TARGET_PROPERTY:ucxx::ucxx,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(${BATCH_MANAGER_STATIC_TARGET} PUBLIC)
endif()
