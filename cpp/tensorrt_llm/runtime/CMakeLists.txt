# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION &
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
    utils/sessionUtils.cpp
    bufferManager.cpp
    decodingOutput.cpp
    gptDecoder.cpp
    gptDecoderBatch.cpp
    gptJsonConfig.cpp
    gptSession.cpp
    iBuffer.cpp
    iTensor.cpp
    memoryCounters.cpp
    ncclCommunicator.cpp
    runtimeBuffers.cpp
    runtimeKernels.cu
    statefulGptDecoder.cpp
    tllmRuntime.cpp
    tllmLogger.cpp
    worldConfig.cpp)

include_directories(${API_INCLUDE_DIR}/tensorrt_llm/runtime)

add_compile_options(-Wall)
add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(runtime_src PRIVATE ${MPI_INCLUDE_PATH})

set(JSON_SRC_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/json)
add_subdirectory(${JSON_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/json)

if(ENABLE_MULTI_DEVICE EQUAL 1)
  target_link_libraries(runtime_src PUBLIC nlohmann_json::nlohmann_json
                                           ${NCCL_LIB})
else()
  target_link_libraries(runtime_src PUBLIC nlohmann_json::nlohmann_json)
endif()
