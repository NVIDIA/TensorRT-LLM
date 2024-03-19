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
# the License. cmake needs this line

cmake_minimum_required(VERSION 3.1)

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")
include(${TRTLLM_DIR}/cpp/cmake/modules/set_ifndef.cmake)
include(${TRTLLM_DIR}/cpp/cmake/modules/find_library_create_target.cmake)

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME tensorrt_llm_executor)
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_CXX_FLAGS "-Wall -pthread ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")

set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

# Declare the executable target built from your sources
add_executable(${TARGET_NAME} main.cpp)

set_ifndef(TRT_LIB_DIR
           /usr/local/tensorrt/targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/lib)
set_ifndef(
  TRT_INCLUDE_DIR
  /usr/local/tensorrt/targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/include)

set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})

#
# tensorrt_llm shared lib
add_library(tensorrt_llm SHARED IMPORTED)
set_property(
  TARGET tensorrt_llm
  PROPERTY IMPORTED_LOCATION
           "${TRTLLM_DIR}/cpp/build/tensorrt_llm/libtensorrt_llm.so")

# nvinfer_plugin_tensorrt_llm shared lib
add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
  TARGET nvinfer_plugin_tensorrt_llm
  PROPERTY
    IMPORTED_LOCATION
    "${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)

target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES} nvinfer
                      nvinfer_plugin_tensorrt_llm tensorrt_llm)

# Set include folders
target_include_directories(${TARGET_NAME} PUBLIC /usr/local/cuda/include)
target_include_directories(${TARGET_NAME} PUBLIC ${TRTLLM_DIR}/cpp/include/)
