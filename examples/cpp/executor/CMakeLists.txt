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

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include(${TRTLLM_DIR}/cpp/cmake/modules/set_ifndef.cmake)
include(${TRTLLM_DIR}/cpp/cmake/modules/find_library_create_target.cmake)
if(NOT TRTLLM_BUILD_DIR)
  set(TRTLLM_BUILD_DIR "${TRTLLM_DIR}/cpp/build")
endif()
set(TRTLLM_LIB_PATH "${TRTLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so")
set(TRTLLM_PLUGIN_PATH
    "${TRTLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")

# Determine CXX11 ABI compatibility
execute_process(
  COMMAND bash -c "nm -f posix -D ${TRTLLM_LIB_PATH} | grep __cxx11"
  RESULT_VARIABLE GLIB_CXX11_FOUND
  OUTPUT_QUIET)
if(GLIB_CXX11_FOUND EQUAL 0)
  set(USE_CXX11_ABI 1)
else()
  set(USE_CXX11_ABI 0)
endif()
message(STATUS "Use CXX11 ABI: ${USE_CXX11_ABI}")
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE 1)

# Define project name
project(executorExamples)

# Compile options
set(CMAKE_CXX_FLAGS "-Wall -pthread -lstdc++")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE release)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

# TRT dependencies
set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_INCLUDE_DIR /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set(TRT_LIB nvinfer)
# On Windows major version is appended to nvinfer libs.
if(WIN32)
  set(TRT_LIB_NAME nvinfer_10)
else()
  set(TRT_LIB_NAME nvinfer)
endif()
find_library_create_target(${TRT_LIB} ${TRT_LIB_NAME} SHARED ${TRT_LIB_DIR})
message(${TRT_INCLUDE_DIR})
include_directories("${TRT_INCLUDE_DIR}")

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

# tensorrt_llm shared lib
add_library(tensorrt_llm SHARED IMPORTED)
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_PATH})
set_property(
  TARGET tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES
                               CUDA::cuda_driver CUDA::cudart_static CUDA::nvml)

# nvinfer_plugin_tensorrt_llm shared lib
add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION
                                                         ${TRTLLM_PLUGIN_PATH})
set_property(TARGET nvinfer_plugin_tensorrt_llm
             PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES tensorrt_llm)

include_directories(${TRTLLM_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

# Basic
add_executable(executorExampleBasic executorExampleBasic.cpp)
target_link_libraries(executorExampleBasic nvinfer_plugin_tensorrt_llm)

# Advanced
if(NOT TARGET cxxopts::cxxopts)
  set(CXXOPTS_SRC_DIR ${TRTLLM_DIR}/3rdparty/cxxopts)
  add_subdirectory(${CXXOPTS_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)
endif()

add_executable(executorExampleAdvanced executorExampleAdvanced.cpp)
target_link_libraries(executorExampleAdvanced nvinfer_plugin_tensorrt_llm
                      cxxopts::cxxopts)
