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

cmake_minimum_required(VERSION 3.27)

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
list(APPEND CMAKE_MODULE_PATH "${TRTLLM_DIR}/cpp/cmake/modules")

if(NOT TRTLLM_BUILD_DIR)
  set(TRTLLM_BUILD_DIR "${TRTLLM_DIR}/cpp/build")
endif()
set(TRTLLM_LIB_PATH "${TRTLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so")
if(NOT EXISTS ${TRTLLM_LIB_PATH})
  message(FATAL_ERROR "Cannot find ${TRTLLM_LIB_PATH}")
endif()

set(TRTLLM_PLUGIN_PATH
    "${TRTLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")

option(
  ENABLE_MULTI_DEVICE
  "Enable multi device/instance examples building (requires MPI headers/libs)"
  ON)

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

# Compile options $<BOOL:${ENABLE_MULTI_DEVICE}> ?
if(ENABLE_MULTI_DEVICE)
  set(EMD 1)
else()
  set(EMD 0)
endif()
set(CMAKE_CXX_FLAGS "-Wall -pthread -lstdc++ -DENABLE_MULTI_DEVICE=${EMD} ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE release)

find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver cudart_static nvml)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

# TRT dependencies
find_package(TensorRT 10 REQUIRED)

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

add_executable(executorExampleDebug executorExampleDebug.cpp)
target_link_libraries(executorExampleDebug nvinfer_plugin_tensorrt_llm)

add_executable(executorExampleKvEvents executorExampleKvEvents.cpp)
target_link_libraries(executorExampleKvEvents nvinfer_plugin_tensorrt_llm
                      cxxopts::cxxopts)

add_executable(executorExampleLogitsProcessor
               executorExampleLogitsProcessor.cpp)
target_link_libraries(executorExampleLogitsProcessor
                      nvinfer_plugin_tensorrt_llm)

# Advanced
if(NOT TARGET cxxopts::cxxopts)
  set(CXXOPTS_SRC_DIR ${TRTLLM_DIR}/3rdparty/cxxopts)
  add_subdirectory(${CXXOPTS_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)
endif()

add_executable(executorExampleAdvanced executorExampleAdvanced.cpp)
target_link_libraries(executorExampleAdvanced nvinfer_plugin_tensorrt_llm
                      cxxopts::cxxopts)

# MultiInstance
if(ENABLE_MULTI_DEVICE)
  find_package(MPI REQUIRED)
  message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
  message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
  include_directories(${MPI_C_INCLUDE_DIRS})

  add_executable(executorExampleAdvancedMultiInstances
                 executorExampleAdvancedMultiInstances.cpp)
  target_link_libraries(
    executorExampleAdvancedMultiInstances nvinfer_plugin_tensorrt_llm
    cxxopts::cxxopts ${MPI_C_LIBRARIES})

  # FastLogits
  add_executable(executorExampleFastLogits executorExampleFastLogits.cpp)
  target_link_libraries(executorExampleFastLogits nvinfer_plugin_tensorrt_llm
                        cxxopts::cxxopts ${MPI_C_LIBRARIES})

  add_executable(executorExampleDisaggregated executorExampleDisaggregated.cpp)
  target_link_libraries(
    executorExampleDisaggregated nvinfer_plugin_tensorrt_llm cxxopts::cxxopts
    ${MPI_C_LIBRARIES})
endif()
