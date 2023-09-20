#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
#

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)

project(tensorrt_llm LANGUAGES CXX)

# Build options
option(BUILD_PYT "Build in PyTorch TorchScript class mode" ON)
option(BUILD_TESTS "Build Google tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm/batch_manager/CMakeLists.txt")
  set(BUILD_BATCH_MANAGER_DEFAULT ON)
else()
  set(BUILD_BATCH_MANAGER_DEFAULT OFF)
endif()

option(BUILD_BATCH_MANAGER "Build batch manager from source"
       ${BUILD_BATCH_MANAGER_DEFAULT})

if(BUILD_BATCH_MANAGER)
  message(STATUS "Building batch manager")
else()
  message(STATUS "Importing batch manager")
endif()

if(BUILD_PYT)
  message(STATUS "Building PyTorch")
else()
  message(STATUS "Not building PyTorch")
endif()

if(BUILD_TESTS)
  message(STATUS "Building Google tests")
else()
  message(STATUS "Not building Google tests")
endif()

if(BUILD_BENCHMARKS)
  message(STATUS "Building benchmarks")
else()
  message(STATUS "Not building benchmarks")
endif()

# Determine CUDA version before enabling the language extension
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  execute_process(
    COMMAND
      "bash" "-c"
      "${CMAKE_CUDA_COMPILER} --version | egrep -o 'V[0-9]+.[0-9]+.[0-9]+' | cut -c2-"
    RESULT_VARIABLE _BASH_SUCCESS
    OUTPUT_VARIABLE CMAKE_CUDA_COMPILER_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _BASH_SUCCESS EQUAL 0)
    message(FATAL_ERROR "Failed to determine CUDA version")
  endif()
else()
  message(FATAL_ERROR "No CUDA compiler found")
endif()

set(CUDA_REQUIRED_VERSION "11.2")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS CUDA_REQUIRED_VERSION)
  message(
    FATAL_ERROR
      "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} must be at least ${CUDA_REQUIRED_VERSION}"
  )
endif()

# Initialize CMAKE_CUDA_ARCHITECTURES before enabling CUDA
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 89-real 90-real)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real)
  endif()
endif()

message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

# TODO: FindCUDA is deprecated and should be replaced by FindCUDAToolkit
# https://cmake.org/cmake/help/latest/module/FindCUDA.html
find_package(CUDA ${CUDA_REQUIRED_VERSION} REQUIRED)

message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

find_library(
  CUDNN_LIB cudnn
  HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR}
  PATH_SUFFIXES lib64 lib)
find_library(
  CUBLAS_LIB cublas
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUBLASLT_LIB cublasLt
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
find_library(
  CUDA_DRV_LIB cuda
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
set(CUDA_LIBRARIES ${CUDART_LIB})

find_library(RT_LIB rt)

set_ifndef(ENABLE_MULTI_DEVICE 1)
if(ENABLE_MULTI_DEVICE EQUAL 1)
  # NCCL dependencies
  set_ifndef(NCCL_LIB_DIR /usr/lib/x86_64-linux-gnu/)
  set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
  find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})
endif()

set(3RDPARTY_DIR ../3rdparty)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_ROOT_DIR}/include
                    ${NCCL_INCLUDE_DIR} ${3RDPARTY_DIR}/cutlass/include)

# TRT dependencies
set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})
find_library_create_target(nvuffparser nvparsers SHARED ${TRT_LIB_DIR})

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
    "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss -DENABLE_MULTI_DEVICE=${ENABLE_MULTI_DEVICE}"
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

if(BUILD_PYT)
  # Build TORCH_CUDA_ARCH_LIST
  set(TORCH_CUDA_ARCH_LIST "")
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_ARCH MATCHES "^([0-9])([0-9])(-real)*$")
      set(TORCH_ARCH "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
      message(FATAL_ERROR "${CUDA_ARCH} is not supported")
    endif()
    if(NOT CUDA_ARCH MATCHES "-real$")
      string(APPEND TORCH_ARCH "+PTX")
    endif()
    list(APPEND TORCH_CUDA_ARCH_LIST ${TORCH_ARCH})
  endforeach()

  message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
  # ignore values passed from the environment
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    message(
      WARNING
        "Ignoring environment variable TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}"
    )
  endif()
  unset(ENV{TORCH_CUDA_ARCH_LIST})

  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Found Python executable at ${Python3_EXECUTABLE}")
  message(STATUS "Found Python libraries at ${Python3_LIBRARY_DIRS}")
  link_directories("${Python3_LIBRARY_DIRS}")
  list(APPEND COMMON_HEADER_DIRS ${Python3_INCLUDE_DIRS})

  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} "-c"
      "from __future__ import print_function; import torch; print(torch.__version__,end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_VERSION)
  if(TORCH_VERSION VERSION_LESS "1.5.0")
    message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR)
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  find_package(Torch REQUIRED)

  message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
  add_compile_options(${TORCH_CXX_FLAGS})
  add_compile_definitions(TORCH_CUDA=1)
endif()

file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" VERSION_STRINGS
     REGEX "#define NV_TENSORRT_.*")
foreach(TYPE MAJOR MINOR PATCH BUILD)
  string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING
               ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

foreach(TYPE MAJOR MINOR PATCH)
  string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING
               ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION
    "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}"
    CACHE STRING "TensorRT project version")
set(TRT_SOVERSION
    "${TRT_SO_MAJOR}"
    CACHE STRING "TensorRT library so version")
message(
  STATUS
    "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}"
)

list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
include_directories(${COMMON_HEADER_DIRS})

add_subdirectory(tensorrt_llm)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
