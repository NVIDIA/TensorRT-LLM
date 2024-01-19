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
include(cmake/modules/resolve_dirs.cmake)
include(cmake/modules/parse_make_options.cmake)

project(tensorrt_llm LANGUAGES CXX)

# Build options
option(BUILD_PYT "Build in PyTorch TorchScript class mode" ON)
option(BUILD_PYBIND "Build Python bindings for C++ runtime and batch manager"
       ON)
option(BUILD_TESTS "Build Google tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(NVTX_DISABLE "Disable all NVTX features" ON)
option(WARNING_IS_ERROR "Treat all warnings as errors" OFF)
option(FAST_BUILD "Skip compiling some kernels to accelerate compiling" OFF)
option(FAST_MATH "Compiling in fast math mode" OFF)

if(NVTX_DISABLE)
  add_compile_definitions("NVTX_DISABLE")
  message(STATUS "NVTX is disabled")
else()
  message(STATUS "NVTX is enabled")
endif()

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

if(FAST_BUILD)
  add_compile_definitions("FAST_BUILD")
  message(WARNING "Skip some kernels to accelerate compilation")
endif()

# Determine CUDA version before enabling the language extension
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  if(NOT WIN32) # Linux
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

  else() # Windows
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} --version
      OUTPUT_VARIABLE versionString
      RESULT_VARIABLE versionResult)

    if(versionResult EQUAL 0 AND versionString MATCHES
                                 "V[0-9]+\\.[0-9]+\\.[0-9]+")
      string(REGEX REPLACE "V" "" version ${CMAKE_MATCH_0})
      set(CMAKE_CUDA_COMPILER_VERSION "${version}")
    else()
      message(FATAL_ERROR "Failed to determine CUDA version")
    endif()
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

enable_language(C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

find_library(
  CUDNN_LIB cudnn
  HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES lib64 lib lib/x64)
find_library(
  CUBLAS_LIB cublas
  HINTS ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUBLASLT_LIB cublasLt
  HINTS ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
  CUDA_DRV_LIB cuda
  HINTS ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES stubs lib lib64 lib/stubs lib64/stubs)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_library(RT_LIB rt)

set_ifndef(ENABLE_MULTI_DEVICE 1)
if(ENABLE_MULTI_DEVICE EQUAL 1)
  # NCCL dependencies
  set_ifndef(NCCL_LIB_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/)
  set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
  find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})
endif()

get_filename_component(TRT_LLM_ROOT_DIR ${CMAKE_SOURCE_DIR} PATH)

set(3RDPARTY_DIR ${TRT_LLM_ROOT_DIR}/3rdparty)
include_directories(
  ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_ROOT_DIR}/include ${NCCL_INCLUDE_DIR}
  ${3RDPARTY_DIR}/cutlass/include ${3RDPARTY_DIR}/NVTX/include
  ${3RDPARTY_DIR}/json/include)

# TRT dependencies
set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_INCLUDE_DIR /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

# MPI MPI isn't used until tensorrt_llm/CMakeLists.txt is invoked. However, if
# it's not called before "CMAKE_CXX_FLAGS" is set, it breaks on Windows for some
# reason, so we just call it here as a workaround.
find_package(MPI REQUIRED)
add_definitions("-DOMPI_SKIP_MPICXX")

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss -DENABLE_MULTI_DEVICE=${ENABLE_MULTI_DEVICE}"
)

# Disable deprecated declarations warnings
if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
else()
  # /wd4996 is the Windows equivalent to turn off warnings for deprecated
  # declarations
  set(CMAKE_CXX_FLAGS "/wd4996 ${CMAKE_CXX_FLAGS}")
endif()

# A Windows header file defines max() and min() macros, which break our macro
# declarations.
if(WIN32)
  set(CMAKE_CXX_FLAGS "/DNOMINMAX ${CMAKE_CXX_FLAGS}")
endif()

if((MSVC))
  if((MSVC_VERSION GREATER_EQUAL 1914))
    # MSVC does not apply the correct __cplusplus version per the C++ standard
    # by default. This is required for compiling CUTLASS 3.0 kernels on windows
    # with C++-17 constexpr enabled. The 2017 15.7 MSVC adds /Zc:__cplusplus to
    # set __cplusplus to 201703 with std=c++17. See
    # https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus for
    # more info.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  /Zc:__cplusplus")
  else()
    message(
      FATAL_ERROR
        "Build is only supported with Visual Studio 2017 version 15.7 or higher"
    )
  endif()
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(FAST_MATH)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
  message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

if(NOT WIN32 AND NOT DEFINED USE_CXX11_ABI)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)
  # Convert the bool variable to integer.
  if(USE_CXX11_ABI)
    set(USE_CXX11_ABI 1)
  else()
    set(USE_CXX11_ABI 0)
  endif()
  message(STATUS "USE_CXX11_ABI is set by python Torch to ${USE_CXX11_ABI}")
endif()

if(BUILD_PYT)
  # Build TORCH_CUDA_ARCH_LIST
  set(TORCH_CUDA_ARCH_LIST "")
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_ARCH MATCHES "^([0-9])([0-9])(-real)*$")
      set(TORCH_ARCH "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    elseif(CUDA_ARCH STREQUAL "native")
      set(TORCH_ARCH "Auto")
    else()
      message(FATAL_ERROR "${CUDA_ARCH} is not supported")
    endif()
    if(NOT CUDA_ARCH MATCHES "-real$" AND NOT CUDA_ARCH STREQUAL "native")
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

  if(DEFINED USE_CXX11_ABI)
    parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
    if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
       AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL ${USE_CXX11_ABI})
      message(
        WARNING
          "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
          "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}, and the project "
          "setting will be discarded.")
    endif()
  endif()

elseif(NOT WIN32)
  if(NOT USE_CXX11_ABI)
    add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  message(STATUS "Build without PyTorch, USE_CXX11_ABI=${USE_CXX11_ABI}")
endif()

file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" VERSION_STRINGS
     REGEX "#define NV_TENSORRT_.*")
foreach(TYPE MAJOR MINOR PATCH BUILD)
  string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING
               ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]+" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION
    "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}"
    CACHE STRING "TensorRT project version")
set(TRT_SOVERSION
    "${TRT_MAJOR}"
    CACHE STRING "TensorRT library so version")
message(
  STATUS
    "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}"
)

list(APPEND COMMON_HEADER_DIRS)
include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})

add_subdirectory(tensorrt_llm)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(${TRT_LLM_ROOT_DIR}/benchmarks/cpp
                   ${CMAKE_BINARY_DIR}/benchmarks)
endif()

# Measure the compile time
option(MEASURE_BUILD_TIME "Measure the build time of each module" OFF)
if(MEASURE_BUILD_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif()
