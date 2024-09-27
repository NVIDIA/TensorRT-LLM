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
option(BUILD_MICRO_BENCHMARKS "Build C++ micro benchmarks" OFF)
option(NVTX_DISABLE "Disable all NVTX features" ON)
option(WARNING_IS_ERROR "Treat all warnings as errors" OFF)
option(FAST_BUILD "Skip compiling some kernels to accelerate compiling" OFF)
option(FAST_MATH "Compiling in fast math mode" OFF)
option(INDEX_RANGE_CHECK "Compiling with index range checks" OFF)

# Always use static NVRTC for IP protection reasons.
set(USE_SHARED_NVRTC OFF)

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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm/executor/CMakeLists.txt")
  set(BUILD_EXECUTOR_DEFAULT ON)
else()
  set(BUILD_EXECUTOR_DEFAULT OFF)
endif()

option(BUILD_EXECUTOR "Build executor from source" ${BUILD_EXECUTOR_DEFAULT})

if(BUILD_EXECUTOR)
  message(STATUS "Building executor")
else()
  message(STATUS "Importing executor")
endif()

if(EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/CMakeLists.txt"
)
  set(BUILD_NVRTC_WRAPPER_DEFAULT ON)
else()
  set(BUILD_NVRTC_WRAPPER_DEFAULT OFF)
endif()

option(BUILD_NVRTC_WRAPPER "Build nvrtc wrapper from source"
       ${BUILD_NVRTC_WRAPPER_DEFAULT})

if(BUILD_NVRTC_WRAPPER)
  message(STATUS "Building nvrtc wrapper")
else()
  message(STATUS "Importing nvrtc wrapper")
endif()

if(EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm/kernels/internal_cutlass_kernels/CMakeLists.txt"
)
  set(BUILD_INTERNAL_CUTLASS_KERNELS_DEFAULT ON)
else()
  set(BUILD_INTERNAL_CUTLASS_KERNELS_DEFAULT OFF)
endif()
option(BUILD_INTERNAL_CUTLASS_KERNELS
       "Build internal cutlass kernels from source"
       ${BUILD_INTERNAL_CUTLASS_KERNELS_DEFAULT})

if(BUILD_INTERNAL_CUTLASS_KERNELS)
  message(STATUS "Building internal cutlass kernels")
else()
  message(STATUS "Importing internal cutlass kernels")
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

if(BUILD_MICRO_BENCHMARKS)
  message(STATUS "Building C++ micro benchmarks")
else()
  message(STATUS "Not building C++ micro benchmarks")
endif()

if(FAST_BUILD)
  add_compile_definitions("FAST_BUILD")
  message(WARNING "Skip some kernels to accelerate compilation")
endif()

if(INDEX_RANGE_CHECK)
  add_compile_definitions("INDEX_RANGE_CHECK")
  message(WARNING "Check index range to detect OOB accesses")
endif()

# Read the project version
set(TRTLLM_VERSION_DIR ${PROJECT_SOURCE_DIR}/../tensorrt_llm)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS
                                    ${TRTLLM_VERSION_DIR}/version.py)
execute_process(
  COMMAND python3 -c "import version; print(version.__version__)"
  WORKING_DIRECTORY ${TRTLLM_VERSION_DIR}
  OUTPUT_VARIABLE TRTLLM_VERSION
  RESULT_VARIABLE TRTLLM_VERSION_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(TRTLLM_VERSION_RESULT EQUAL 0)
  message(STATUS "TensorRT-LLM version: ${TRTLLM_VERSION}")
else()
  message(FATAL_ERROR "Failed to determine Tensorrt-LLM version")
endif()

configure_file(
  cmake/templates/version.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt_llm/executor/version.h)

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

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  # Detect highest available compute capability
  set(OUTPUTFILE ${PROJECT_BINARY_DIR}/detect_cuda_arch)
  set(CUDAFILE ${CMAKE_SOURCE_DIR}/cmake/utils/detect_cuda_arch.cu)
  execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o
                          ${OUTPUTFILE})
  message(VERBOSE "Detecting native CUDA compute capability")
  execute_process(
    COMMAND ${OUTPUTFILE}
    RESULT_VARIABLE CUDA_RETURN_CODE
    OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)
  if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
    message(WARNING "Detecting native CUDA compute capability - fail")
    message(
      WARNING "CUDA compute capability detection failed, compiling for 'all'")
    set(CMAKE_CUDA_ARCHITECTURES_ORIG "all")
  else()
    message(STATUS "Detecting native CUDA compute capability - done")
    set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CUDA_ARCH_OUTPUT}")
  endif()
else()
  # Store CMAKE_CUDA_ARCHITECTURES for later use since torch sets this to "OFF"
  set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES_ORIG}")

enable_language(C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

# Prevent CMake from creating a response file for CUDA compiler, so clangd can
# pick up on the includes
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

if(USE_SHARED_NVRTC)
  if(WIN32)
    message(FATAL_ERROR "Cannot use NVRTC shared library on Windows.")
  else()
    find_library(
      NVRTC_LIB nvrtc
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
    find_library(
      NVRTC_BUILTINS_LIB nvrtc-builtins
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
  endif()
else()
  if(WIN32)
    find_library(
      NVRTC_LIB nvrtc
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
  else()
    find_library(
      NVRTC_LIB nvrtc_static
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
    find_library(
      NVRTC_BUILTINS_LIB nvrtc-builtins_static
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
    find_library(
      NVPTXCOMPILER_LIB nvptxcompiler_static
      HINTS ${CUDAToolkit_LIBRARY_DIR}
      PATH_SUFFIXES lib64 lib lib/x64)
  endif()
endif()

set(CUBLAS_LIB CUDA::cublas)
set(CUBLASLT_LIB CUDA::cublasLt)
set(CUDA_DRV_LIB CUDA::cuda_driver)
set(CUDA_NVML_LIB CUDA::nvml)
set(CUDA_RT_LIB CUDA::cudart_static)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_library(RT_LIB rt)

set_ifndef(ENABLE_MULTI_DEVICE 1)
if(ENABLE_MULTI_DEVICE)
  # NCCL dependencies
  set_ifndef(NCCL_LIB_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/)
  set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
  find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})
endif()

get_filename_component(TRT_LLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

set(3RDPARTY_DIR ${TRT_LLM_ROOT_DIR}/3rdparty)
include_directories(
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDNN_ROOT_DIR}/include
  ${NCCL_INCLUDE_DIR}
  ${3RDPARTY_DIR}/cutlass/include
  ${3RDPARTY_DIR}/cutlass/tools/util/include
  ${3RDPARTY_DIR}/NVTX/include
  ${3RDPARTY_DIR}/json/include)

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

if(ENABLE_MULTI_DEVICE)
  # MPI MPI isn't used until tensorrt_llm/CMakeLists.txt is invoked. However, if
  # it's not called before "CMAKE_CXX_FLAGS" is set, it breaks on Windows for
  # some reason, so we just call it here as a workaround.
  find_package(MPI REQUIRED)
  add_definitions("-DOMPI_SKIP_MPICXX")
endif()

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fno-inline")
endif()

# Note: The following are desirable settings that should be enabled if we
# decrease shared library size. See e.g.
# https://github.com/rapidsai/cudf/pull/6134 for a similar issue in another
# project. set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
# "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss -DENABLE_MULTI_DEVICE=${ENABLE_MULTI_DEVICE}"
)

# Fix linking issue with TRT 10, the detailed description about `--mcmodel` can
# be found in
# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mcmodel_003dmedium-1
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium  -Wl,--no-relax")
endif()

# Disable deprecated declarations warnings
if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
else()
  # /wd4996 is the Windows equivalent to turn off warnings for deprecated
  # declarations

  # /wd4505
  # https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements-2019?view=msvc-170#warning-for-unused-internal-linkage-functions
  # "warning C4505: <>: unreferenced function with internal linkage has been
  # removed"

  # /wd4100
  # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-4-c4100?view=msvc-170
  # warning C4100: 'c': unreferenced formal parameter

  set(CMAKE_CXX_FLAGS "/wd4996 /wd4505 /wd4100 ${CMAKE_CXX_FLAGS}")
endif()

# A Windows header file defines max() and min() macros, which break our macro
# declarations.
if(WIN32)
  set(CMAKE_CXX_FLAGS "/DNOMINMAX ${CMAKE_CXX_FLAGS}")
endif()

if((WIN32))
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

# Defer UCX/UCXX setup until after USE_CXX11_ABI is well defined, as UCXX will
# need to be built to have aligned symbols
set_ifndef(ENABLE_UCX 0)
if(ENABLE_UCX)
  # Only enable UCX related features if the system has UCX library
  find_package(ucx)
  if(NOT ${ucx_FOUND})
    set(ENABLE_UCX 0)
  else()
    # installing ucxx via add_subdirectory results in strange cudart linking
    # error, thus using their installation script to isolate the installation
    # process until the issue is understood. And always trigger the build so
    # that change in USE_CXX11_ABI will not be ignored.
    execute_process(
      COMMAND
        ${3RDPARTY_DIR}/ucxx/build.sh libucxx -n
        --cmake-args=\"-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}\"
        COMMAND_ECHO STDOUT)
    find_package(ucxx REQUIRED PATHS ${3RDPARTY_DIR}/ucxx/cpp/build
                 NO_DEFAULT_PATH)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_UCX=${ENABLE_UCX}")

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

if(${TRT_MAJOR} LESS 10)
  message(FATAL_ERROR "TensorRT version must be at least 10.0")
endif()

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

if(BUILD_MICRO_BENCHMARKS)
  add_subdirectory(${TRT_LLM_ROOT_DIR}/cpp/micro_benchmarks
                   ${CMAKE_BINARY_DIR}/micro_benchmarks)
endif()

# Measure the compile time
option(MEASURE_BUILD_TIME "Measure the build time of each module" OFF)
if(MEASURE_BUILD_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif()
