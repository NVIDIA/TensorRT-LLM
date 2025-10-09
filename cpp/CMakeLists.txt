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

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(resolve_dirs)
include(parse_make_options)
include(cuda_configuration)
include(sanitizers)

project(tensorrt_llm LANGUAGES CXX)

# Build options
option(BUILD_PYT "Build in PyTorch TorchScript class mode" ON)
option(BUILD_TESTS "Build Google tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_DEEP_EP "Build the Deep EP module" ON)
option(BUILD_DEEP_GEMM "Build the DeepGEMM module" ON)
option(BUILD_FLASH_MLA "Build the FlashMLA module" ON)
option(BUILD_MICRO_BENCHMARKS "Build C++ micro benchmarks" OFF)
option(NVTX_DISABLE "Disable all NVTX features" ON)
option(WARNING_IS_ERROR "Treat all warnings as errors" OFF)
option(FAST_BUILD "Skip compiling some kernels to accelerate compiling" OFF)
option(FAST_MATH "Compiling in fast math mode" OFF)
option(INDEX_RANGE_CHECK "Compiling with index range checks" OFF)
option(COMPRESS_FATBIN "Compress everything in fatbin" ON)
option(TIMING_NVCC "Enable nvcc build timing report" OFF)
option(ENABLE_MULTI_DEVICE
       "Enable building with multi device support (requires NCCL, MPI,...)" ON)
option(ENABLE_UCX "Enable building with UCX (Uniform Communication X) support"
       ON)
option(NVRTC_DYNAMIC_LINKING "Link against the dynamic NVRTC libraries" OFF)
option(CUBLAS_DYNAMIC_LINKING
       "Link against the dynamic cublas/cublasLt libraries" ON)
option(CURAND_DYNAMIC_LINKING "Link against the dynamic curand library" ON)

option(ENABLE_NVSHMEM "Enable building with NVSHMEM support" OFF)
option(USING_OSS_CUTLASS_LOW_LATENCY_GEMM
       "Using open sourced Cutlass low latency gemm kernel" ON)
option(USING_OSS_CUTLASS_FP4_GEMM "Using open sourced Cutlass fp4 gemm kernel"
       ON)
option(ENABLE_CUBLASLT_FP4_GEMM "Enable cuBLASLt FP4 GEMM support" ON)
if(NOT ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "12.8")
  set(ENABLE_CUBLASLT_FP4_GEMM
      OFF
      CACHE BOOL "" FORCE)
  message(
    STATUS
      "CUDA ${CUDAToolkit_VERSION} < 12.8: disabling ENABLE_CUBLASLT_FP4_GEMM")
endif()
option(USING_OSS_CUTLASS_MOE_GEMM "Using open sourced Cutlass moe gemm kernel"
       ON)
option(USING_OSS_CUTLASS_ALLREDUCE_GEMM
       "Using open sourced Cutlass AR gemm kernel" ON)

message(STATUS "ENABLE_NVSHMEM is ${ENABLE_NVSHMEM}")

if(NVTX_DISABLE)
  add_compile_definitions("NVTX_DISABLE")
  message(STATUS "NVTX is disabled")
else()
  message(STATUS "NVTX is enabled")
endif()

# Add TensorRT LLM Gen export interface and CUDA support
add_compile_definitions("TLLM_GEN_EXPORT_INTERFACE")
add_compile_definitions("TLLM_ENABLE_CUDA")

set(BINDING_TYPE
    "nanobind"
    CACHE STRING
          "Binding type of Python bindings for C++ runtime and batch manager")

set(INTERNAL_CUTLASS_KERNELS_PATH
    ""
    CACHE
      PATH
      "The path to internal cutlass kernels sources. Prebuilt binary for internal cutlass kernels will be used if this is not set."
)
if(INTERNAL_CUTLASS_KERNELS_PATH)
  message(
    STATUS
      "Building internal cutlass kernels from source at ${INTERNAL_CUTLASS_KERNELS_PATH}"
  )
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
  COMMAND ${Python_EXECUTABLE} -c "import version; print(version.__version__)"
  WORKING_DIRECTORY ${TRTLLM_VERSION_DIR}
  OUTPUT_VARIABLE TRTLLM_VERSION
  RESULT_VARIABLE TRTLLM_VERSION_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(TRTLLM_VERSION_RESULT EQUAL 0)
  message(STATUS "TensorRT LLM version: ${TRTLLM_VERSION}")
else()
  message(FATAL_ERROR "Failed to determine TensorRT LLM version")
endif()

configure_file(
  cmake/templates/version.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt_llm/executor/version.h)

setup_cuda_compiler()

enable_language(C CXX CUDA)

# Configure CUDA Architectures after enabling CUDA.

# Old CMake rejects family conditional architectures during enable_language, But
# after that CMake handles it just fine.
setup_cuda_architectures()

set(CUDA_DRV_LIB CUDA::cuda_driver)
set(CUDA_NVML_LIB CUDA::nvml)
set(CUDA_RT_LIB CUDA::cudart_static)
set(NVPTX_LIB CUDA::nvptxcompiler_static)

set(CUDA_TOOLKIT_COMPONENTS cudart_static cuda_driver nvml nvptxcompiler_static)

if(CUBLAS_DYNAMIC_LINKING)
  set(CUBLAS_LIB CUDA::cublas)
  set(CUBLASLT_LIB CUDA::cublasLt)
  list(APPEND CUDA_TOOLKIT_COMPONENTS cublas cublasLt)
else()
  if(WIN32)
    message(FATAL_ERROR "Static cublas not available on windows")
  endif()
  message(DEBUG "Linking with static cublas libs")

  set(CUBLAS_LIB CUDA::cublas_static)
  set(CUBLASLT_LIB CUDA::cublasLt_static)
  list(APPEND CUDA_TOOLKIT_COMPONENTS cublas_static cublasLt_static)
endif()

if(CURAND_DYNAMIC_LINKING)
  set(CURAND_LIB CUDA::curand)
  list(APPEND CUDA_TOOLKIT_COMPONENTS curand)
else()
  if(WIN32)
    message(FATAL_ERROR "Static curand not available on windows")
  endif()
  message(DEBUG "Linking with static curand lib")

  set(CURAND_LIB CUDA::curand_static)
  list(APPEND CUDA_TOOLKIT_COMPONENTS curand_static)
endif()

if(NVRTC_DYNAMIC_LINKING)
  set(NVRTC_LIB CUDA::nvrtc)
  set(NVRTC_BUILTINS_LIB CUDA::nvrtc_builtins)
  list(APPEND CUDA_TOOLKIT_COMPONENTS nvrtc nvrtc_builtins)
else()
  set(NVRTC_LIB CUDA::nvrtc_static)
  set(NVRTC_BUILTINS_LIB CUDA::nvrtc_builtins_static)
  list(APPEND CUDA_TOOLKIT_COMPONENTS nvrtc_static nvrtc_builtins_static)
endif()

find_package(CUDAToolkit 11.2 REQUIRED COMPONENTS ${CUDA_TOOLKIT_COMPONENTS})

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA_NVML_LIB: ${CUDA_NVML_LIB}")

# Prevent CMake from creating a response file for CUDA compiler, so clangd can
# pick up on the includes
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

find_library(RT_LIB rt)

if(ENABLE_MULTI_DEVICE)
  # NCCL dependencies
  find_package(NCCL 2 REQUIRED)
  set(NCCL_LIB NCCL::nccl)
endif()

# TRT dependencies
find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser)
set(TRT_LIB TensorRT::NvInfer)

get_filename_component(TRT_LLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

set(3RDPARTY_DIR ${TRT_LLM_ROOT_DIR}/3rdparty)
if(BINDING_TYPE STREQUAL "pybind"
   OR BUILD_DEEP_EP
   OR BUILD_DEEP_GEMM)
  add_subdirectory(${3RDPARTY_DIR}/pybind11
                   ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
endif()
if(BINDING_TYPE STREQUAL "nanobind")
  add_subdirectory(${3RDPARTY_DIR}/nanobind
                   ${CMAKE_CURRENT_BINARY_DIR}/nanobind)
endif()

# include as system to suppress warnings
include_directories(
  SYSTEM
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}/cccl
  ${CUDNN_ROOT_DIR}/include
  $<TARGET_PROPERTY:TensorRT::NvInfer,INTERFACE_INCLUDE_DIRECTORIES>
  ${3RDPARTY_DIR}/cutlass/include
  ${3RDPARTY_DIR}/cutlass/tools/util/include
  ${3RDPARTY_DIR}/NVTX/include
  ${3RDPARTY_DIR}/json/include)
if(BINDING_TYPE STREQUAL "pybind"
   OR BUILD_DEEP_EP
   OR BUILD_DEEP_GEMM)
  include_directories(${3RDPARTY_DIR}/pybind11/include)
endif()
if(BINDING_TYPE STREQUAL "nanobind")
  include_directories(${3RDPARTY_DIR}/nanobind/include)
endif()

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

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "12.8")
  add_definitions("-DENABLE_FP4")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 12.8, enable -DENABLE_FP4 flag"
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
# project.

# set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}
# --generate-line-info")

# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss ")

# note: cmake expr generation $<BOOL:${ENABLE_MULTI_DEVICE}> is a build time
# evaluation so hard to debug at cmake time
if(ENABLE_MULTI_DEVICE)
  # Add target definitions for both C++ and CUDA
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_MULTI_DEVICE=1>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_MULTI_DEVICE=1>)
else()
  # Add target definitions for both C++ and CUDA
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_MULTI_DEVICE=0>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_MULTI_DEVICE=0>)
endif()

if(ENABLE_NVSHMEM)
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_NVSHMEM=1>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_NVSHMEM=1>)
else()
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_NVSHMEM=0>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_NVSHMEM=0>)
endif()

# Fix linking issue with TRT 10, the detailed description about `--mcmodel` can
# be found in
# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mcmodel_003dmedium-1
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-relax")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-relax")
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

setup_sanitizers()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(FAST_MATH)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()
if(COMPRESS_FATBIN)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fatbin-options -compress-all")
endif()
if(NVCC_TIMING)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} --time ${CMAKE_CURRENT_BINARY_DIR}/nvcc-timing.csv")
endif()
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

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
  # ignore values passed from the environment
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    message(
      WARNING
        "Ignoring environment variable TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}"
    )
  endif()
  unset(ENV{TORCH_CUDA_ARCH_LIST})
  # Torch maintains custom logic to add CUDA architecture flags into
  # CMAKE_CUDA_FLAGS based on TORCH_CUDA_ARCH_LIST variable, instead of using
  # the native support introduced in newer CMake versions. And it always tries
  # to add some flags, even given empty TORCH_CUDA_ARCH_LIST.

  # We prefer CMake's native support to be able to easily customize the CUDA
  # architectures to be compiled for, for each kernel individually. So we set
  # TORCH_CUDA_ARCH_LIST to a placeholder value and remove the generated flags
  # then to effectively prevent Torch from adding CUDA architecture flags.
  message(
    STATUS
      "Set TORCH_CUDA_ARCH_LIST to placeholder value \"8.0\" to make Torch happy. "
      "This is NOT the list of architectures that will be compiled for.")
  set(TORCH_CUDA_ARCH_LIST "8.0")

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
  if(NOT _PYTHON_SUCCESS EQUAL 0)
    message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  set(USE_SYSTEM_NVTX ON)
  set(nvtx3_dir ${3RDPARTY_DIR}/NVTX/include)
  set(CMAKE_CUDA_ARCHITECTURES_BACKUP ${CMAKE_CUDA_ARCHITECTURES})
  find_package(Torch REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_BACKUP})
  message(
    STATUS
      "Removing Torch generated placeholder CUDA architecture flags: -gencode arch=compute_80,code=sm_80."
  )
  string(REPLACE "-gencode arch=compute_80,code=sm_80 " "" CMAKE_CUDA_FLAGS_NEW
                 "${CMAKE_CUDA_FLAGS}")
  if("${CMAKE_CUDA_FLAGS_NEW}" STREQUAL "${CMAKE_CUDA_FLAGS}")
    message(
      FATAL_ERROR
        "Torch didn't generate expected placeholder CUDA architecture flags.")
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_NEW}")

  add_compile_definitions(TORCH_CUDA=1)

  if(DEFINED TORCH_CXX_FLAGS)
    message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
    add_compile_options(${TORCH_CXX_FLAGS})
    if(DEFINED USE_CXX11_ABI)
      parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
      if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
         AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL
             ${USE_CXX11_ABI})
        message(
          WARNING
            "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
            "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}, and the project "
            "setting will be discarded.")
      endif()
    endif()
  endif()
else()
  if(NOT WIN32)
    if(NOT USE_CXX11_ABI)
      add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
    message(STATUS "Build without PyTorch, USE_CXX11_ABI=${USE_CXX11_ABI}")
  endif()
endif()

# Defer UCX/UCXX setup until after USE_CXX11_ABI is well defined, as UCXX will
# need to be built to have aligned symbols
if(ENABLE_UCX)
  # Only enable UCX related features if the system has UCX library
  find_package(ucx)
  if(NOT ${ucx_FOUND})
    set(ENABLE_UCX 0)
  else()
    if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
      if(EXISTS "${3RDPARTY_DIR}/ucxx/fetch_rapids.cmake")
        file(READ "${3RDPARTY_DIR}/ucxx/fetch_rapids.cmake" FILE_CONTENTS)
        string(
          REPLACE "https://raw.githubusercontent.com/rapidsai/rapids-cmake"
                  "$ENV{GITHUB_MIRROR}/rapidsai/rapids-cmake/raw/refs/heads"
                  FILE_CONTENTS "${FILE_CONTENTS}")
        file(WRITE "${3RDPARTY_DIR}/ucxx/fetch_rapids.cmake" "${FILE_CONTENTS}")
        message(WARNING "Replace UCXX fetch_rapids.cmake with internal mirror")
      endif()
    endif()
    # installing ucxx via add_subdirectory results in strange cudart linking
    # error, thus using their installation script to isolate the installation
    # process until the issue is understood. And always trigger the build so
    # that change in USE_CXX11_ABI will not be ignored.
    execute_process(
      COMMAND
        ${CMAKE_COMMAND} -E env LIB_BUILD_DIR=${CMAKE_BINARY_DIR}/ucxx/build
        ${3RDPARTY_DIR}/ucxx/build.sh libucxx -n
        --cmake-args=\"-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}\"
      OUTPUT_VARIABLE UCXX_BUILD_OUTPUT
      RESULT_VARIABLE UCXX_BUILD_RESULT)
    if(UCXX_BUILD_RESULT)
      message(${UCXX_BUILD_OUTPUT})
      message(FATAL_ERROR "ucxx build failed")
    endif()
    find_package(ucxx REQUIRED PATHS ${CMAKE_BINARY_DIR}/ucxx/build
                 NO_DEFAULT_PATH)
  endif()
  find_package(NIXL)
endif()
if(ENABLE_UCX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_UCX=1")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_UCX=0")
endif()

list(APPEND COMMON_HEADER_DIRS)
include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

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

set(BUILD_WHEEL_TARGETS
    tensorrt_llm;nvinfer_plugin_tensorrt_llm
    CACHE STRING "Targets used to build wheel")
add_custom_target(build_wheel_targets DEPENDS ${BUILD_WHEEL_TARGETS})
