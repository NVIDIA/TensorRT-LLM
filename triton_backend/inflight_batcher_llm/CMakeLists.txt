# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of NVIDIA CORPORATION nor the names
# of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
# EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/set_ifndef.cmake)

set_ifndef(TRTLLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include_directories(${TRTLLM_DIR} ${TRTLLM_DIR}/cpp/include)

list(APPEND CMAKE_MODULE_PATH "${TRTLLM_DIR}/cpp/cmake/modules")

project(tritontensorrtllmbackend LANGUAGES C CXX)

add_compile_options("-DENABLE_MULTI_DEVICE=1")
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
option(USE_CXX11_ABI "Using CXX11 ABI of libstdc++" OFF)
message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")
if(USE_CXX11_ABI)
  add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
else()
  add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

find_package(CUDAToolkit REQUIRED)

#
# Options
#
# Must include options required for this project as well as any projects
# included in this one by FetchContent.
#
# TRITON_ENABLE_GPU is set to OFF as currently the code does not use any GPU
# related features since TRT-LLM backend manages the usage on GPUs itself.
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(BUILD_TESTS "Build Google tests" OFF)

if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(
    FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
endif()

set(TRITON_REPO_ORGANIZATION
    "https://github.com/triton-inference-server"
    CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/backend repo")

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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all repos that
# we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to hide all
# symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_tensorrtllm.ldscript
               libtriton_tensorrtllm.ldscript COPYONLY)

set(SRCS src/libtensorrtllm.cc src/model_instance_state.cc src/model_state.cc
         src/utils.cc src/namedTensor.cpp)

add_library(triton-tensorrt-llm-backend SHARED ${SRCS})

enable_language(CUDA)

find_package(CUDA ${CUDA_REQUIRED_VERSION} REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

find_library(
  tensorrt_llm libtensorrt_llm.so REQUIRED
  PATHS ${Python3_SITEARCH}/tensorrt_llm/libs
        ${TRTLLM_DIR}/cpp/build/tensorrt_llm)

find_library(
  nvinfer_plugin_tensorrt_llm libnvinfer_plugin_tensorrt_llm.so REQUIRED
  PATHS ${Python3_SITEARCH}/tensorrt_llm/libs
        ${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins)

find_program(
  TRTLLM_EXECUTOR_WORKER executorWorker REQUIRED
  PATHS ${Python3_SITEARCH}/tensorrt_llm/bin
        ${TRTLLM_DIR}/cpp/build/tensorrt_llm/executor_worker)
install(
  PROGRAMS ${TRTLLM_EXECUTOR_WORKER}
  DESTINATION ${CMAKE_BINARY_DIR}
  RENAME trtllmExecutorWorker)

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
find_library(
  NVIDIA_ML_LIB nvidia-ml
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
set(CUDA_LIBRARIES ${CUDART_LIB} ${NVIDIA_ML_LIB})

find_package(MPI REQUIRED)
message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_LIBRARIES: ${MPI_LIBRARIES}")

# NCCL dependencies
find_package(NCCL 2 REQUIRED)
set(NCCL_LIB NCCL::nccl)

# TRT dependencies
set(TensorRT_ROOT /usr/local/tensorrt)
find_package(TensorRT 10 REQUIRED)
add_definitions("-DTRT_LLM_USE_DIM64")

list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS})
include_directories(${COMMON_HEADER_DIRS})

target_include_directories(
  triton-tensorrt-llm-backend
  PRIVATE ${TRTLLM_DIR}/cpp
          ${TRTLLM_DIR}/cpp/include
          ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CUDA_INCLUDE_DIRS}
          ${CUDNN_ROOT_DIR}/include
          ${NCCL_INCLUDE_DIR}
          ${3RDPARTY_DIR}/cutlass/include
          ${MPI_INCLUDE_PATH}
          ${COMMON_HEADER_DIR})

target_compile_features(triton-tensorrt-llm-backend PRIVATE cxx_std_17)

set(COMPILE_OPTIONS
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-declarations
    -Wno-type-limits>
    $<$<CXX_COMPILER_ID:MSVC>:/Wall
    /D_WIN32_WINNT=0x0A00
    /EHsc>)

target_compile_options(triton-tensorrt-llm-backend PRIVATE ${COMPILE_OPTIONS})

if(TRITON_ENABLE_METRICS)
  list(APPEND REPORTER_SRCS
       src/custom_metrics_reporter/custom_metrics_reporter.cc)
  list(APPEND REPORTER_HDRS
       src/custom_metrics_reporter/custom_metrics_reporter.h)

  add_library(triton-custom-metrics-reporter-library EXCLUDE_FROM_ALL
              ${REPORTER_SRCS} ${REPORTER_HDRS})
  target_compile_features(triton-custom-metrics-reporter-library
                          PRIVATE cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(triton-custom-metrics-reporter-library
                           PRIVATE /W1 /D_WIN32_WINNT=0x0A00 /EHsc)
  else()
    target_compile_options(
      triton-custom-metrics-reporter-library
      PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations
              -Werror)
  endif()

  set_target_properties(triton-custom-metrics-reporter-library
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(
    triton-custom-metrics-reporter-library
    PUBLIC triton-common-json # from repo-common
           triton-common-logging # from repo-common
           triton-core-serverapi # from repo-core
           triton-core-serverstub # from repo-core
           triton-backend-utils # from repo-backend
           ${tensorrt_llm})

  target_compile_definitions(triton-tensorrt-llm-backend
                             PRIVATE TRITON_ENABLE_METRICS=1)
  target_link_libraries(triton-tensorrt-llm-backend
                        PRIVATE triton-custom-metrics-reporter-library)
endif()

target_link_libraries(
  triton-tensorrt-llm-backend
  PUBLIC ${tensorrt_llm}
         triton-core-serverapi # from repo-core
         triton-core-backendapi # from repo-core
         triton-core-serverstub # from repo-core
         triton-backend-utils # from repo-backend
         ${MPI_LIBRARIES}
         ${CUDA_LIBRARIES}
         TensorRT::NvInfer
         ${nvinfer_plugin_tensorrt_llm})

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0)

FetchContent_MakeAvailable(json)

target_link_libraries(triton-tensorrt-llm-backend
                      PRIVATE nlohmann_json::nlohmann_json)

if(WIN32)
  set_target_properties(
    triton-tensorrt-llm-backend PROPERTIES POSITION_INDEPENDENT_CODE ON
                                           OUTPUT_NAME triton_tensorrtllm)
else()
  set_target_properties(
    triton-tensorrt-llm-backend
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME triton_tensorrtllm
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorrtllm.ldscript
      LINK_FLAGS
      "-Wl,--version-script libtriton_tensorrtllm.ldscript -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined"
  )
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
