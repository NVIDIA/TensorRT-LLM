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
# the License.
cmake_minimum_required(VERSION 3.18)
project(xqa LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89-real 90a-real)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_XQA_TESTS "Build XQA tests" OFF)

# todo: remove include_directories link_directories and link libs like
# CUDA::cuda_driver CUDA::cudart CUDA::nvrtc
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
                 ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -march=haswell -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas"
)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --expt-relaxed-constexpr -t 0 -res-usage"
)
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills"
)# -Werror -v
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -keep --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -keep")
# add_definitions(-DSPEC_DEC) set(CMAKE_CUDA_FLAGS_DEBUG
# "${CMAKE_CUDA_FLAGS_RELEASE}")

set(XQA_SOURCES
    "cuda_hint.cuh"
    "defines.h"
    "ldgsts.cuh"
    "mha.h"
    "mhaUtils.cuh"
    "mma.cuh"
    "platform.h"
    "utils.cuh"
    "utils.h"
    "mha_stdheaders.cuh"
    "gmma.cuh"
    "gmma_impl.cuh"
    "barriers.cuh"
    "tma.h"
    "mha_components.cuh"
    "mla_sm120.cuh"
    "mha.cu"
    "mha_sm90.cu"
    "mla_sm120.cu")

# For ${Python3_EXECUTABLE}
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(XQA_SOURCES_H ${CMAKE_CURRENT_BINARY_DIR}/xqa_sources.h)
add_custom_command(
  OUTPUT ${XQA_SOURCES_H}
  COMMAND ${Python3_EXECUTABLE} gen_cpp_header.py -o ${XQA_SOURCES_H}
          --cuda_root ${CUDAToolkit_LIBRARY_ROOT}
  COMMENT "Generating xqa_sources.h for XQAJIT..."
  DEPENDS gen_cpp_header.py ${XQA_SOURCES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM)
add_custom_target(xqa_sources_h DEPENDS ${XQA_SOURCES_H})

if(BUILD_XQA_TESTS)
  # Try to find system installed GTest first
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "System GTest not found, fetching from repository")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
  endif()

  # Try to find system installed Eigen first
  find_package(Eigen3 3.4 QUIET)
  if(NOT Eigen3_FOUND)
    message(STATUS "System Eigen not found, fetching from repository")
    include(FetchContent)
    FetchContent_Declare(
      eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0)
    FetchContent_MakeAvailable(eigen)
  endif()

  enable_testing()
  add_executable(
    unitTests
    mha.cu
    mha_sm90.cu
    mla_sm120.cu
    tensorMap.cpp
    test/warmup.cu
    test/test.cpp
    test/refAttention.cpp)
  target_include_directories(unitTests PUBLIC ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(unitTests PUBLIC GTest::gtest GTest::gtest_main cuda
                                         Eigen3::Eigen)

  find_library(
    NVRTC_LIB nvrtc
    HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib
    PATH_SUFFIXES lib64 lib lib/x64)
  if(NOT NVRTC_LIB)
    message("Nvrtc not found")
    add_definitions(-DENABLE_NVRTC=0)
  else()
    add_definitions(-DENABLE_NVRTC=1)
    target_link_libraries(unitTests PUBLIC ${NVRTC_LIB})
    # Generate xqa_sources.h for nvrtc testing.
    include_directories(${PROJECT_BINARY_DIR})
    set(GENERATED_XQA_SOURCES
        "${CMAKE_CURRENT_BINARY_DIR}/generated/xqa_sources.h")
    add_custom_command(
      OUTPUT ${GENERATED_XQA_SOURCES}
      COMMAND
        ./gen_cpp_header.py -o ${GENERATED_XQA_SOURCES} --embed-cuda-headers
        --cuda_root ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/..
      DEPENDS gen_cpp_header.py ${XQA_SOURCES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_sources(unitTests PUBLIC ${GENERATED_XQA_SOURCES})
  endif()

  add_test(NAME unitTests COMMAND unitTests)
endif()
