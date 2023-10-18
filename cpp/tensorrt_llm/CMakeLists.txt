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
set(TARGET_NAME tensorrt_llm)
set(SHARED_TARGET ${TARGET_NAME})
set(SHARED_TARGET
    ${SHARED_TARGET}
    PARENT_SCOPE)
set(STATIC_TARGET ${TARGET_NAME}_static)
set(STATIC_TARGET
    ${STATIC_TARGET}
    PARENT_SCOPE)
set(API_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

find_package(MPI REQUIRED)
message(STATUS "Using MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "Using MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include
                    ${API_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(runtime)

set(BATCH_MANAGER_TARGET tensorrt_llm_batch_manager_static)
set(BATCH_MANAGER_TARGET_ARCH "unknown")

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(NOT WIN32) # Linux
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(BATCH_MANAGER_TARGET_ARCH "x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(BATCH_MANAGER_TARGET_ARCH "aarch64-linux-gnu")
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
else() # Windows
  # AMD64, IA64, ARM64, EM64T, X86
  set(BATCH_MANAGER_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}-WINDOWS")
  string(TOLOWER ${BATCH_MANAGER_TARGET_ARCH} ${BATCH_MANAGER_TARGET_ARCH})
endif()

if(BUILD_BATCH_MANAGER)
  add_subdirectory(batch_manager)
else()
  add_library(${BATCH_MANAGER_TARGET} STATIC IMPORTED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)

  message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")

  if(USE_CXX11_ABI)
    set_property(
      TARGET ${BATCH_MANAGER_TARGET}
      PROPERTY
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/${BATCH_MANAGER_TARGET_ARCH}/libtensorrt_llm_batch_manager_static.a"
    )
  else()
    set_property(
      TARGET ${BATCH_MANAGER_TARGET}
      PROPERTY
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/${BATCH_MANAGER_TARGET_ARCH}/libtensorrt_llm_batch_manager_static.pre_cxx11.a"
    )
  endif()
endif()

set(TRTLLM_LINK_LIBS
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDNN_LIB}
    ${CMAKE_DL_LIBS}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIB}
    ${TRT_LIB}
    common_src
    kernels_src
    layers_src
    runtime_src
    ${BATCH_MANAGER_TARGET})

# ################################# SHARED LIBRARY
# ##############################################################################

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(${SHARED_TARGET} SHARED)

set_target_properties(
  ${SHARED_TARGET} PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
                              CXX_EXTENSIONS "NO")

if(NOT MSVC) # Unix-like compilers
  set(ALLOW_UNDEFINED_FLAG "-Wl, --no-undefined")
else() # MSVC
  set(UNDEFINED_FLAG "")
endif()

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS}
                                              ${UNDEFINED_FLAG})

# ################################# STATIC LIBRARY
# ##############################################################################

add_library(${STATIC_TARGET} STATIC)

set_target_properties(
  ${STATIC_TARGET} PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
                              CXX_EXTENSIONS "NO" POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${STATIC_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

# Cyclic dependency of batch manager on TRT-LLM
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE ${STATIC_TARGET})

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

add_subdirectory(plugins)
