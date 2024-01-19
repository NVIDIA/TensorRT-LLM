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
message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include
                    ${API_INCLUDE_DIR} ${MPI_C_INCLUDE_DIRS})

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(runtime)

set(BATCH_MANAGER_TARGET tensorrt_llm_batch_manager_static)
set(BATCH_MANAGER_TARGET_ARCH "unknown")

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(NOT WIN32) # Linux
  execute_process(
    COMMAND grep -oP "(?<=^ID=).+" /etc/os-release
    COMMAND tr -d "\""
    COMMAND tr -d "\n"
    RESULT_VARIABLE _OS_ID_SUCCESS
    OUTPUT_VARIABLE OS_ID)
  execute_process(
    COMMAND grep -oP "(?<=^VERSION_ID=).+" /etc/os-release
    COMMAND tr -d "\""
    COMMAND tr -d "\n"
    RESULT_VARIABLE _OS_VERSION_ID_SUCCESS
    OUTPUT_VARIABLE OS_VERSION_ID)
  message(STATUS "Operating System: ${OS_ID}, ${OS_VERSION_ID}")

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(BATCH_MANAGER_TARGET_ARCH "x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(BATCH_MANAGER_TARGET_ARCH "aarch64-linux-gnu")
    if(NOT ${OS_ID} MATCHES "ubuntu" OR ${OS_VERSION_ID} VERSION_LESS 22.04)
      message(
        FATAL_ERROR
          "The minimum system requirement for aarch64 is Ubuntu 22.04.")
    endif()
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
else() # Windows
  # AMD64, IA64, ARM64, EM64T, X86
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    set(BATCH_MANAGER_TARGET_ARCH "x86_64-windows-msvc")
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

if(BUILD_BATCH_MANAGER)
  add_subdirectory(batch_manager)
else()
  add_library(${BATCH_MANAGER_TARGET} STATIC IMPORTED)
  if(NOT WIN32) # Linux
    if(USE_CXX11_ABI)
      set(BATCH_MANAGER_LIB_LOC
          "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/${BATCH_MANAGER_TARGET_ARCH}/libtensorrt_llm_batch_manager_static.a"
      )
    else()
      set(BATCH_MANAGER_LIB_LOC
          "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/${BATCH_MANAGER_TARGET_ARCH}/libtensorrt_llm_batch_manager_static.pre_cxx11.a"
      )
    endif()
  else() # Windows
    set(BATCH_MANAGER_LIB_LOC
        "${CMAKE_CURRENT_SOURCE_DIR}/batch_manager/${BATCH_MANAGER_TARGET_ARCH}/tensorrt_llm_batch_manager_static.lib"
    )
  endif()
  set_property(TARGET ${BATCH_MANAGER_TARGET} PROPERTY IMPORTED_LOCATION
                                                       ${BATCH_MANAGER_LIB_LOC})
  file(SIZE ${BATCH_MANAGER_LIB_LOC} BATCH_MANAGER_LIB_SIZE)
  if(BATCH_MANAGER_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The batch manager library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

if(NOT MSVC)
  if(USE_CXX11_ABI)
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol"
      COMMAND nm -C $<TARGET_FILE:${BATCH_MANAGER_TARGET}> | grep -q
              'std::__cxx11::'
      DEPENDS ${BATCH_MANAGER_TARGET})
  else()
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol"
      COMMAND nm -C $<TARGET_FILE:${BATCH_MANAGER_TARGET}> | grep -qv
              'std::__cxx11::'
      DEPENDS ${BATCH_MANAGER_TARGET})
  endif()
  add_custom_target(check_symbol
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol")
else()
  add_custom_target(check_symbol)
endif()

set(TRTLLM_LINK_LIBS
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDNN_LIB}
    ${CMAKE_DL_LIBS}
    ${MPI_C_LIBRARIES}
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
  set(UNDEFINED_FLAG "-Wl,--no-undefined")
else() # MSVC
  set(UNDEFINED_FLAG "")
endif()

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS}
                                              ${UNDEFINED_FLAG})
add_dependencies(${SHARED_TARGET} check_symbol)

# ################################# STATIC LIBRARY
# ##############################################################################

add_library(${STATIC_TARGET} STATIC)

set_target_properties(
  ${STATIC_TARGET} PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
                              CXX_EXTENSIONS "NO" POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${STATIC_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})
add_dependencies(${STATIC_TARGET} check_symbol)

# Cyclic dependency of batch manager on TRT-LLM
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE ${STATIC_TARGET})

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

if(BUILD_PYBIND)
  add_subdirectory(pybind)
endif()

add_subdirectory(plugins)
