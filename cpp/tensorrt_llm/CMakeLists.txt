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

set(EXECUTOR_TARGET tensorrt_llm_executor_static)
set(EXECUTOR_TARGET_ARCH "unknown")

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
    set(EXECUTOR_TARGET_ARCH "x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(BATCH_MANAGER_TARGET_ARCH "aarch64-linux-gnu")
    set(EXECUTOR_TARGET_ARCH "aarch64-linux-gnu")
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
    set(EXECUTOR_TARGET_ARCH "x86_64-windows-msvc")
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

if(BUILD_EXECUTOR)
  add_subdirectory(executor)
else()
  add_library(${EXECUTOR_TARGET} STATIC IMPORTED)
  if(NOT WIN32) # Linux
    if(USE_CXX11_ABI)
      set(EXECUTOR_LIB_LOC
          "${CMAKE_CURRENT_SOURCE_DIR}/executor/${EXECUTOR_TARGET_ARCH}/libtensorrt_llm_executor_static.a"
      )
    else()
      set(EXECUTOR_LIB_LOC
          "${CMAKE_CURRENT_SOURCE_DIR}/executor/${EXECUTOR_TARGET_ARCH}/libtensorrt_llm_executor_static.pre_cxx11.a"
      )
    endif()
  else() # Windows
    set(EXECUTOR_LIB_LOC
        "${CMAKE_CURRENT_SOURCE_DIR}/executor/${EXECUTOR_TARGET_ARCH}/tensorrt_llm_executor_static.lib"
    )
  endif()
  set_property(TARGET ${EXECUTOR_TARGET} PROPERTY IMPORTED_LOCATION
                                                  ${EXECUTOR_LIB_LOC})
  file(SIZE ${EXECUTOR_LIB_LOC} EXECUTOR_LIB_SIZE)
  if(EXECUTOR_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The executor library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE Threads::Threads)
target_link_libraries(${EXECUTOR_TARGET} INTERFACE Threads::Threads)

if(NOT WIN32)
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

if(NOT WIN32)
  if(USE_CXX11_ABI)
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol_executor"
      COMMAND nm -C $<TARGET_FILE:${EXECUTOR_TARGET}> | grep -q 'std::__cxx11::'
      DEPENDS ${EXECUTOR_TARGET})
  else()
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol_executor"
      COMMAND nm -C $<TARGET_FILE:${EXECUTOR_TARGET}> | grep -qv
              'std::__cxx11::'
      DEPENDS ${EXECUTOR_TARGET})
  endif()
  add_custom_target(
    check_symbol_executor
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol_executor")
else()
  add_custom_target(check_symbol_executor)
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
    cutlass_src_pre_hopper
    cutlass_src_hopper
    layers_src
    runtime_src)

if(NOT WIN32) # Unix-like compilers
  set(UNDEFINED_FLAG "-Wl,--no-undefined")
  set(AS_NEEDED_FLAG "-Wl,--as-needed")
else() # Windows
  set(UNDEFINED_FLAG "")
  set(AS_NEEDED_FLAG "")
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(${SHARED_TARGET} SHARED)

set_target_properties(
  ${SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

if(WIN32)
  target_link_libraries(${SHARED_TARGET}
                        PUBLIC $<TARGET_FILE:${BATCH_MANAGER_TARGET}>)
  set_target_properties(
    ${SHARED_TARGET} PROPERTIES LINK_FLAGS
                                "/WHOLEARCHIVE:${BATCH_MANAGER_TARGET}")
else()
  # Assume everything else is like gcc
  target_link_libraries(
    ${SHARED_TARGET}
    PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:${BATCH_MANAGER_TARGET}>
            "-Wl,--no-whole-archive")
endif()

if(WIN32)
  target_link_libraries(${SHARED_TARGET}
                        PUBLIC $<TARGET_FILE:${EXECUTOR_TARGET}>)
  set_target_properties(
    ${SHARED_TARGET} PROPERTIES LINK_FLAGS "/WHOLEARCHIVE:${EXECUTOR_TARGET}")
else()
  # Assume everything else is like gcc
  target_link_libraries(
    ${SHARED_TARGET}
    PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:${EXECUTOR_TARGET}>
            "-Wl,--no-whole-archive")
endif()

add_dependencies(${SHARED_TARGET} check_symbol)
add_dependencies(${SHARED_TARGET} check_symbol_executor)

# Cyclic dependency of batch manager on TRT-LLM
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE ${SHARED_TARGET})

# Cyclic dependency of executor on TRT-LLM
target_link_libraries(${EXECUTOR_TARGET} INTERFACE ${SHARED_TARGET})

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

if(BUILD_PYBIND)
  add_subdirectory(pybind)
endif()

add_subdirectory(plugins)
