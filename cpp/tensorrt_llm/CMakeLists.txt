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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include
                    ${API_INCLUDE_DIR})

if(ENABLE_MULTI_DEVICE EQUAL 1)
  find_package(MPI REQUIRED)
  message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
  message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
  include_directories(${MPI_C_INCLUDE_DIRS})
endif()

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(runtime)
add_subdirectory(executor_worker)

set(TARGET_ARCH "unknown")

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
    set(TARGET_ARCH "x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TARGET_ARCH "aarch64-linux-gnu")
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
    set(TARGET_ARCH "x86_64-windows-msvc")
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

set(BATCH_MANAGER_TARGET tensorrt_llm_batch_manager_static)
set(BATCH_MANAGER_TARGET_ARCH ${TARGET_ARCH})

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

set(EXECUTOR_TARGET tensorrt_llm_executor_static)
set(EXECUTOR_TARGET_ARCH ${TARGET_ARCH})

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

set(NVRTC_WRAPPER_TARGET tensorrt_llm_nvrtc_wrapper)
set(NVRTC_WRAPPER_TARGET_ARCH ${TARGET_ARCH})

if(BUILD_NVRTC_WRAPPER)
  add_subdirectory(
    kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper)
else()
  add_library(${NVRTC_WRAPPER_TARGET} SHARED IMPORTED)
  if(NOT WIN32) # Linux
    set(NVRTC_WRAPPER_LIB_SOURCE_REL_LOC
        "kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/${NVRTC_WRAPPER_TARGET_ARCH}/libtensorrt_llm_nvrtc_wrapper.so"
    )
    set(NVRTC_WRAPPER_LIB_BINARY_REL_LOC
        "kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/libtensorrt_llm_nvrtc_wrapper.so"
    )
  else()
    set(NVRTC_WRAPPER_LIB_BINARY_REL_DIR
        "kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper"
    )
    set(NVRTC_WRAPPER_DLL_NAME "tensorrt_llm_nvrtc_wrapper.dll")
    set(NVRTC_WRAPPER_LIB_NAME "tensorrt_llm_nvrtc_wrapper.lib")

    set(NVRTC_WRAPPER_LIB_SOURCE_REL_LOC
        "${NVRTC_WRAPPER_LIB_BINARY_REL_DIR}/${NVRTC_WRAPPER_TARGET_ARCH}/${NVRTC_WRAPPER_DLL_NAME}"
    )
    set(NVRTC_WRAPPER_LIB_BINARY_REL_LOC
        "${NVRTC_WRAPPER_LIB_BINARY_REL_DIR}/${NVRTC_WRAPPER_DLL_NAME}")
    set(NVRTC_WRAPPER_IMPLIB_SOURCE_REL_LOC
        "${NVRTC_WRAPPER_LIB_BINARY_REL_DIR}/${NVRTC_WRAPPER_TARGET_ARCH}/${NVRTC_WRAPPER_LIB_NAME}"
    )
    set(NVRTC_WRAPPER_IMPLIB_BINARY_REL_LOC
        "${NVRTC_WRAPPER_LIB_BINARY_REL_DIR}/${NVRTC_WRAPPER_LIB_NAME}")
  endif()
  set(NVRTC_WRAPPER_LIB_LOC
      "${CMAKE_CURRENT_SOURCE_DIR}/${NVRTC_WRAPPER_LIB_SOURCE_REL_LOC}")
  # Copy the .so to build directory, which is needed in build_wheel.py.
  configure_file(${NVRTC_WRAPPER_LIB_SOURCE_REL_LOC}
                 ${NVRTC_WRAPPER_LIB_BINARY_REL_LOC} COPYONLY)
  set_property(TARGET ${NVRTC_WRAPPER_TARGET} PROPERTY IMPORTED_LOCATION
                                                       ${NVRTC_WRAPPER_LIB_LOC})
  if(WIN32)
    set(NVRTC_WRAPPER_IMPLIB_LOC
        "${CMAKE_CURRENT_SOURCE_DIR}/${NVRTC_WRAPPER_IMPLIB_SOURCE_REL_LOC}")
    configure_file(${NVRTC_WRAPPER_IMPLIB_SOURCE_REL_LOC}
                   ${NVRTC_WRAPPER_IMPLIB_BINARY_REL_LOC} COPYONLY)
    set_property(TARGET ${NVRTC_WRAPPER_TARGET}
                 PROPERTY IMPORTED_IMPLIB ${NVRTC_WRAPPER_IMPLIB_LOC})
  endif()

  file(SIZE ${NVRTC_WRAPPER_LIB_LOC} NVRTC_WRAPPER_LIB_SIZE)
  if(NVRTC_WRAPPER_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The nvrtc wrapper library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

set(TRTLLM_LINK_LIBS
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CMAKE_DL_LIBS}
    ${TRT_LIB}
    common_src
    kernels_src
    decoder_attention_src
    fpA_intB_gemm_src
    moe_gemm_src
    gemm_swiglu_sm90_src
    cutlass_src
    layers_src
    runtime_src)

if(ENABLE_MULTI_DEVICE EQUAL 1)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()

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

function(link_whole_archive TARGET LIBRARY_TO_LINK)
  if(WIN32)
    target_link_libraries(${TARGET} PUBLIC $<TARGET_FILE:${LIBRARY_TO_LINK}>)
    set_target_properties(
      ${TARGET} PROPERTIES LINK_FLAGS "/WHOLEARCHIVE:${LIBRARY_TO_LINK}")
  else()
    # Assume everything else is like gcc
    target_link_libraries(
      ${TARGET} PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:${LIBRARY_TO_LINK}>
                        "-Wl,--no-whole-archive")
  endif()
endfunction()

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

link_whole_archive(${SHARED_TARGET} ${BATCH_MANAGER_TARGET})
link_whole_archive(${SHARED_TARGET} ${EXECUTOR_TARGET})
# Cyclic dependency of batch manager on TRT-LLM
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE ${SHARED_TARGET})
# Cyclic dependency of executor on TRT-LLM
target_link_libraries(${EXECUTOR_TARGET} INTERFACE ${SHARED_TARGET})

if(NOT WIN32)
  set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS
                                                    "-Wl,-rpath='$ORIGIN'")
endif()

target_link_libraries(${SHARED_TARGET} PUBLIC ${NVRTC_WRAPPER_TARGET})

add_dependencies(${SHARED_TARGET} check_symbol)
add_dependencies(${SHARED_TARGET} check_symbol_executor)

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

if(BUILD_PYBIND)
  add_subdirectory(pybind)
endif()

add_subdirectory(plugins)
