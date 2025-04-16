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

if(ENABLE_MULTI_DEVICE)
  find_package(MPI REQUIRED)
  message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
  message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
  include_directories(${MPI_C_INCLUDE_DIRS})
endif()

if(NOT WIN32)
  set(DECODER_SHARED_TARGET_0 decoder_attention_0)
  set(DECODER_SHARED_TARGET_1 decoder_attention_1)
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
add_subdirectory(batch_manager)

set(EXECUTOR_TARGET tensorrt_llm_executor_static)
set(EXECUTOR_TARGET_ARCH ${TARGET_ARCH})
set(UCX_WRAPPER_TARGET tensorrt_llm_ucx_wrapper)
add_subdirectory(executor)

set(INTERNAL_CUTLASS_KERNELS_TARGET
    tensorrt_llm_internal_cutlass_kernels_static)
set(INTERNAL_CUTLASS_KERNELS_TARGET_ARCH ${TARGET_ARCH})
if(BUILD_INTERNAL_CUTLASS_KERNELS)
  add_subdirectory(kernels/internal_cutlass_kernels)
else()
  add_library(${INTERNAL_CUTLASS_KERNELS_TARGET} STATIC IMPORTED)
  set(INTERNAL_CUTLASS_KERNELS_LIB_TARBALL
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/internal_cutlass_kernels/${INTERNAL_CUTLASS_KERNELS_TARGET_ARCH}/${INTERNAL_CUTLASS_KERNELS_TARGET}.tar.xz"
  )
  if(NOT WIN32) # Linux
    set(INTERNAL_CUTLASS_KERNELS_LIB_NAME
        "lib${INTERNAL_CUTLASS_KERNELS_TARGET}.a")
  else() # Windows
    set(INTERNAL_CUTLASS_KERNELS_LIB_NAME
        "${INTERNAL_CUTLASS_KERNELS_TARGET}.lib")
  endif()
  set(INTERNAL_CUTLASS_KERNELS_LIB_PATH
      "${CMAKE_CURRENT_BINARY_DIR}/${INTERNAL_CUTLASS_KERNELS_LIB_NAME}")
  add_custom_command(
    OUTPUT ${INTERNAL_CUTLASS_KERNELS_LIB_PATH}
    COMMAND ${CMAKE_COMMAND} -E tar xf ${INTERNAL_CUTLASS_KERNELS_LIB_TARBALL}
    DEPENDS ${INTERNAL_CUTLASS_KERNELS_LIB_TARBALL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)
  add_custom_target(${INTERNAL_CUTLASS_KERNELS_TARGET}_helper
                    DEPENDS ${INTERNAL_CUTLASS_KERNELS_LIB_PATH})
  add_dependencies(${INTERNAL_CUTLASS_KERNELS_TARGET}
                   ${INTERNAL_CUTLASS_KERNELS_TARGET}_helper)
  set_property(TARGET ${INTERNAL_CUTLASS_KERNELS_TARGET}
               PROPERTY IMPORTED_LOCATION ${INTERNAL_CUTLASS_KERNELS_LIB_PATH})
  target_link_libraries(${INTERNAL_CUTLASS_KERNELS_TARGET}
                        INTERFACE ${INTERNAL_CUTLASS_KERNELS_LIB_PATH})
  file(SIZE ${INTERNAL_CUTLASS_KERNELS_LIB_TARBALL}
       INTERNAL_CUTLASS_KERNELS_LIB_SIZE)
  if(INTERNAL_CUTLASS_KERNELS_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The internal_cutlass_kernels library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE Threads::Threads)
target_link_libraries(${EXECUTOR_TARGET} INTERFACE Threads::Threads)

set(NVRTC_WRAPPER_TARGET tensorrt_llm_nvrtc_wrapper)
set(NVRTC_WRAPPER_TARGET_ARCH ${TARGET_ARCH})

if(BUILD_NVRTC_WRAPPER)
  add_subdirectory(
    kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper)
else()
  add_library(${NVRTC_WRAPPER_TARGET} SHARED IMPORTED)
  set(NVRTC_WRAPPER_LIB_TARBALL
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/${NVRTC_WRAPPER_TARGET_ARCH}/${NVRTC_WRAPPER_TARGET}.tar.xz"
  )
  set(NVRTC_WRAPPER_LIB_BINARY_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper"
  )
  if(NOT WIN32) # Linux
    set(NVRTC_WRAPPER_LIB_NAME "lib${NVRTC_WRAPPER_TARGET}.so")
  else() # Windows
    set(NVRTC_WRAPPER_LIB_NAME "${NVRTC_WRAPPER_TARGET}.lib")
    set(NVRTC_WRAPPER_DLL_NAME "${NVRTC_WRAPPER_TARGET}.dll")
    set(NVRTC_WRAPPER_DLL_PATH
        "${NVRTC_WRAPPER_LIB_BINARY_DIR}/${NVRTC_WRAPPER_DLL_NAME}")
  endif()
  set(NVRTC_WRAPPER_LIB_PATH
      "${NVRTC_WRAPPER_LIB_BINARY_DIR}/${NVRTC_WRAPPER_LIB_NAME}")
  add_custom_command(
    OUTPUT ${NVRTC_WRAPPER_LIB_PATH} ${NVRTC_WRAPPER_DLL_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NVRTC_WRAPPER_LIB_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E chdir ${NVRTC_WRAPPER_LIB_BINARY_DIR}
            ${CMAKE_COMMAND} -E tar xf ${NVRTC_WRAPPER_LIB_TARBALL}
    DEPENDS ${NVRTC_WRAPPER_LIB_TARBALL}
    VERBATIM)
  add_custom_target(${NVRTC_WRAPPER_TARGET}_helper
                    DEPENDS ${NVRTC_WRAPPER_LIB_PATH} ${NVRTC_WRAPPER_DLL_PATH})
  add_dependencies(${NVRTC_WRAPPER_TARGET} ${NVRTC_WRAPPER_TARGET}_helper)
  set_property(TARGET ${NVRTC_WRAPPER_TARGET}
               PROPERTY IMPORTED_LOCATION ${NVRTC_WRAPPER_LIB_PATH})
  if(WIN32)
    set_property(TARGET ${NVRTC_WRAPPER_TARGET}
                 PROPERTY IMPORTED_IMPLIB ${NVRTC_WRAPPER_DLL_PATH})
  endif()

  file(SIZE ${INTERNAL_CUTLASS_KERNELS_LIB_TARBALL} NVRTC_WRAPPER_LIB_SIZE)
  if(NVRTC_WRAPPER_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The nvrtc wrapper library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

set(TRTLLM_LINK_LIBS
    ${CUDA_DRV_LIB}
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CMAKE_DL_LIBS}
    ${TRT_LIB}
    common_src
    kernels_src
    flash_mla_src
    context_attention_src
    decoder_attention_src
    trtllm_gen_fmha
    trtllm_gen_blockscale_gemm
    trtllm_gen_fp8_block_scale_moe
    trtllm_gen_gemm
    selective_scan_src
    ws_layernorm_src
    fpA_intB_gemm_src
    # moe_gemm_src
    fb_gemm_src
    fp8_blockscale_gemm_src
    ar_gemm_src
    gemm_swiglu_sm90_src
    cutlass_src
    layers_src
    runtime_src
    userbuffers_src
    ${DECODER_SHARED_TARGET_0}
    ${DECODER_SHARED_TARGET_1})

if(ENABLE_MULTI_DEVICE)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()

if(NOT WIN32) # Unix-like compilers
  set(UNDEFINED_FLAG "-Wl,--no-undefined")
  set(AS_NEEDED_FLAG "-Wl,--as-needed")
  set(NO_AS_NEEDED_FLAG "-Wl,--no-as-needed")
else() # Windows
  set(UNDEFINED_FLAG "")
  set(AS_NEEDED_FLAG "")
  set(NO_AS_NEEDED_FLAG "")
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(${SHARED_TARGET} SHARED)

set_target_properties(
  ${SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

target_link_libraries(
  ${SHARED_TARGET}
  PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${BATCH_MANAGER_TARGET}>
          $<LINK_LIBRARY:WHOLE_ARCHIVE,${EXECUTOR_TARGET}>
          $<LINK_LIBRARY:WHOLE_ARCHIVE,fp8_blockscale_gemm_src>
          $<LINK_LIBRARY:WHOLE_ARCHIVE,${INTERNAL_CUTLASS_KERNELS_TARGET}>)

# Link kernel_src and cutlass_src. static internal cutlass lib overridden.
target_link_libraries(${SHARED_TARGET} PUBLIC kernels_src cutlass_src)

# Cyclic dependency of batch manager on TRT-LLM
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE ${SHARED_TARGET})
# Cyclic dependency of executor on TRT-LLM
target_link_libraries(${EXECUTOR_TARGET} INTERFACE ${SHARED_TARGET})
# Cyclic dependency of internal_cutlass_kernels on TRT-LLM
target_link_libraries(${INTERNAL_CUTLASS_KERNELS_TARGET}
                      INTERFACE ${SHARED_TARGET})

# Cyclic dependency of UCX data transceiver on TRT-LLM
if(TARGET ${UCX_WRAPPER_TARGET})
  target_link_libraries(${UCX_WRAPPER_TARGET} INTERFACE ${SHARED_TARGET})
  add_dependencies(${SHARED_TARGET} ${UCX_WRAPPER_TARGET})
endif()

if(NOT WIN32)
  set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS
                                                    "-Wl,-rpath='$ORIGIN'")
endif()

target_link_libraries(${SHARED_TARGET} PUBLIC ${NVRTC_WRAPPER_TARGET})

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

if(BUILD_PYBIND)
  add_subdirectory(pybind)
endif()

add_subdirectory(plugins)
