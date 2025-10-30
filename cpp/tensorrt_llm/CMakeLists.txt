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

if(ENABLE_MULTI_DEVICE)
  find_package(MPI REQUIRED)
  message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
  message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
  include_directories(${MPI_C_INCLUDE_DIRS})
endif()

if(ENABLE_NVSHMEM)
  # Add hints for aarch64
  find_package(NVSHMEM REQUIRED HINTS /usr/lib/sbsa-linux-gnu/cmake/nvshmem/)
  include_directories(/usr/include/nvshmem/)
endif()

if(NOT WIN32)
  set(DECODER_SHARED_TARGET_0 decoder_attention_0)
  set(DECODER_SHARED_TARGET_1 decoder_attention_1)
endif()

# Build internal cutlass kernels as subproject
if(INTERNAL_CUTLASS_KERNELS_PATH)
  set(BUILD_INTERNAL_CUTLASS_KERNELS ON)
  set(BUILD_NVRTC_WRAPPER ON)
  if(NOT EXISTS ${INTERNAL_CUTLASS_KERNELS_PATH}/CMakeLists.txt
     AND EXISTS ${INTERNAL_CUTLASS_KERNELS_PATH}/cpp/CMakeLists.txt)
    set(INTERNAL_CUTLASS_KERNELS_PATH ${INTERNAL_CUTLASS_KERNELS_PATH}/cpp)
  endif()

  add_subdirectory(${INTERNAL_CUTLASS_KERNELS_PATH}
                   ${PROJECT_BINARY_DIR}/internal_cutlass_kernels)
endif()

# Import internal cutlass kernels
set(INTERNAL_CUTLASS_KERNELS_TARGET
    tensorrt_llm_internal_cutlass_kernels_static)
set(INTERNAL_CUTLASS_KERNELS_TARGET_ARCH ${TARGET_ARCH})
if(NOT INTERNAL_CUTLASS_KERNELS_PATH)
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
  target_include_directories(
    ${INTERNAL_CUTLASS_KERNELS_TARGET}
    INTERFACE
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/internal_cutlass_kernels/include")
  file(SIZE ${INTERNAL_CUTLASS_KERNELS_LIB_TARBALL}
       INTERNAL_CUTLASS_KERNELS_LIB_SIZE)
  if(INTERNAL_CUTLASS_KERNELS_LIB_SIZE LESS 1024)
    message(
      FATAL_ERROR
        "The internal_cutlass_kernels library is truncated or incomplete. This is usually caused by using Git LFS (Large File Storage) incorrectly. Please try running command `git lfs install && git lfs pull`."
    )
  endif()
endif()

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(runtime)
add_subdirectory(testing)
add_subdirectory(executor_worker)

set(BATCH_MANAGER_TARGET tensorrt_llm_batch_manager_static)
set(BATCH_MANAGER_TARGET_ARCH ${TARGET_ARCH})
add_subdirectory(batch_manager)

set(EXECUTOR_TARGET tensorrt_llm_executor_static)
set(EXECUTOR_TARGET_ARCH ${TARGET_ARCH})
set(UCX_WRAPPER_TARGET tensorrt_llm_ucx_wrapper)

if(NIXL_ROOT)
  set(NIXL_WRAPPER_TARGET tensorrt_llm_nixl_wrapper)
endif()

add_subdirectory(executor)

find_package(Threads REQUIRED)
target_link_libraries(${BATCH_MANAGER_TARGET} INTERFACE Threads::Threads)
target_link_libraries(${EXECUTOR_TARGET} INTERFACE Threads::Threads)

set(TRTLLM_LINK_LIBS
    ${CUDA_DRV_LIB}
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CURAND_LIB}
    ${CMAKE_DL_LIBS}
    ${TRT_LIB}
    common_src
    kernels_src
    flash_mla_src
    context_attention_src
    decoder_attention_src
    trtllm_gen_fmha
    trtllm_gen_fp8_block_scale_moe
    trtllm_gen_gemm
    trtllm_gen_gemm_gated_act
    trtllm_gen_batched_gemm
    selective_scan_src
    ws_layernorm_src
    fpA_intB_gemm_src
    # moe_gemm_src
    fb_gemm_src
    gemm_swiglu_sm90_src
    cutlass_src
    layers_src
    runtime_src
    testing_src
    userbuffers_src
    ${DECODER_SHARED_TARGET_0}
    ${DECODER_SHARED_TARGET_1})
if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} low_latency_gemm_src)
  message(STATUS "USING_OSS_CUTLASS_LOW_LATENCY_GEMM")
endif()

if(USING_OSS_CUTLASS_FP4_GEMM)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} fp4_gemm_src)
  message(STATUS "USING_OSS_CUTLASS_FP4_GEMM")
endif()

if(USING_OSS_CUTLASS_MOE_GEMM)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} moe_gemm_src)
  message(STATUS "USING_OSS_CUTLASS_MOE_GEMM")
endif()

if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} ar_gemm_src)
  message(STATUS "USING_OSS_CUTLASS_ALLREDUCE_GEMM")
endif()

if(ENABLE_MULTI_DEVICE)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()

if(ENABLE_NVSHMEM)
  set(TRTLLM_LINK_LIBS ${TRTLLM_LINK_LIBS} nvshmem::nvshmem_host
                       nvshmem::nvshmem_device)
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

# Cyclic dependency of UCX data transceiver on TRT-LLM
if(TARGET ${UCX_WRAPPER_TARGET})
  target_link_libraries(${UCX_WRAPPER_TARGET} INTERFACE ${SHARED_TARGET})
  add_dependencies(${SHARED_TARGET} ${UCX_WRAPPER_TARGET})
endif()

if(TARGET ${NIXL_WRAPPER_TARGET})
  target_link_libraries(${NIXL_WRAPPER_TARGET} INTERFACE ${SHARED_TARGET})
  add_dependencies(${SHARED_TARGET} ${NIXL_WRAPPER_TARGET})
endif()

if(NOT WIN32)
  # Load libraries at $PREFIX/lib from
  # $PREFIX/lib/python3.12/site-packages/tensorrt_llm/libs
  set_target_properties(${SHARED_TARGET}
                        PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../../../..")
endif()

if(BUILD_PYT)
  add_subdirectory(thop)
endif()

if(BINDING_TYPE STREQUAL "pybind")
  add_subdirectory(pybind)
endif()

if(BINDING_TYPE STREQUAL "nanobind")
  add_subdirectory(nanobind)
endif()

if(BUILD_DEEP_EP)
  add_subdirectory(deep_ep)
endif()

if(BUILD_DEEP_GEMM)
  add_subdirectory(deep_gemm)
endif()

if(BUILD_FLASH_MLA)
  add_subdirectory(flash_mla)
endif()

add_subdirectory(plugins)
