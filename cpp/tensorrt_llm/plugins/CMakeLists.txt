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

set(PLUGIN_TARGET_NAME nvinfer_plugin_tensorrt_llm)
set(PLUGIN_SHARED_TARGET ${PLUGIN_TARGET_NAME})

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map) # Linux
set(PLUGIN_EXPORT_DEF ${TARGET_DIR}/exports.def) # Windows

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 997")

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-overloaded-virtual")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    bertAttentionPlugin
    cpSplitPlugin
    fusedLayernormPlugin
    gptAttentionCommon
    gptAttentionPlugin
    identityPlugin
    gemmPlugin
    gemmSwigluPlugin
    fp8RowwiseGemmPlugin
    smoothQuantGemmPlugin
    fp4GemmPlugin
    quantizePerTokenPlugin
    quantizeTensorPlugin
    quantizeToFP4Plugin
    layernormQuantizationPlugin
    rmsnormQuantizationPlugin
    weightOnlyGroupwiseQuantMatmulPlugin
    weightOnlyQuantMatmulPlugin
    lookupPlugin
    loraPlugin
    doraPlugin
    mixtureOfExperts
    selectiveScanPlugin
    mambaConv1dPlugin
    lruPlugin
    cumsumLastDimPlugin
    topkLastDimPlugin
    lowLatencyGemmPlugin
    eaglePlugin
    lowLatencyGemmSwigluPlugin
    qserveGemmPlugin
    cudaStreamPlugin
    gemmAllReducePlugin)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
  include_directories(${PLUGIN_ITER})
  add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

if(ENABLE_MULTI_DEVICE)
  include_directories(ncclPlugin)
  add_subdirectory(ncclPlugin)
endif()
include_directories(common)
add_subdirectory(common)

# Set gencodes
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/api/tllmPlugin.cpp")

# ################################# SHARED LIBRARY
# ##############################################################################

if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif()

add_library(${PLUGIN_SHARED_TARGET} SHARED ${PLUGIN_SOURCES})
add_cuda_architectures(${PLUGIN_SHARED_TARGET} 89)

target_include_directories(
  ${PLUGIN_SHARED_TARGET}
  PUBLIC ${CUDA_INSTALL_DIR}/include
  PUBLIC
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE ${TARGET_DIR})

if(USING_OSS_CUTLASS_FP4_GEMM)
  target_compile_definitions(${PLUGIN_SHARED_TARGET}
                             PUBLIC USING_OSS_CUTLASS_FP4_GEMM)
endif()

if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  target_compile_definitions(${PLUGIN_SHARED_TARGET}
                             PUBLIC USING_OSS_CUTLASS_ALLREDUCE_GEMM)
endif()

if(USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(${PLUGIN_SHARED_TARGET}
                             PUBLIC USING_OSS_CUTLASS_MOE_GEMM)
endif()

if(ENABLE_MULTI_DEVICE)
  target_include_directories(${PLUGIN_SHARED_TARGET}
                             PUBLIC ${MPI_C_INCLUDE_DIRS})
endif()

if(CUDA_VERSION VERSION_LESS 11.0)
  target_include_directories(${PLUGIN_SHARED_TARGET} PUBLIC ${CUB_ROOT_DIR})
endif()

set_target_properties(
  ${PLUGIN_SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17"
             CXX_STANDARD_REQUIRED "YES"
             CXX_EXTENSIONS "NO"
             ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

if(WIN32)
  set_target_properties(
    ${PLUGIN_SHARED_TARGET}
    PROPERTIES LINK_FLAGS "/DEF:${PLUGIN_EXPORT_DEF} ${UNDEFINED_FLAG}")
else()
  set_target_properties(
    ${PLUGIN_SHARED_TARGET}
    PROPERTIES
      LINK_FLAGS
      "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,-rpath,'$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()

set_property(TARGET ${PLUGIN_SHARED_TARGET} PROPERTY CUDA_STANDARD 17)

target_link_libraries(
  ${PLUGIN_SHARED_TARGET}
  ${CUBLAS_LIB}
  ${CUBLASLT_LIB}
  ${TRT_LIB}
  ${CUDA_DRV_LIB}
  ${CUDA_NVML_LIB}
  ${CUDA_RT_LIB}
  ${CMAKE_DL_LIBS}
  ${SHARED_TARGET})

if(WIN32)
  target_link_libraries(${PLUGIN_SHARED_TARGET} context_attention_src)
endif()

if(ENABLE_MULTI_DEVICE)
  target_link_libraries(${PLUGIN_SHARED_TARGET} ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()
