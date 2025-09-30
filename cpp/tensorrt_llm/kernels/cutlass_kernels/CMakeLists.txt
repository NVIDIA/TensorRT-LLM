#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The Python executable will only be defined if building with Torch support. If
# not, we need to find it here.
if(NOT Python3_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
endif()

execute_process(
  WORKING_DIRECTORY ${3RDPARTY_DIR}/cutlass/python/
  COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user
  RESULT_VARIABLE _CUTLASS_LIBRARY_SUCCESS)

if(NOT _CUTLASS_LIBRARY_SUCCESS EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to set up the CUTLASS library due to ${_CUTLASS_LIBRARY_SUCCESS}."
  )
endif()

set(GENERATE_KERNELS_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
set_directory_properties(
  PROPERTIES CMAKE_CONFIGURE_DEPENDS
             ${GENERATE_KERNELS_SCRIPT_DIR}/generate_kernels.py)

set(INSTANTIATION_GENERATION_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/cutlass_instantiations)

execute_process(
  WORKING_DIRECTORY ${GENERATE_KERNELS_SCRIPT_DIR}
  COMMAND ${Python3_EXECUTABLE} generate_kernels.py -a
          "${CMAKE_CUDA_ARCHITECTURES_ORIG}" -o ${INSTANTIATION_GENERATION_DIR}
  RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

if(NOT _KERNEL_GEN_SUCCESS EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to generate internal CUTLASS kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
  )
endif()

function(process_target target_name enable_hopper enable_blackwell)

  if(${enable_hopper} AND "90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    # No kernels should be parsed, unless hopper is specified. This is a build
    # time improvement
    target_compile_definitions(${target_name} PUBLIC COMPILE_HOPPER_TMA_GEMMS)
    target_compile_definitions(${target_name}
                               PUBLIC COMPILE_HOPPER_TMA_GROUPED_GEMMS)
  endif()

  if(${enable_blackwell}
     AND ("100" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
          OR "103" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
          OR "120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
          OR "121" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
         ))

    # Both 100 and 103 support these kernels
    if("100" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
       OR "103" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      # No kernels should be parsed, unless blackwell is specified. This is a
      # build time improvement
      target_compile_definitions(${target_name}
                                 PUBLIC COMPILE_BLACKWELL_TMA_GEMMS)
      target_compile_definitions(${target_name}
                                 PUBLIC COMPILE_BLACKWELL_TMA_GROUPED_GEMMS)
    endif()
    # SM103 only kernels
    if("103" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      target_compile_definitions(${target_name}
                                 PUBLIC COMPILE_BLACKWELL_SM103_TMA_GEMMS)
      target_compile_definitions(
        ${target_name} PUBLIC COMPILE_BLACKWELL_SM103_TMA_GROUPED_GEMMS)
    endif()
    if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
       OR "121" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      target_compile_definitions(${target_name}
                                 PRIVATE COMPILE_BLACKWELL_SM120_TMA_GEMMS)
      target_compile_definitions(
        ${target_name} PRIVATE COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS)
    endif()
  endif()

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()

  if(ENABLE_MULTI_DEVICE)
    target_compile_options(${target_name} PRIVATE "-DENABLE_MULTI_DEVICE=1")
    target_link_libraries(${target_name} PRIVATE ${MPI_C_LIBRARIES})
  endif()

endfunction()

function(add_instantiations library base_dir)
  macro(glob_src_create_target ARCH BUILD_ARCHS)
    file(GLOB_RECURSE INSTANTIATIONS_GENERATED_${ARCH} ${base_dir}/${ARCH}/*.cu)
    list(LENGTH INSTANTIATIONS_GENERATED_${ARCH} n)
    if(${n} GREATER 0)
      set(TARGET_NAME "_${library}_instantiations_${ARCH}")
      message(
        STATUS "Adding target ${TARGET_NAME} with instantiations for ${ARCH}")
      add_library(${TARGET_NAME} OBJECT ${INSTANTIATIONS_GENERATED_${ARCH}})
      target_link_libraries(${library} PRIVATE ${TARGET_NAME})
      set_cuda_architectures(${TARGET_NAME} ${BUILD_ARCHS})

      if(${ARCH} EQUAL 90)
        process_target(${TARGET_NAME} true false)
      elseif(${ARCH} GREATER_EQUAL 100)
        process_target(${TARGET_NAME} false true)
      endif()
    endif()
  endmacro()

  glob_src_create_target(80 "80;86;90;100f;120f") # we use sm80 kernels to
                                                  # support fp16 of all archs
  glob_src_create_target(90 90)
  glob_src_create_target(100 100f)
  glob_src_create_target(103 103)
  glob_src_create_target(120 120f)
endfunction()

# Get the sources for Mixed Input GEMM launchers
file(GLOB_RECURSE MIXED_SRC_CPP fpA_intB_gemm/*.cpp)
file(GLOB_RECURSE MIXED_SRC_CU fpA_intB_gemm/*.cu)

# Get the sources for MOE Grouped GEMM launchers
file(GLOB_RECURSE GROUPED_SRC_CPP moe_gemm/*.cpp)
file(GLOB_RECURSE GROUPED_SRC_CU moe_gemm/*.cu)

# Get the sources for FP8 Rowwise GEMM launchers
file(GLOB_RECURSE FBGEMM_SRC_CU fp8_rowwise_gemm/*.cu)

# Get the sources for FP8 Blockwise GEMM launchers
file(GLOB_RECURSE FP8_BLOCKSCALE_GEMM_SRC_CU fp8_blockscale_gemm/*.cu)

# Get the sources for FP4 GEMM launchers
file(GLOB_RECURSE FP4_SRC_CU fp4_gemm/*.cu)

# Get the sources for AllReduce GEMM
file(GLOB_RECURSE ARGEMM_SRC_CU allreduce_gemm/*.cu)

# Get the sources for all the remaining sources
file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
set(ALL_SRCS ${SRC_CPP};${SRC_CU})
list(FILTER ALL_SRCS EXCLUDE REGEX "fpA_intB_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "moe_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "fp8_rowwise_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "fp8_blockscale_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "fp4_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "allreduce_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "low_latency_gemm/.*")
list(REMOVE_ITEM ALL_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/fused_gated_gemm/gemm_swiglu_e4m3.cu")

message(
  VERBOSE
  "Mixed srcs ${MIXED_SRC_CPP} ${MIXED_SRC_CU} ${MIXED_CU_INSTANTIATIONS}")
message(
  VERBOSE
  "Group srcs ${GROUPED_SRC_CU} ${GROUPED_SRC_CPP} ${GROUPED_CU_INSTANTIATIONS}"
)
message(VERBOSE "Fbgemm srcs ${FBGEMM_SRC_CU} ${FBGEMM_CU_INSTANTIATIONS}")
message(VERBOSE "FP8 Blockscale GEMM srcs ${FP8_BLOCKSCALE_GEMM_SRC_CU}")
message(VERBOSE "FP4 srcs ${FP4_SRC_CU} ${FP4_CU_INSTANTIATIONS}")
message(VERBOSE "ARgemm srcs ${ARGEMM_SRC_CU}")
message(VERBOSE "All srcs ${ALL_SRCS}")

add_library(cutlass_src STATIC ${ALL_SRCS})

add_library(fpA_intB_gemm_src STATIC ${MIXED_SRC_CPP} ${MIXED_SRC_CU})
add_cuda_architectures(fpA_intB_gemm_src 89)
add_instantiations(fpA_intB_gemm_src ${INSTANTIATION_GENERATION_DIR}/gemm)

add_library(fb_gemm_src STATIC ${FBGEMM_SRC_CU} ${FBGEMM_CU_INSTANTIATIONS})
set_cuda_architectures(fb_gemm_src 89 90 120f)
# add_instantiations(fb_gemm_src
# ${INSTANTIATION_GENERATION_DIR}/fp8_rowwise_gemm)

add_library(fp8_blockscale_gemm_src STATIC ${FP8_BLOCKSCALE_GEMM_SRC_CU})
set_cuda_architectures(fp8_blockscale_gemm_src 89 90 100f)

set(GEMM_SWIGLU_SM90_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_gated_gemm/gemm_swiglu_e4m3.cu)
add_library(gemm_swiglu_sm90_src STATIC ${GEMM_SWIGLU_SM90_SRC_CU})
set_cuda_architectures(gemm_swiglu_sm90_src 90)

if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  file(GLOB LOW_LATENCY_GEMM_SRC_CU
       "${CMAKE_CURRENT_SOURCE_DIR}/low_latency_gemm/*.cu")
  add_library(low_latency_gemm_src STATIC ${LOW_LATENCY_GEMM_SRC_CU})
  set_cuda_architectures(low_latency_gemm_src 90)
  target_include_directories(low_latency_gemm_src
                             PRIVATE ${3RDPARTY_DIR}/cutlass/examples)
endif()

if(USING_OSS_CUTLASS_FP4_GEMM)
  file(GLOB FP4_SRC_CU "${CMAKE_CURRENT_SOURCE_DIR}/fp4_gemm/*.cu")
  add_library(fp4_gemm_src STATIC ${FP4_SRC_CU})
  set_cuda_architectures(fp4_gemm_src 100f 120f)
  # add_instantiations(fp4_gemm_src ${INSTANTIATION_GENERATION_DIR}/fp4_gemm)
endif()

if(USING_OSS_CUTLASS_MOE_GEMM)
  file(GLOB MOE_GEMM_SRC_CU "${CMAKE_CURRENT_SOURCE_DIR}/moe_gemm/*.cu")
  set(MOE_GEMM_SRC_CU_LAUNCHER ${MOE_GEMM_SRC_CU})
  list(FILTER MOE_GEMM_SRC_CU_LAUNCHER EXCLUDE REGEX ".*moe_gemm_kernels_.*")
  list(FILTER MOE_GEMM_SRC_CU INCLUDE REGEX ".*moe_gemm_kernels_.*")
  set(MOE_GEMM_SRC_CU_HOPPER_FP4 ${MOE_GEMM_SRC_CU})
  list(FILTER MOE_GEMM_SRC_CU_HOPPER_FP4 INCLUDE REGEX
       ".*moe_gemm_kernels_(bf16|fp16)_fp4.*")
  list(FILTER MOE_GEMM_SRC_CU EXCLUDE REGEX
       ".*moe_gemm_kernels_(bf16|fp16)_fp4.*")
  set(MOE_GEMM_SRC_CU_FP4 ${MOE_GEMM_SRC_CU})
  list(FILTER MOE_GEMM_SRC_CU_FP4 INCLUDE REGEX ".*fp4.*")
  list(FILTER MOE_GEMM_SRC_CU EXCLUDE REGEX ".*fp4.*")
  set(MOE_GEMM_SRC_CU_FP8 ${MOE_GEMM_SRC_CU})
  list(FILTER MOE_GEMM_SRC_CU_FP8 INCLUDE REGEX ".*fp8.*")
  list(FILTER MOE_GEMM_SRC_CU EXCLUDE REGEX ".*fp8.*")

  add_library(moe_gemm_src STATIC ${MOE_GEMM_SRC_CU} ${GROUPED_SRC_CPP})

  add_library(_moe_gemm_launcher OBJECT ${MOE_GEMM_SRC_CU_LAUNCHER})
  add_cuda_architectures(_moe_gemm_launcher 89)

  add_library(_moe_gemm_hopper_fp4 OBJECT ${MOE_GEMM_SRC_CU_HOPPER_FP4})
  set_cuda_architectures(_moe_gemm_hopper_fp4 90)
  process_target(_moe_gemm_hopper_fp4 true false)

  add_library(_moe_gemm_fp4 OBJECT ${MOE_GEMM_SRC_CU_FP4})
  set_cuda_architectures(_moe_gemm_fp4 100f 103 120f)
  process_target(_moe_gemm_fp4 false true)

  add_library(_moe_gemm_fp8 OBJECT ${MOE_GEMM_SRC_CU_FP8})
  set_cuda_architectures(_moe_gemm_fp8 89 90 100f 120f)
  process_target(_moe_gemm_fp8 true true)

  add_instantiations(moe_gemm_src ${INSTANTIATION_GENERATION_DIR}/gemm_grouped)
  target_link_libraries(
    moe_gemm_src PRIVATE _moe_gemm_launcher _moe_gemm_hopper_fp4 _moe_gemm_fp4
                         _moe_gemm_fp8)
  target_include_directories(
    moe_gemm_src
    PUBLIC ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include)
endif()

if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  add_library(
    ar_gemm_src STATIC
    ${ARGEMM_SRC_CU} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/ipcNvlsMemory.cu)
  target_include_directories(
    ar_gemm_src
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../internal_cutlass_kernels/include)
  if(ENABLE_NVSHMEM)
    target_link_libraries(ar_gemm_src PRIVATE nvshmem::nvshmem_host
                                              nvshmem::nvshmem_device)
  endif()
  set_cuda_architectures(ar_gemm_src 90 100f)
endif()

set(ALL_KERNELS_LIB)

set(TARGET_LIB
    fpA_intB_gemm_src;fb_gemm_src;fp8_blockscale_gemm_src;gemm_swiglu_sm90_src)
if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  list(APPEND TARGET_LIB low_latency_gemm_src)
endif()
foreach(target_name ${TARGET_LIB})
  process_target(${target_name} true false) # Name, Hopper, Blackwell
endforeach()

list(APPEND ALL_KERNELS_LIB ${TARGET_LIB})

set(TARGET_LIB fpA_intB_gemm_src;fb_gemm_src)
if(USING_OSS_CUTLASS_FP4_GEMM)
  list(APPEND TARGET_LIB fp4_gemm_src)
endif()
foreach(target_name ${TARGET_LIB})
  process_target(${target_name} false true) # Name, Hopper, Blackwell
endforeach()

list(APPEND ALL_KERNELS_LIB ${TARGET_LIB})

set(TARGET_LIB "")
if(USING_OSS_CUTLASS_MOE_GEMM)
  list(APPEND TARGET_LIB moe_gemm_src)
endif()
if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  list(APPEND TARGET_LIB ar_gemm_src)
endif()
foreach(target_name ${TARGET_LIB})
  process_target(${target_name} true true) # Name, Hopper, Blackwell
endforeach()

list(APPEND ALL_KERNELS_LIB ${TARGET_LIB})

add_custom_target(cutlass_kernels DEPENDS ${ALL_KERNELS_LIB})

# foreach(target_name moe_gemm_src) process_target(${target_name} true true) #
# Name, Hopper, Blackwell endforeach()
