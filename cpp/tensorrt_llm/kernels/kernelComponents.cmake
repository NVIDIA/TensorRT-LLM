#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

function(add_tllm_kernel_library target)
  set(multi_value_args SOURCES LINK_LIBRARIES INCLUDE_DIRECTORIES)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "${multi_value_args}")
  if(ARG_UNPARSED_ARGUMENTS
     OR ARG_KEYWORDS_MISSING_VALUES
     OR NOT ARG_SOURCES)
    message(FATAL_ERROR "${target}: explicit kernel sources are required")
  endif()

  set(sources)
  foreach(source IN LISTS ARG_SOURCES)
    list(APPEND sources "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/${source}")
  endforeach()
  add_library(${target} STATIC ${sources})
  string(REGEX REPLACE "^tllm_" "tllm::" alias ${target})
  add_library(${alias} ALIAS ${target})
  target_link_libraries(${target} PUBLIC tllm::common_cuda
                                         ${ARG_LINK_LIBRARIES})
  target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(
    ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(
    ${target}
    PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_cuda_architectures(${target} 89)

  foreach(source_list SRC_CPP SRC_CU MOE_KERNELS_SRC)
    list(REMOVE_ITEM ${source_list} ${sources})
    set(${source_list}
        ${${source_list}}
        PARENT_SCOPE)
  endforeach()
  set(TLLM_KERNEL_COMPONENT_TARGETS
      ${TLLM_KERNEL_COMPONENT_TARGETS} ${target}
      PARENT_SCOPE)
endfunction()

add_tllm_kernel_library(tllm_kernel_logits SOURCES logitsBitmask.cu)

add_tllm_kernel_library(
  tllm_kernel_nvfp4_cold_page SOURCES nvfp4ColdPageKernels.cu LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_rope
  SOURCES
  attentionMask.cu
  gptKernels.cu
  unfusedAttentionKernels.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_bf16_bf16.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_bf16_fp4.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_bf16_fp8.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_bf16_int8.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_float_float.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_float_fp8.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_float_int8.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_half_fp4.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_half_fp8.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_half_half.cu
  unfusedAttentionKernels/unfusedAttentionKernels_2_half_int8.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(tllm_kernel_sparse_attention SOURCES
                        sparseAttentionKernels.cu)

add_tllm_kernel_library(tllm_kernel_mla SOURCES mlaKernels.cu LINK_LIBRARIES
                        tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_mla_chunked_prefill SOURCES mlaChunkedPrefill.cu
  INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include)
