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

add_library(tllm_kernel_cutlass_headers INTERFACE)
add_library(tllm::kernel_cutlass_headers ALIAS tllm_kernel_cutlass_headers)
target_include_directories(
  tllm_kernel_cutlass_headers
  INTERFACE ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include
            ${CMAKE_BINARY_DIR}/_deps/cutlass-src/tools/util/include
            ${CMAKE_CURRENT_LIST_DIR}/../cutlass_extensions/include)
target_link_libraries(tllm_kernel_cutlass_headers INTERFACE tllm::common_cuda)

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

add_tllm_kernel_library(
  tllm_kernel_moe_communication SOURCES
  moe/communication/fusedMoeCommKernels.cu LINK_LIBRARIES
  tllm::common_environment CUDA::cuda_driver)

add_tllm_kernel_library(
  tllm_kernel_moe_load_balance SOURCES moe/loadBalance/moeLoadBalanceKernels.cu
  LINK_LIBRARIES tllm::common_environment)

if(USING_OSS_CUTLASS_MOE_GEMM)
  add_tllm_kernel_library(
    tllm_kernel_moe_lora SOURCES moe/cutlass/moe_lora_pointer_expand.cu
    moe/cutlass/moe_lora_problem_builder.cu moe/cutlass/moe_lora_slot_expand.cu)
  target_include_directories(
    tllm_kernel_moe_lora PUBLIC ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include)
endif()

add_tllm_kernel_library(
  tllm_kernel_attention_mask SOURCES trtllmGenKernels/fmha/prepareCustomMask.cu
  LINK_LIBRARIES trtllm_gen_fmha_interface)
target_include_directories(tllm_kernel_attention_mask
                           PUBLIC ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include)

add_tllm_kernel_library(
  tllm_kernel_cuda_core_gemm SOURCES weightOnlyBatchedGemv/cudaCoreGemm.cu
  LINK_LIBRARIES tllm::common_environment tllm::kernel_cutlass_headers)

add_tllm_kernel_library(
  tllm_kernel_gemm_utilities SOURCES cutlass_kernels/cutlass_heuristic.cpp
  cutlass_kernels/cutlass_preprocessors.cpp LINK_LIBRARIES
  tllm::kernel_cutlass_headers)

list(APPEND TLLM_KERNEL_COMPONENT_TARGETS gemm_swiglu_sm90_src)

add_tllm_kernel_library(
  tllm_kernel_int8_gemm
  SOURCES
  cutlass_kernels/int8_gemm/int8_gemm_bf16.cu
  cutlass_kernels/int8_gemm/int8_gemm_fp16.cu
  cutlass_kernels/int8_gemm/int8_gemm_fp32.cu
  cutlass_kernels/int8_gemm/int8_gemm_int32.cu
  LINK_LIBRARIES
  tllm::kernel_gemm_utilities)

add_tllm_kernel_library(
  tllm_kernel_smooth_quant SOURCES weightOnlyBatchedGemv/int8SQ.cu
  LINK_LIBRARIES tllm::kernel_cutlass_headers)

set_property(TARGET tllm_kernel_int8_gemm
             PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_tllm_kernel_library(
  tllm_kernel_weight_only
  SOURCES
  preQuantScaleKernel.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4GroupwiseColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4GroupwiseColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4GroupwiseColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4PerChannelColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4PerChannelColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int4PerChannelColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8GroupwiseColumnMajoInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8GroupwiseColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8GroupwiseColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8PerChannelColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8PerChannelColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherBf16Int8PerChannelColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8GroupwiseColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8GroupwiseColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8GroupwiseColumnMajorInterleavedTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorFalse.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorInterleavedForHopperTrue.cu
  weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorInterleavedTrue.cu
  LINK_LIBRARIES
  tllm::kernel_cutlass_headers
  tllm::common_environment)

list(APPEND TLLM_KERNEL_COMPONENT_TARGETS
     trtllm_gen_fp8_block_scale_moe_routing)

add_tllm_kernel_library(
  tllm_kernel_moe_block_scale SOURCES moe/trtllmGen/DevKernel.cu LINK_LIBRARIES
  tllm::kernel_cutlass_headers)

set_property(TARGET tllm_kernel_moe_block_scale
             PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_options(tllm_kernel_moe_block_scale
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>)

add_tllm_kernel_library(
  tllm_kernel_cascade_attention SOURCES
  decoderMaskedMultiheadAttention/cascadeAttentionKernel.cu LINK_LIBRARIES
  tllm::common_environment)
set_property(TARGET tllm_kernel_cascade_attention
             PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

if(WIN32)
  set(mmha_targets tllm_mmha)
  list(APPEND TLLM_KERNEL_COMPONENT_TARGETS tllm_mmha)
else()
  set(mmha_targets ${DECODER_SHARED_TARGET_0} ${DECODER_SHARED_TARGET_1})
endif()
add_tllm_kernel_library(
  tllm_kernel_attention_decode SOURCES decoderMaskedMultiheadAttention.cu
  LINK_LIBRARIES ${mmha_targets})

list(APPEND TLLM_KERNEL_COMPONENT_TARGETS trtllm_gen_fmha)

add_tllm_kernel_library(
  tllm_kernel_lora
  SOURCES
  groupGemm.cu
  splitkGroupGemm.cu
  lora/lora.cpp
  lora/dora.cpp
  lora/loraGroupGEMMParamFillRowReorderFusion.cu
  doraScaling.cu
  LINK_LIBRARIES
  tllm::common_cublas
  tllm::kernel_cutlass_headers
  tllm::runtime_buffers
  tllm::common_environment)
