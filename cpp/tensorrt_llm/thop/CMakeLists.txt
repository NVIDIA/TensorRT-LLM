# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
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

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(th_utils STATIC thUtils.cpp)
# Shared-library consumers must explicitly link ${SHARED_TARGET} before the
# transitive static components of tllm::torch_utils to share its logger and
# driver-wrapper singletons (see cmake/modules/modeling_v2.cmake).
add_library(tllm::torch_utils ALIAS th_utils)
# Torch headers require C++20 reversed operator== synthesis.
target_compile_features(th_utils PUBLIC cxx_std_20)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(th_utils PUBLIC tllm::runtime_buffers ${TORCH_LIBRARIES}
                                      Python3::Python)

add_library(th_tensor_allocators STATIC userbuffersTensor.cpp)
add_library(tllm::torch_tensor_allocators ALIAS th_tensor_allocators)
set_property(TARGET th_tensor_allocators PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(th_tensor_allocators PUBLIC cxx_std_20)
target_link_libraries(
  th_tensor_allocators
  PUBLIC tllm::torch_utils
  PRIVATE tllm::kernel_userbuffers)

add_library(th_cublas STATIC cublasScaledMM.cpp)
add_library(tllm::torch_cublas ALIAS th_cublas)
set_property(TARGET th_cublas PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(th_cublas PUBLIC cxx_std_20)
target_link_libraries(
  th_cublas
  PUBLIC tllm::torch_utils
  PRIVATE tllm::torch_tensor_allocators tllm::common_op_utils
          tllm::common_nccl_comm)

add_library(th_quantization STATIC fp4Quantize.cpp fp8Op.cpp)
add_library(tllm::torch_quantization ALIAS th_quantization)
set_property(TARGET th_quantization PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(th_quantization PUBLIC cxx_std_20)
target_link_libraries(
  th_quantization
  PUBLIC tllm::torch_utils
  PRIVATE tllm::common_fp8 tllm::kernel_quantization
          tllm::kernel_activation_quantization)

function(add_th_common_family target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" ""
                        "SOURCES;LINK_LIBRARIES;OBJECT_LIBRARIES")
  if(ARG_UNPARSED_ARGUMENTS
     OR ARG_KEYWORDS_MISSING_VALUES
     OR NOT ARG_SOURCES)
    message(FATAL_ERROR "${target}: specify SOURCES and valid family arguments")
  endif()
  add_library(${target} OBJECT ${ARG_SOURCES})
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20
                                             CXX_STANDARD_REQUIRED ON)
  target_link_libraries(
    ${target} PRIVATE tllm::torch_utils ${ARG_LINK_LIBRARIES}
                      ${ARG_OBJECT_LIBRARIES})
  foreach(backend IN LISTS ARG_OBJECT_LIBRARIES)
    target_sources(${target} INTERFACE $<TARGET_OBJECTS:${backend}>)
  endforeach()
endfunction()

add_th_common_family(
  th_common_attention
  SOURCES
  mlaPreprocessOp.cpp
  attentionOp.cpp
  fmhaPackMaskOp.cpp
  deepseekV4BlockTableOp.cpp
  inverseRopeFp8QuantOp.cpp
  kdaDecodeOp.cpp
  attnResOp.cpp
  fusedQKNormRopeOp.cpp
  IndexerKCacheGatherOp.cpp
  IndexerKCacheScatterOp.cpp
  IndexerTopKOp.cpp
  sparseKvCacheCompactOp.cpp
  mlaRopeInplaceOp.cpp
  relativeAttentionBiasOp.cpp
  dsv3RopeOp.cpp
  convertReqIndexToGlobalOp.cpp
  nvfp4MlaKvCacheGatherOp.cpp
  pagedFp8ContextFmhaOp.cpp
  trtllmGenQKVProcessOp.cpp
  csa2SparseContextOp.cpp
  csa2SparseDecodeOp.cpp
  LINK_LIBRARIES
  tllm::common_attention
  tllm::common_op_utils
  tllm::runtime_debug
  tllm::kernel_mla
  tllm::kernel_mla_chunked_prefill
  tllm::kernel_rope
  tllm::kernel_sparse_attention
  tllm::kernel_attention_utilities
  tllm::kernel_cache_indexing
  tllm::kernel_indexer
  tllm::kernel_tensor_operations
  tllm::kernel_minimax_m3
  trtllm_gen_fmha_interface
  OBJECT_LIBRARIES
  kda_decode_src
  kimi_k3_attn_res_src
  csa2_sparse_context_src
  csa2_sparse_decode_src
  minimax_m3_attn_src
  minimax_m3_sm107_src)
add_library(tllm::torch_attention ALIAS th_common_attention)
target_include_directories(th_common_attention
                           PRIVATE ${TRTLLM_FMHA_PARAMS_GENERATED_INCLUDE_DIR})

# MiniMax-M3 registers its operators on the attention family.
add_subdirectory(minimaxM3)

add_th_common_family(
  th_common_gemm_quant
  SOURCES
  cutlassScaledMM.cpp
  cublasScaledMMOp.cpp
  cublasFp4ScaledMM.cpp
  cudaNvfp4MM.cpp
  marlinNvfp4MM.cpp
  marlinRepack.cpp
  cudaScaledMM.cpp
  fp8QuantizationOp.cpp
  fp8PerTensorScalingTrtllmGenGemm.cpp
  fp4Op.cpp
  fp4Gemm.cpp
  fp4GemmTrtllmGen.cpp
  fp8BatchedGemmTrtllmGen.cpp
  fp4QuantizeOp.cpp
  fp4BatchedQuantize.cpp
  fp4xFp8GemmTrtllmGen.cpp
  fp8BlockScalingGemm.cpp
  fp8RowwiseGemm.cpp
  fp8Quantize.cpp
  dsv3FusedAGemmOp.cpp
  deepseekV4QNormOp.cpp
  fusedAddRMSNormQuant.cpp
  fusedActivationQuant.cpp
  fusedGatedRMSNormQuant.cpp
  rmsNormFp4Quant.cpp
  groupRmsNormOp.cpp
  mxFp8Quantize.cpp
  mxfp8Gemm.cpp
  fusedCatFp4Op.cpp
  fusedCatFp8Op.cpp
  weightOnlyQuantGemm.cpp
  weightOnlyQuantOp.cpp
  finegrained_mixed_dtype_gemm_thop.cpp
  lmHeadArgmaxOp.cpp
  tinygemm2.cpp
  LINK_LIBRARIES
  tllm::torch_tensor_allocators
  tllm::torch_cublas
  tllm::torch_quantization
  tllm::common_op_utils
  tllm::common_fp8
  tllm::common_nccl_comm
  tllm::kernel_cuda_core_gemm
  tllm::kernel_cuda_core_gemm_nvfp4
  tllm::kernel_gemm_utilities
  tllm::kernel_quantization
  tllm::kernel_activation_quantization
  tllm::kernel_normalization
  tllm::kernel_lm_head_argmax
  tllm::kernel_weight_only_gemm
  tllm::kernel_fp8_block_scale_gemm
  tllm::kernel_tiny_gemm
  fb_gemm_src
  OBJECT_LIBRARIES
  trtllm_gen_gemm
  trtllm_gen_gemm_gated_act
  trtllm_gen_batched_gemm
  ws_layernorm_src
  fusedGatedRMSNormQuant_src
  groupRmsNormKernels_src
  marlin_src
  dsv3_min_latency_kernels)
add_library(tllm::torch_gemm_quant ALIAS th_common_gemm_quant)
if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  target_link_libraries(th_common_gemm_quant PRIVATE low_latency_gemm_src)
else()
  target_link_libraries(th_common_gemm_quant
                        PRIVATE ${INTERNAL_CUTLASS_KERNELS_TARGET})
endif()
if(USING_OSS_CUTLASS_FP4_GEMM)
  target_link_libraries(th_common_gemm_quant PRIVATE fp4_gemm_src)
else()
  target_link_libraries(th_common_gemm_quant
                        PRIVATE ${INTERNAL_CUTLASS_KERNELS_TARGET})
endif()

# TODO: th_common_moe does not compile with USING_OSS_CUTLASS_MOE_GEMM=OFF
# because moe/utils/moeUtilOp.cpp depends on OSS-specific MoE utility types and
# function signatures.
add_th_common_family(
  th_common_moe
  SOURCES
  fusedTopkSoftmax.cpp
  llama4MinLatency.cpp
  noAuxTcOp.cpp
  dsv3RouterGemmOp.cpp
  LINK_LIBRARIES
  tllm::torch_cublas
  tllm::common_op_utils
  tllm::common_environment
  tllm::kernel_moe
  tllm::kernel_moe_custom_routing
  tllm::kernel_moe_prepare
  tllm::kernel_moe_communication
  tllm::kernel_moe_load_balance
  tllm::kernel_quantization
  tllm::kernel_fp8_block_scale_gemm
  tllm::kernel_grouped_gemm
  tllm::runtime_moe_load_balance
  tllm::runtime_host_accessible_memory
  tllm::runtime_distributed
  pg_utils
  OBJECT_LIBRARIES
  llama4_min_latency_kernels
  dsv3_min_latency_kernels
  marlin_src
  trtllm_gen_fp8_block_scale_moe
  trtllm_gen_batched_gemm
  cute_dsl_src
  moe_dynamic_eplb_src)
add_library(tllm::torch_moe ALIAS th_common_moe)
if(USING_OSS_CUTLASS_MOE_GEMM)
  target_link_libraries(th_common_moe PRIVATE tllm::kernel_moe_lora)
endif()

# MoE torch ops live in moe/<backend|role>/; that directory adds them to
# th_common_moe with target_sources() so the list stays next to the files.
add_subdirectory(moe)

add_th_common_family(
  th_common_distributed
  SOURCES
  allgatherOp.cpp
  allreduceOp.cpp
  alltoallOp.cpp
  asyncUlyssesOp.cpp
  ulyssesPostUnscatterOp.cpp
  ulyssesPermuteScatterOp.cpp
  helixPostProcessOp.cpp
  ncclCommunicatorOp.cpp
  nativeMemoryOp.cpp
  allocateOutput.cpp
  reducescatterOp.cpp
  localityDomainAllocator.cpp
  userbuffersFinalizeOp.cpp
  userbuffersTensorOp.cpp
  virtualMemoryAllocator.cpp
  fusedGemmAllreduceOp.cpp
  dwdpFabricGatherOp.cpp
  LINK_LIBRARIES
  tllm::torch_tensor_allocators
  tllm::torch_quantization
  tllm::common_op_utils
  tllm::common_nccl
  tllm::common_nccl_comm
  tllm::common_mcast_registry
  tllm::runtime_distributed
  tllm::runtime_buffers
  tllm::runtime_nccl
  tllm::runtime_ipc_nvls
  tllm::runtime_locality
  tllm::runtime_multicast
  tllm::kernel_custom_all_reduce
  tllm::kernel_all_reduce_fusion
  tllm::kernel_communication
  tllm::kernel_moe_prepare
  tllm::kernel_ulysses
  tllm::kernel_userbuffers
  tllm::kernel_gemm_all_reduce
  pg_utils)
add_library(tllm::torch_distributed ALIAS th_common_distributed)

add_th_common_family(
  th_common_sequence
  SOURCES
  causalConv1dOp.cpp
  gdnVerifyOp.cpp
  mambaConv1dOp.cpp
  mamba2MTPSSMCacheOp.cpp
  selectiveScanOp.cpp
  LINK_LIBRARIES
  tllm::kernel_recurrent
  tllm::common_cuda
  causalConv1d_src
  selective_scan_src
  mamba2_mtp_ssm_cache_src)
# Object-library dependencies need their objects forwarded to final consumers.
target_sources(
  th_common_sequence
  INTERFACE $<TARGET_OBJECTS:causalConv1d_src>
            $<TARGET_OBJECTS:selective_scan_src>
            $<TARGET_OBJECTS:mamba2_mtp_ssm_cache_src>)
add_library(tllm::torch_sequence ALIAS th_common_sequence)

add_th_common_family(
  th_common_visual_gen
  SOURCES
  fusedAdaptiveLayerNormOp.cpp
  fusedDiTQKNormRopeOp.cpp
  fusedDiTSplitQKNormRopeOp.cpp
  fusedDiTSplitNormOp.cpp
  fusedDiTGateResidNormShiftScaleOp.cpp
  sageQuantizeOp.cpp
  blockCausalLayoutOp.cpp
  rotateRowsOp.cpp
  scatterKvSlotsOp.cpp
  denseFp8FmhaOp.cpp
  fusedDiTQKNormRopeFp8Op.cpp
  LINK_LIBRARIES
  tllm::kernel_dit
  tllm::kernel_attention_utilities
  tllm::kernel_tensor_operations
  tllm::kernel_cache_indexing
  trtllm_gen_fmha
  trtllm_gen_fmha_interface)
add_library(tllm::torch_visual_gen ALIAS th_common_visual_gen)

add_th_common_family(
  th_common_decoding
  SOURCES
  logitsBitmaskOp.cpp
  parallelDecodeKVCacheUpdateOp.cpp
  specDecOp.cpp
  fusedSamplingOp.cpp
  chainSpeculativeSamplingOp.cpp
  dynamicTreeOp.cpp
  loraOp.cpp
  inplaceSliceCopyOp.cpp
  mhcOp.cpp
  compressorOp.cpp
  LINK_LIBRARIES
  tllm::kernel_logits
  tllm::kernel_speculative_decoding
  tllm::kernel_lora
  tllm::kernel_grouped_gemm
  tllm::common_op_utils
  OBJECT_LIBRARIES
  fusedSamplingKernels_src
  mhcKernels_src
  compressorKernels_src)
add_library(tllm::torch_decoding ALIAS th_common_decoding)

add_library(
  th_common SHARED
  $<TARGET_OBJECTS:th_common_attention>
  $<TARGET_OBJECTS:th_common_gemm_quant>
  $<TARGET_OBJECTS:th_common_moe>
  $<TARGET_OBJECTS:th_common_distributed>
  $<TARGET_OBJECTS:th_common_sequence>
  $<TARGET_OBJECTS:th_common_visual_gen>
  $<TARGET_OBJECTS:th_common_decoding>)
tllm_localize_cutlass_runtime_helpers(th_common)
set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(th_common PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED
                                                           ON)
target_link_libraries(
  th_common
  PRIVATE ${TORCH_LIBRARIES}
          th_utils
          ${Python3_LIBRARIES}
          ${SHARED_TARGET}
          pg_utils
          th_tensor_allocators
          th_cublas
          th_quantization)

if(BUILD_DEEP_GEMM)
  target_compile_definitions(th_common_decoding
                             PRIVATE TRTLLM_MHC_ENABLE_FUSED_HC)
  target_compile_definitions(th_common PRIVATE TRTLLM_MHC_ENABLE_FUSED_HC)
endif()

if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  target_compile_definitions(th_common_gemm_quant
                             PRIVATE "USING_OSS_CUTLASS_LOW_LATENCY_GEMM")
  target_compile_definitions(th_common
                             PUBLIC "USING_OSS_CUTLASS_LOW_LATENCY_GEMM")
endif()

if(USING_OSS_CUTLASS_FP4_GEMM)
  target_compile_definitions(th_common_gemm_quant
                             PRIVATE USING_OSS_CUTLASS_FP4_GEMM)
  target_compile_definitions(th_common PUBLIC USING_OSS_CUTLASS_FP4_GEMM)
endif()

if(USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(th_common_moe PRIVATE USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(th_common PUBLIC USING_OSS_CUTLASS_MOE_GEMM)
endif()

if(ENABLE_CUBLASLT_FP4_GEMM)
  target_link_libraries(th_common PRIVATE ${CUBLASLT_LIB})
  target_link_libraries(th_common_gemm_quant PRIVATE ${CUBLASLT_LIB})
endif()

if(ENABLE_MULTI_DEVICE)
  target_include_directories(th_common_distributed
                             PRIVATE ${MPI_C_INCLUDE_DIRS})
  target_include_directories(th_common PUBLIC ${MPI_C_INCLUDE_DIRS})
  target_link_libraries(th_common PRIVATE ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()

if(NOT WIN32)
  set_target_properties(
    th_common PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../../nvidia/nccl/lib")
  set_target_properties(
    th_common PROPERTIES LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
else()
  target_link_libraries(th_common PRIVATE context_attention_src)
endif()
