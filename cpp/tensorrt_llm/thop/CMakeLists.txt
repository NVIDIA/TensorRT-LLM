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
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# Declare the dependency on the main shared library explicitly so consumers
# (e.g. thUtilsTest) place it after th_utils on the link line; this was
# previously satisfied transitively via the removed TensorRT plugin target.
target_link_libraries(th_utils PUBLIC ${SHARED_TARGET} ${TORCH_LIBRARIES}
                                      ${CUBLAS_LIB} ${CURAND_LIB})

function(add_th_common_family target)
  add_library(${target} OBJECT ${ARGN})
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20
                                             CXX_STANDARD_REQUIRED ON)
  target_link_libraries(
    ${target} PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                      ${SHARED_TARGET} pg_utils)
endfunction()

add_th_common_family(
  th_common_attention
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
  minimaxM3Fp8IndexerOp.cpp
  minimaxM3SelectBlocksOp.cpp
  mlaRopeInplaceOp.cpp
  relativeAttentionBiasOp.cpp
  dsv3RopeOp.cpp
  convertReqIndexToGlobalOp.cpp
  nvfp4MlaKvCacheGatherOp.cpp
  trtllmGenQKVProcessOp.cpp)

add_th_common_family(
  th_common_gemm_quant
  cutlassScaledMM.cpp
  cublasScaledMM.cpp
  cublasFp4ScaledMM.cpp
  cudaNvfp4MM.cpp
  marlinNvfp4MM.cpp
  marlinRepack.cpp
  cudaScaledMM.cpp
  fp8Op.cpp
  fp8PerTensorScalingTrtllmGenGemm.cpp
  fp4Op.cpp
  fp4Gemm.cpp
  fp4GemmTrtllmGen.cpp
  fp8BatchedGemmTrtllmGen.cpp
  fp4Quantize.cpp
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
  tinygemm2.cpp)

# TODO: th_common_moe does not compile with USING_OSS_CUTLASS_MOE_GEMM=OFF
# because moe/utils/moeUtilOp.cpp depends on OSS-specific MoE utility types and
# function signatures.
add_th_common_family(th_common_moe fusedTopkSoftmax.cpp llama4MinLatency.cpp
                     noAuxTcOp.cpp dsv3RouterGemmOp.cpp)

# MoE torch ops live in moe/<backend|role>/; that directory adds them to
# th_common_moe with target_sources() so the list stays next to the files.
add_subdirectory(moe)

add_th_common_family(
  th_common_distributed
  allgatherOp.cpp
  allreduceOp.cpp
  alltoallOp.cpp
  asyncUlyssesOp.cpp
  ulyssesPostUnscatterOp.cpp
  ulyssesPermuteScatterOp.cpp
  helixPostProcessOp.cpp
  ncclCommunicatorOp.cpp
  allocateOutput.cpp
  reducescatterOp.cpp
  localityDomainAllocator.cpp
  userbuffersFinalizeOp.cpp
  userbuffersTensor.cpp
  virtualMemoryAllocator.cpp
  fusedGemmAllreduceOp.cpp)

add_th_common_family(th_common_sequence causalConv1dOp.cpp mambaConv1dOp.cpp
                     mamba2MTPSSMCacheOp.cpp selectiveScanOp.cpp)

add_th_common_family(
  th_common_visual_gen fusedAdaptiveLayerNormOp.cpp fusedDiTQKNormRopeOp.cpp
  fusedDiTSplitQKNormRopeOp.cpp fusedDiTSplitNormOp.cpp
  fusedDiTGateResidNormShiftScaleOp.cpp)

add_th_common_family(
  th_common_decoding
  logitsBitmaskOp.cpp
  parallelDecodeKVCacheUpdateOp.cpp
  specDecOp.cpp
  fusedSamplingOp.cpp
  dynamicTreeOp.cpp
  loraOp.cpp
  inplaceSliceCopyOp.cpp
  mhcOp.cpp
  compressorOp.cpp)

add_library(
  th_common SHARED
  $<TARGET_OBJECTS:th_common_attention>
  $<TARGET_OBJECTS:th_common_gemm_quant>
  $<TARGET_OBJECTS:th_common_moe>
  $<TARGET_OBJECTS:th_common_distributed>
  $<TARGET_OBJECTS:th_common_sequence>
  $<TARGET_OBJECTS:th_common_visual_gen>
  $<TARGET_OBJECTS:th_common_decoding>)
set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(
  th_common PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                    ${SHARED_TARGET} pg_utils)

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
  target_compile_definitions(th_common_gemm_quant
                             PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(th_common PUBLIC ENABLE_CUBLASLT_FP4_GEMM)
  target_link_libraries(th_common PRIVATE ${CUBLASLT_LIB})
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
