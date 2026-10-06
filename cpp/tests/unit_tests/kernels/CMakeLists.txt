# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
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

add_tllm_gtest(
  logitsBitmaskTest
  OWNER
  kernel
  SOURCES
  logitsBitmaskTest.cpp
  LINK_LIBRARIES
  tllm::kernel_logits
  tllm::runtime_buffers
  LABELS
  "unit;kernels;logits;gpu")
add_tllm_gtest(
  cascadeAttentionKernelTest
  OWNER
  kernel
  SOURCES
  cascadeAttentionKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_cascade_attention
  LABELS
  "unit;kernels;cascade-attention;gpu")
add_tllm_gtest(
  cascadeAttentionNumericsTest
  OWNER
  kernel
  SOURCES
  cascadeAttentionNumericsTest.cu
  LINK_LIBRARIES
  tllm::kernel_cascade_attention
  tllm::kernel_attention_decode
  LABELS
  "unit;kernels;cascade-attention;gpu")

macro(remove_compile_definition TARGET_NAME DEFINITION)
  get_target_property(DEFS ${TARGET_NAME} COMPILE_DEFINITIONS)
  if(DEFS)
    list(REMOVE_ITEM DEFS ${DEFINITION})
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_DEFINITIONS
                                                    "${DEFS}")
  endif()
endmacro()

add_gtest(mixtureOfExpertsTest moe/mixtureOfExpertsTest.cu)

# If we are using oss cutlass, build an explicit internal test
if(USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(mixtureOfExpertsTest
                             PUBLIC USING_OSS_CUTLASS_MOE_GEMM)

  add_gtest(mixtureOfExpertsInternalTest moe/mixtureOfExpertsTest.cu
            TEST_PREFIX "Internal.")
  remove_compile_definition(mixtureOfExpertsInternalTest
                            USING_OSS_CUTLASS_MOE_GEMM)
endif()

add_tllm_gtest(
  ropeTest
  OWNER
  kernel
  SOURCES
  ropeTest.cu
  LINK_LIBRARIES
  tllm::kernel_unfused_attention
  tllm::runtime_buffers
  LABELS
  "unit;kernels;rope;gpu")
add_tllm_gtest(
  shiftKCacheKernelTest
  OWNER
  kernel
  SOURCES
  shiftKCacheKernelTest.cu
  LINK_LIBRARIES
  tllm::kernel_unfused_attention
  tllm::runtime_buffers
  LABELS
  "unit;kernels;rope;gpu")
add_tllm_gtest(
  smoothQuantKernelTest
  OWNER
  kernel
  SOURCES
  smoothQuant/smoothQuantKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_smooth_quant
  tllm::kernel_int8_gemm
  LABELS
  "unit;kernels;smooth-quant;gpu")
add_tllm_gtest(
  weightOnlyKernelTest
  OWNER
  kernel
  SOURCES
  weightOnly/weightOnlyKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_weight_only
  tllm::kernel_weight_only_gemm
  LABELS
  "unit;kernels;weight-only;gpu")
add_tllm_gtest(
  mlaPreprocessTest
  OWNER
  kernel
  SOURCES
  mlaPreprocessTest.cu
  LINK_LIBRARIES
  tllm::kernel_mla
  tllm::runtime_buffers
  LABELS
  "unit;kernels;mla;gpu")
add_tllm_gtest(
  sparseAttentionKernelsTest
  OWNER
  kernel
  SOURCES
  sparseAttentionKernelsTest.cpp
  LINK_LIBRARIES
  tllm::kernel_sparse_attention
  tllm::runtime_buffers
  LABELS
  "unit;kernels;sparse-attention;gpu")

add_tllm_gtest(
  cudaCoreGemmKernelTest
  OWNER
  kernel
  SOURCES
  cudaCoreGemm/cudaCoreGemmKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_cuda_core_gemm
  CUDA::cublasLt
  LABELS
  "unit;kernels;cuda-core-gemm;gpu")

add_tllm_gtest(
  mlaChunkedPrefillTest
  OWNER
  kernel
  SOURCES
  mlaChunkedPrefillTest.cu
  LINK_LIBRARIES
  tllm::kernel_mla_chunked_prefill
  tllm::runtime_buffers
  LABELS
  "unit;kernels;mla-chunked-prefill;gpu")

add_tllm_gtest(
  fusedMoeCommKernelTest
  OWNER
  kernel
  SOURCES
  moe/fusedMoeCommKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_moe_communication
  Threads::Threads
  LABELS
  "unit;kernels;moe-communication;gpu")

add_tllm_gtest(
  gemmSwigluRunnerTest
  OWNER
  kernel
  SOURCES
  fused_gated_gemm/gemmSwigluRunnerTest.cu
  LINK_LIBRARIES
  tllm::kernel_gated_gemm
  LABELS
  "unit;kernels;gated-gemm;gpu"
  NO_GTEST_MAIN)

add_executable(gemmSwigluKernelTestSm90Fp8 EXCLUDE_FROM_ALL
               fused_gated_gemm/gemmSwigluKernelTestSm90Fp8.cu)
target_link_libraries(gemmSwigluKernelTestSm90Fp8
                      PRIVATE gtest tllm::kernel_cutlass_headers)
target_compile_features(gemmSwigluKernelTestSm90Fp8 PRIVATE cxx_std_20)
add_test(NAME gemmSwigluKernelTestSm90Fp8 COMMAND gemmSwigluKernelTestSm90Fp8)
set_tests_properties(
  gemmSwigluKernelTestSm90Fp8 PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=LAZY"
                                         LABELS "unit;kernels;gated-gemm;gpu")
add_dependencies(kernel-tests gemmSwigluKernelTestSm90Fp8)
add_dependencies(google-tests gemmSwigluKernelTestSm90Fp8)

foreach(target_name gemmSwigluRunnerTest;gemmSwigluKernelTestSm90Fp8)
  set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    # No kernels should be parsed, unless hopper is specified. This is a build
    # time improvement
    target_compile_definitions(${target_name} PRIVATE COMPILE_HOPPER_TMA_GEMMS)
    target_compile_definitions(${target_name}
                               PRIVATE COMPILE_HOPPER_TMA_GROUPED_GEMMS)
  endif()

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()
endforeach()

add_tllm_gtest(
  routingKernelsTest
  OWNER
  kernel
  SOURCES
  moe/routing/routingTest.cpp
  moe/routing/routingLlama4Test.cpp
  moe/routing/routingCustomTest.cpp
  moe/routing/routingDeepSeekTest.cpp
  LINK_LIBRARIES
  tllm::kernel_moe_routing
  tllm::runtime_buffers
  Python3::Python
  LABELS
  "unit;kernels;moe-routing;gpu")

add_tllm_gtest(
  blockScaleMoeActivationTest
  OWNER
  kernel
  SOURCES
  moe/blockScaleMoeActivationTest.cu
  LINK_LIBRARIES
  tllm::kernel_moe_block_scale
  tllm::runtime_buffers
  LABELS
  "unit;kernels;moe-block-scale;gpu")
add_tllm_gtest(
  blockScaleMoeFinalizeTest
  OWNER
  kernel
  SOURCES
  moe/blockScaleMoeFinalizeTest.cu
  LINK_LIBRARIES
  tllm::kernel_moe_block_scale
  tllm::runtime_buffers
  LABELS
  "unit;kernels;moe-block-scale;gpu")

add_tllm_gtest(
  moeLoadBalanceKernelTest
  OWNER
  kernel
  SOURCES
  moe/moeLoadBalanceKernelTest.cpp
  LINK_LIBRARIES
  tllm::kernel_moe_load_balance
  Threads::Threads
  LABELS
  "unit;kernels;moe-load-balance;gpu")

if(USING_OSS_CUTLASS_MOE_GEMM)
  add_tllm_gtest(
    moeLoraPointerExpandTest
    OWNER
    kernel
    SOURCES
    moe/moeLoraPointerExpandTest.cu
    LINK_LIBRARIES
    tllm::kernel_moe_lora
    LABELS
    "unit;kernels;moe-lora;gpu")
  add_tllm_gtest(
    moeLoraProblemBuilderTest
    OWNER
    kernel
    SOURCES
    moe/moeLoraProblemBuilderTest.cu
    LINK_LIBRARIES
    tllm::kernel_moe_lora
    LABELS
    "unit;kernels;moe-lora;gpu")
  add_tllm_gtest(
    moeLoraSlotExpandTest
    OWNER
    kernel
    SOURCES
    moe/moeLoraSlotExpandTest.cu
    LINK_LIBRARIES
    tllm::kernel_moe_lora
    LABELS
    "unit;kernels;moe-lora;gpu")
endif()

add_tllm_gtest(
  sparseKvCacheTest
  OWNER
  kernel
  SOURCES
  sparseKvCacheTest.cu
  LINK_LIBRARIES
  tllm::kernel_unfused_attention
  tllm::runtime_buffers
  LABELS
  "unit;kernels;rope;sparse-kv-cache;gpu")
add_tllm_gtest(
  prepareCustomMaskTest
  OWNER
  kernel
  SOURCES
  prepareCustomMaskTest.cpp
  LINK_LIBRARIES
  tllm::kernel_attention_mask
  tllm::runtime_buffers
  LABELS
  "unit;kernels;attention-mask;gpu")
add_gtest(fmhaDsv4EpilogueFusionTest fmhaDsv4EpilogueFusionTest.cpp)

add_tllm_gtest(
  nvfp4ColdPageKernelsTest
  OWNER
  kernel
  SOURCES
  nvfp4ColdPageKernelsTest.cpp
  LINK_LIBRARIES
  tllm::kernel_nvfp4_cold_page
  tllm::kv_cache_host_memory
  LABELS
  "unit;kernels;nvfp4-cold-page;gpu")
