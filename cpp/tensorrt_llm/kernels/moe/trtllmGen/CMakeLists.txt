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

set(ROUTING_CUSTOM_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomBlock.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomCluster.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomClusterLarge.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomEntry.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomSmallToken.cu)

set(ROUTING_OTHER_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingDeepSeek.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingFromTopKIds.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingLlama4.cu)
set(ROUTING_SRC_CU ${ROUTING_CUSTOM_SRC_CU} ${ROUTING_OTHER_SRC_CU})

# Custom routing uses one translation unit per kernel family and disables NVCC's
# internal split pipeline so ptxas can process the families in parallel.
add_library(trtllm_gen_fp8_block_scale_moe_routing STATIC ${ROUTING_SRC_CU})
add_library(tllm::kernel_moe_routing ALIAS
            trtllm_gen_fp8_block_scale_moe_routing)
target_link_libraries(trtllm_gen_fp8_block_scale_moe_routing
                      PUBLIC tllm::kernel_cutlass_headers)
target_compile_features(trtllm_gen_fp8_block_scale_moe_routing
                        PUBLIC cxx_std_20)
set_property(TARGET trtllm_gen_fp8_block_scale_moe_routing
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_gen_fp8_block_scale_moe_routing
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_source_files_properties(
  ${ROUTING_CUSTOM_SRC_CU}
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--split-compile=1>")
set_source_files_properties(
  ${ROUTING_OTHER_SRC_CU}
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>")

add_library(trtllm_gen_fp8_block_scale_moe OBJECT runner.cu)
target_link_libraries(
  trtllm_gen_fp8_block_scale_moe PUBLIC tllm::kernel_moe_routing
                                        tllm::kernel_moe_block_scale)
set_property(TARGET trtllm_gen_fp8_block_scale_moe
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_gen_fp8_block_scale_moe
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# Split-compile to parallelize ptxas for files with many kernel instantiations.
target_compile_options(trtllm_gen_fp8_block_scale_moe
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>)
