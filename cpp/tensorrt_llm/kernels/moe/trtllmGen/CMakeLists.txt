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

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
set(ROUTING_CUSTOM_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomBlock.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomCluster.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomClusterLarge.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/routing/RoutingCustomEntry.cu)

list(REMOVE_ITEM SRC_CU ${ROUTING_CUSTOM_SRC_CU})

# Build routing as multiple translation units without NVCC's internal parallel
# split pipeline.
add_library(trtllm_gen_fp8_block_scale_moe_routing OBJECT
            ${ROUTING_CUSTOM_SRC_CU})
set_property(TARGET trtllm_gen_fp8_block_scale_moe_routing
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_gen_fp8_block_scale_moe_routing
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(trtllm_gen_fp8_block_scale_moe_routing
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=1>)

add_library(trtllm_gen_fp8_block_scale_moe OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET trtllm_gen_fp8_block_scale_moe
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_gen_fp8_block_scale_moe
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# Split-compile to parallelize ptxas for files with many kernel instantiations.
target_compile_options(trtllm_gen_fp8_block_scale_moe
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>)
