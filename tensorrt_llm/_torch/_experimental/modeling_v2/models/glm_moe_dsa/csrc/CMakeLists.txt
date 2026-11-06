#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
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

# glm_moe_dsa native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(glm_moe_dsa)

# The kernels are written for the SM100 (datacenter Blackwell) family, for
# example its 32-byte global loads.
add_modeling_v2_target(
  glm_moe_dsa glm_5_2_nvfp4__sm_103__tp8
  ARCHS 100f
  SOURCES *.cu *.cpp)

# Preserve the C1 megakernel specialization and native Rubin code generation.
set(mega_source
    "${CMAKE_CURRENT_SOURCE_DIR}/../glm_5_2_nvfp4__sm_107__low_latency/csrc/glmMegakernel.cu"
)
set_property(
  SOURCE "${mega_source}"
  APPEND
  PROPERTY COMPILE_DEFINITIONS
           GLM52_FC2_SPARSE_WIDE=1
           GLM52_OWNER_FC1_OVERLAP=1
           GLM52_ROUTER_LG=2
           GLM52_CAND_FILTER_ROLL=0
           GLM52_SEL_BATCH_MMA=1
           GLM52_LEAN=1
           GLM52_FLAGS_MODE=5
           GLM52_IO_MODE=3
           GLM52_STAMPS_MODE=1
           GLM52_SETUP_MODE=1
           GLM52_LISTS_MODE=2
           GLM52_TOPK_UNION=1
           GLM52_TOPK_RSH=1
           GLM52_STATIC_FLAGS=2752ll
           GLM52_SMEM_MAIN=200704)
set_property(
  SOURCE "${mega_source}" APPEND
  PROPERTY COMPILE_OPTIONS --maxrregcount=200 --ftz=false --prec-div=true
           --prec-sqrt=true)
if(107 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
  set_property(
    SOURCE "${mega_source}" APPEND
    PROPERTY COMPILE_OPTIONS --generate-code=arch=compute_107a,code=sm_107a)
endif()

add_modeling_v2_target(
  glm_moe_dsa glm_5_2_nvfp4__sm_107__low_latency
  ARCHS 100f
  SOURCES *.cu *.cpp
  DEPS internal_cutlass trtllm_gen_fmha_interface
  REQUIRES MULTI_DEVICE)

add_modeling_v2_target(
  glm_moe_dsa glm_5_2_nvfp4__sm_107__high_throughput
  ARCHS 100f
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)
