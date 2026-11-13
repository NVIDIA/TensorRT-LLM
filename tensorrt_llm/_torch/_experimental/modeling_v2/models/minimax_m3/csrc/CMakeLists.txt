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

# minimax_m3 native code; see the modeling_v2 README.
#
# m3/: the family's copies of the MiniMax-M3 multicast all-reduce + RMSNorm,
# low-M MXFP8 GEMM, MSA KV-length update and SM107 TP2 / TP8 sparse decode
# kernels, with their fixes. m31/: kernels the MiniMax-M3.1-Flash targets share.
# msa_vllm_topk/: the MSA top-k selection headers. Each target keeps its own
# code in <target>/csrc/.
add_modeling_v2_family(minimax_m3)

# 100f runs on SM100 / SM103 / SM107; 107a adds exact SM107 code when the build
# includes SM107.
add_modeling_v2_shared(
  minimax_m3
  ARCHS 100f 107a
  SOURCES m3/*.cu m3/*.cpp m31/*.cu m31/*.cpp
  DEPS cublaslt
  REQUIRES MULTI_DEVICE)

# The sampling kernels match FlashInfer's compilation of the same math; the
# Markov-block sampler is bit for bit the Triton kernels it replaces only
# without FMA contraction.
set_source_files_properties(
  m31/chainSpecSamplingMcFiSm107.cu
  m31/toppRenormDetSm103.cu
  m31/toppRenormDetV3Sm103.cu
  m31/softmaxMcSm103.cu
  m31/softmaxPathBSm107.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/samplingFromProbsMcFiSm107.cu
  PROPERTIES COMPILE_OPTIONS "--use_fast_math")
set_source_files_properties(m31/markovSamplingSm107.cu
                            PROPERTIES COMPILE_OPTIONS "-fmad=false")

add_modeling_v2_target(
  minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__tp2
  ARCHS 100f 107a
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__tp8
  ARCHS 100f 107a
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__dep
  ARCHS 100f 107a
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__dep2ctx
  ARCHS 100f 107a
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)
# Runs the family's shared code only.
add_modeling_v2_target(minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__tp2ctx
                       ARCHS 100f 107a)

# The MiniMax-M3 targets call the shared fixed copies in m3/ (family_ops()), so
# a build that selects only them still builds the family library.
add_modeling_v2_target(
  minimax_m3 minimax_m3_nvfp4__sm_103__tp4
  ARCHS 100f
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_nvfp4__sm_103__tp8
  ARCHS 100f
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_nvfp4__sm_107__tp2
  ARCHS 100f 107a
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_nvfp4__sm_107__tp4
  ARCHS 100f 107a
  REQUIRES MULTI_DEVICE)
add_modeling_v2_target(
  minimax_m3 minimax_m3_nvfp4__sm_107__tp8
  ARCHS 100f 107a
  REQUIRES MULTI_DEVICE)
