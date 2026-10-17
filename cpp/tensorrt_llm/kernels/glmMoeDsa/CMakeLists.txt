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

add_library(
  glm_moe_dsa_kernels_src OBJECT
  crossRankTopk.cu
  crossRankTopkClusterMerge.cu
  crossRankTopkGlobalEmit.cu
  denseMlp.cu
  draftLiveRowHop.cu
  draftRoutedMoe.cu
  draftTail.cu
  indexerLayerRopeKvAppend.cu
  postAttentionHop.cu
  postAttentionHopFromPartials.cu
  postAttentionHopRouting.cu
  ropeKvAppend.cu
  sparseMlaDecode.cu)

# The GLM-MoE-DSA kernels are written for the SM100 (datacenter Blackwell)
# family (for example 32-byte global loads), so pin the target to that family
# instead of inheriting the global architecture list. On builds without any
# sm_100-family architecture the target still compiles: device code that needs
# newer instructions is guarded by __CUDA_ARCH__ and traps.
set_cuda_architectures(glm_moe_dsa_kernels_src 100f)

set_property(TARGET glm_moe_dsa_kernels_src PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)
set_property(TARGET glm_moe_dsa_kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                     ON)
