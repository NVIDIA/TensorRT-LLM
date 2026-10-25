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

# glm5_next native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(glm5_next)

# GLM-5.3-Flash's step program. Its kernels use tcgen05 and tensor memory, so
# they build for the arch-specific SM100 and SM103 targets; the sparse prefill
# instantiates FlashMLA's SM100 kernels; the BF16 GEMMs run on cuBLASLt and the
# large collectives on NCCL.
add_modeling_v2_target(
  glm5_next glm_5_3_flash_nvfp4__sm_10x__tp4
  ARCHS 100a 103a
  SOURCES *.cpp kernels/*.cu
  DEPS cublaslt flashmla nccl
  REQUIRES MULTI_DEVICE)
