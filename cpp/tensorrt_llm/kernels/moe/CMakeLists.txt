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

# MoE domain directory. The backend sub-directories own their build targets;
# everything else here (communication/, loadBalance/, routing/, utils/) compiles
# into kernels_src, exactly as it did before the move.
add_subdirectory(cuteDsl) # cute_dsl_src
add_subdirectory(trtllmGen) # trtllm_gen_fp8_block_scale_moe{,_routing}
# cutlass/: moe_gemm_src and its per-arch OBJECT libraries are defined in
# ../cutlass_kernels/CMakeLists.txt, next to the CUTLASS instantiation generator
# they depend on.

# ../CMakeLists.txt excludes all of moe/ from its glob and takes these back into
# kernels_src, so the backend directories are named here, next to the
# add_subdirectory() calls they correspond to, and nowhere else.
file(GLOB_RECURSE MOE_SRC *.cpp *.cu)
list(FILTER MOE_SRC EXCLUDE REGEX "/kernels/moe/(cutlass|trtllmGen|cuteDsl)/.*")
set(MOE_KERNELS_SRC
    ${MOE_SRC}
    PARENT_SCOPE)
