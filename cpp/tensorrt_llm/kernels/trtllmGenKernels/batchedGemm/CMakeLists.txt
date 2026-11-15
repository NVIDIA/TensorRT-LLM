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

add_library(trtllm_gen_batched_gemm OBJECT)
# Keep the prebuilt Sm107a cubins (and their metadata rows) in builds that do not
# enable SM107, as fmha/CMakeLists.txt does: 107 is omitted below, and the
# globally-set EXCLUDE_SM_107 is cleared for this directory. The runner offers
# them only with TLLM_BATCHED_GEMM_SM107A=1 (KernelRunner.cpp).
set(BMM_PRUNABLE_ARCHS 100 103 100f)
remove_definitions(-DEXCLUDE_SM_107)
filter_source_cuda_architectures(SOURCE_LIST SRC_CPP ARCHS ${BMM_PRUNABLE_ARCHS})

target_sources(trtllm_gen_batched_gemm PRIVATE ${SRC_CPP} ${SRC_CU})
target_compile_definitions(trtllm_gen_batched_gemm
                           PUBLIC TLLM_GEN_EXPORT_INTERFACE TLLM_ENABLE_CUDA)

set_property(TARGET trtllm_gen_batched_gemm PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)
set_property(TARGET trtllm_gen_batched_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                     ON)

# Embed cubins via INCBIN.
include(tllm_cubin_archive)
tllm_add_cubin_archive_sources(
  trtllm_gen_batched_gemm
  "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_export/cubins"
  NAMESPACE batchedGemm tensorrt_llm ${TRTLLM_ABI_NAMESPACE} kernels
  ARCHS ${BMM_PRUNABLE_ARCHS})
