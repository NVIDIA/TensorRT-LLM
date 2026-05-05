#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

# Library sources are listed explicitly. Profiling/benchmark/test harnesses in
# this directory (bench_*.cu, probe_*.cu, profile_*.cu, test_*.cu) are
# standalone binaries with their own main() and must NOT be compiled into the
# trtllm shared library.
set(SRC_CU mhcKernels.cu mhcFusedHcKernel.cu)
add_library(mhcKernels_src OBJECT ${SRC_CU})

set_property(TARGET mhcKernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET mhcKernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(mhcKernels_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Expose DeepGEMM helper headers (sm100_utils.cuh, utils.cuh, tma_utils.cuh,
# reduction.cuh) Used by the tcgen05.mma-based fused post-mapping + GEMM kernel
# on SM100.
target_include_directories(
  mhcKernels_src
  PRIVATE ${CMAKE_BINARY_DIR}/_deps/deepgemm-src/deep_gemm/include)
