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

if(NOT ENABLE_MULTI_DEVICE EQUAL 0)
  add_tllm_gtest(
    allReduceKernelTest
    OWNER
    kernel
    SOURCES
    allReduce/allReduceKernelTest.cu
    LINK_LIBRARIES
    tllm::kernel_custom_all_reduce
    tllm::runtime_ipc
    tllm::runtime_buffers
    tllm::runtime_distributed
    LABELS
    "unit;kernels;all-reduce;multi-gpu;gpu")
  add_tllm_gtest(
    allReduceFusionTest
    OWNER
    kernel
    SOURCES
    allReduce/allReduceFusionTest.cu
    LINK_LIBRARIES
    tllm::kernel_all_reduce_fusion
    tllm::kernel_rms_norm
    tllm::kernel_quantization
    tllm::runtime_distributed
    ${NCCL_LIB}
    LABELS
    "unit;kernels;all-reduce;multi-gpu;gpu")
  add_tllm_gtest(
    gemmAllReduceTest
    OWNER
    kernel
    SOURCES
    allReduce/gemmAllReduceTest.cu
    LINK_LIBRARIES
    tllm::kernel_gemm_all_reduce
    tllm::kernel_cutlass_headers
    tllm::kernel_userbuffers
    tllm::runtime_ipc_nvls
    tllm::runtime_distributed
    ${NCCL_LIB}
    LABELS
    "unit;kernels;all-reduce;multi-gpu;gpu"
    NO_GTEST_MAIN)
  if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
    target_compile_definitions(gemmAllReduceTest
                               PRIVATE USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  endif()
endif()
