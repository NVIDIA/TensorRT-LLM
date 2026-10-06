#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
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

add_library(
  llama4_min_latency_kernels OBJECT
  llama4Bf16Bf16Gemm.cu llama4Fp8Bf16Gemm.cu llama4Fp8Fp8GemmSwiGLU.cu
  llama4MinLatencyMoEOp.cu)
set_property(TARGET llama4_min_latency_kernels
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llama4_min_latency_kernels
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_cuda_architectures(llama4_min_latency_kernels 90 100f)
