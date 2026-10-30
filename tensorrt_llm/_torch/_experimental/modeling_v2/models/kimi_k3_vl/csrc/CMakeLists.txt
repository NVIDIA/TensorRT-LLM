# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

add_library(
  kimi_k3_model_src OBJECT k3AttnResDecode.cu k3MnnvlAllreduce.cu
                           mnnvlAllGatherKernels.cu mnnvlAllreduceAttnRes.cu)
set_cuda_architectures(kimi_k3_model_src 100f)
set_property(TARGET kimi_k3_model_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kimi_k3_model_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_source_files_properties(k3AttnResDecode.cu PROPERTIES COMPILE_OPTIONS
                                                          "--use_fast_math")
