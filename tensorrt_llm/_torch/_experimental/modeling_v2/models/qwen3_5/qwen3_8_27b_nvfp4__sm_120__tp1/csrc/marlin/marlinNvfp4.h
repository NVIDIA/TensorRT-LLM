/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host-side header for the Marlin NVFP4 W4A16 kernels.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace marlin_nvfp4
{

inline bool isMarlinNvfp4SmSupported(int sm)
{
    return sm >= 89 && sm < 100;
}

inline bool isMarlinNvfp4DenseSmSupported(int sm)
{
    return isMarlinNvfp4SmSupported(sm) || (sm >= 120 && sm < 130);
}

// launch_hint of marlinNvfp4Gemm: the stock launch configuration.
constexpr int kMarlinNvfp4LaunchHintDefault = 0;
// launch_hint of marlinNvfp4Gemm: the tuned SM120 decode launch tables. One full-K CTA per column for
// (N, K) = (5120, 6144) (bitwise equal to the split-K launch) and (16384 or 14336, 5120) at rows <= 8, per-(N, K)
// launches at rows 9-16 and, with use_fp32_reduce, 17-32 on 188-SM GPUs, and with PDL an L2 prefetch of the first
// pipeline stages' weights before the grid-dependency wait of the in_proj instances.
constexpr int kMarlinNvfp4LaunchHintDecodeTables = 1;

// enable_pdl: launch with programmatic stream serialization (the kernel waits on
// its predecessor with griddepcontrol.wait before any global memory access).
void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream, bool enable_pdl = false,
    int launch_hint = kMarlinNvfp4LaunchHintDefault);

} // namespace marlin_nvfp4

namespace marlin_nvfp4_dispatch
{

struct thread_config_t
{
    int thread_k;
    int thread_n;
    int num_threads;
};

struct exec_config_t
{
    int blocks_per_sm;
    thread_config_t tb_cfg;
};

extern thread_config_t const kSmallBatchConfigs[];
extern thread_config_t const kLargeBatchConfigs[];

extern int const kSmallBatchConfigCount;
extern int const kLargeBatchConfigCount;

int get_scales_cache_size(
    thread_config_t const& th_config, int prob_n, int prob_k, int num_bits, int group_size, int stages);

bool is_config_feasible(thread_config_t const& cfg, int prob_n, int prob_k);

} // namespace marlin_nvfp4_dispatch

TRTLLM_MV2_NAMESPACE_END
