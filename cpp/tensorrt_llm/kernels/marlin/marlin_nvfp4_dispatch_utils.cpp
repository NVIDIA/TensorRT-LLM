/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "marlin_nvfp4.h"

namespace marlin_nvfp4_dispatch
{

thread_config_t const kSmallBatchConfigs[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
thread_config_t const kLargeBatchConfigs[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};

int const kSmallBatchConfigCount = sizeof(kSmallBatchConfigs) / sizeof(thread_config_t);
int const kLargeBatchConfigCount = sizeof(kLargeBatchConfigs) / sizeof(thread_config_t);

int get_scales_cache_size(
    thread_config_t const& th_config, int prob_n, int prob_k, int num_bits, int group_size, int stages)
{
    int tb_n = th_config.thread_n;
    int tb_k = th_config.thread_k;
    int tb_groups;
    if (group_size == -1)
        tb_groups = 1;
    else if (group_size == 0)
        tb_groups = (tb_k + 31) / 32;                     // div_ceil(tb_k, 32)
    else
        tb_groups = (tb_k + group_size - 1) / group_size; // div_ceil(tb_k, group_size)
    return tb_groups * tb_n * 2 * stages;
}

bool is_config_feasible(thread_config_t const& cfg, int prob_n, int prob_k)
{
    if (cfg.thread_k == -1 || cfg.thread_n == -1 || cfg.num_threads == -1)
        return false;
    if (prob_k % cfg.thread_k != 0 || prob_n % cfg.thread_n != 0)
        return false;
    // min_thread_n = 64, min_thread_k = 64 (from marlin.cuh constants)
    if (cfg.thread_n < 64 || cfg.thread_k < 64)
        return false;
    if (cfg.num_threads < 128)
        return false;
    return true;
}

} // namespace marlin_nvfp4_dispatch
