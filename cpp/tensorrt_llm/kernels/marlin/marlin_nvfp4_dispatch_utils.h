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

// Shared dispatch utilities for Marlin NVFP4 single-expert and MoE GEMMs.
// Declarations only — definitions are in marlin_nvfp4_dispatch_utils.cpp.

#pragma once

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

// Thread configs ordered by priority.
extern thread_config_t const kSmallBatchConfigs[];
extern thread_config_t const kLargeBatchConfigs[];

extern int const kSmallBatchConfigCount;
extern int const kLargeBatchConfigCount;

// Scale cache size shared by both single-expert and MoE dispatchers.
int get_scales_cache_size(
    thread_config_t const& th_config, int prob_n, int prob_k, int num_bits, int group_size, int stages);

// Basic config validation shared by both dispatchers.
bool is_config_feasible(thread_config_t const& cfg, int prob_n, int prob_k);

} // namespace marlin_nvfp4_dispatch
