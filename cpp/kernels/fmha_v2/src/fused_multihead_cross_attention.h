/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cuda.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_utils.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace bert
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#if USE_DEMO_BERT_PARAMS

// TODO TRT plugins use a different parameter struct taken from the old XMMA fork.
//      Until all cubins in the plugin are replaced with new kernels, we need to conform to that.
#include <fused_multihead_attention_demo_bert_params.h>

#endif // USE_DEMO_BERT_PARAMS

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gmem_params
{
    // The matrix.
    void* ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t stride_in_bytes;

    // The number of heads
    int h;

    // Hidden dim per head
    int d;

    // array of length b+1 holding prefix sum of actual sequence lengths.
    int* cu_seqlens;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_mhca : Fused_multihead_attention_params_v2
{
    // Sequence length of Q
    int s_q;
    int d_padded;
    bool force_unroll;
    Gmem_params gmem_q_params;
    Gmem_params gmem_kv_params;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace bert
