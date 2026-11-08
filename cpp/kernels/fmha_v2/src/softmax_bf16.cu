/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "softmax_impl.h"

void run_softmax_bf16(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi)
{
    run_softmax<fmha::bf16_t, float>(dst, src, mask, attention_sinks, softmax_sum_d, cu_q_seqlens_d, s_inner, s_outer,
        b, h, 0.f, 0.f, softcapping_scale_bmm1, warps_n, has_alibi);
}
