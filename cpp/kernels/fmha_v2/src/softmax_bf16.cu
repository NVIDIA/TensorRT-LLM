/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "softmax_impl.h"

void run_softmax_bf16(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi)
{
    run_softmax<fmha::bf16_t, float>(dst, src, mask, attention_sinks, softmax_sum_d, cu_q_seqlens_d, s_inner, s_outer,
        b, h, 0.f, 0.f, softcapping_scale_bmm1, warps_n, has_alibi);
}
