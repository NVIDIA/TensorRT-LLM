/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

namespace tensorrt_llm
{
namespace kernels
{

// MultiQueryToken kernels.
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin[];

// MultiQueryToken kernels.
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len;

static const struct XQAKernelMetaInfo
{
    Data_type mDataType;
    Data_type mKVDataType;
    unsigned int mHeadDim;
    unsigned int mBeamWidth;
    unsigned int mNumQHeadsOverKV;
    unsigned int mMTileSize;
    unsigned int mTokensPerPage;
    bool mPagedKVCache;
    bool mMultiQueryTokens;
    unsigned int mSM;
    unsigned long long const* mCubin;
    unsigned int mCubinSize;
    char const* mFuncName;
} sXqaKernelMetaInfo[] = {
    // SingleQueryToken kernels.
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 0, false, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 1, 8, 8, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    // MultiQueryToken kernels.
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 0, false, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 0, false, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 64, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 64, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 128, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 128, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_80,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 0, false, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 0, false, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 64, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 64, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 128, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 128, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_80,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_80_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 0, false, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 0, false, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 64, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 64, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 128, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 128, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_86,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 0, false, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 0, false, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 64, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 64, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 128, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 128, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_86,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_86_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 0, false, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 64, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 128, true, true, kSM_89,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_89_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_bf16_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_int8_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 0, false, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 64, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_64_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 16, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_16_sm_90_cubin_len, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 128, 1, 0, 32, 128, true, true, kSM_90,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin,
        xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_pagedKV_128_nqpkv_0_m_32_sm_90_cubin_len, "kernel_mha"},
    // MHA with beamWidth=4
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_80, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_86, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 4, 1, 1, 64, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 4, 1, 1, 128, true, false, kSM_89, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_FP16, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_BF16, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_INT8, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 4, 1, 1, 64, true, false, kSM_90, nullptr, 0, "kernel_mha"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, 256, 4, 1, 1, 128, true, false, kSM_90, nullptr, 0, "kernel_mha"}};

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
