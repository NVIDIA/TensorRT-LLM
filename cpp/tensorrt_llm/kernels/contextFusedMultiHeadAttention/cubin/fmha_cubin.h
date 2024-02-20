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
// clang-format off




extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_96_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_104_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin[];



extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin_len;

extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin[];

extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin[];

extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV1
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
} sMhaKernelMetaInfosV1[] = {

};

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    bool mInterleaved;
    bool mFlashAttention;
    bool mFP32Accumulation;
    int mAttentionMaskType;
    bool mAlibiSupported;
    bool mTiled;
} sMhaKernelMetaInfosV2[] = {
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_sliding_window_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_sliding_window_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm87_kernel_nl", 6144, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm87_kernel_nl", 6144, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sliding_window_causal_sm87_kernel_nl", 6144, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm87_kernel_nl", 12288, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm87_kernel_nl", 12288, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sliding_window_causal_sm87_kernel_nl", 12288, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm87_kernel_nl", 16384, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm87_kernel_nl", 16384, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm87_kernel_nl", 16384, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm87_kernel_nl", 16384, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm87_kernel_nl", 32768, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm87_kernel_nl", 32768, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm87_kernel_nl", 32768, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm87_kernel_nl", 32768, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm87_kernel_nl", 49152, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm87_kernel_nl", 49152, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm87_kernel_nl", 49152, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm87_kernel_nl", 49152, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_sliding_window_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_sliding_window_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_sm87_kernel_nl", 6144, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_causal_sm87_kernel_nl", 6144, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_sliding_window_causal_sm87_kernel_nl", 6144, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_sm87_kernel_nl", 12288, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_causal_sm87_kernel_nl", 12288, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_sliding_window_causal_sm87_kernel_nl", 12288, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_sm87_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_sm87_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_sm87_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_sm87_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_sm87_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_sm87_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sliding_window_causal_sm87_kernel_nl_tiled", 16384, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sliding_window_causal_sm87_kernel_nl_tiled", 32768, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sliding_window_causal_sm87_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sliding_window_causal_sm87_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_kernel_nl", 6144, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm87_kernel_nl", 6144, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sliding_window_causal_sm87_kernel_nl", 6144, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_kernel_nl", 12288, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm87_kernel_nl", 12288, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sliding_window_causal_sm87_kernel_nl", 12288, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sliding_window_causal_sm87_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sliding_window_causal_sm87_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_87,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm87_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sliding_window_causal_sm87_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_16_sliding_window_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_32_sliding_window_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_40_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_128_S_64_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_80_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_128_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_160_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 0, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 1, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_256_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, 2, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm80_kernel_nl", 6144, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm80_kernel_nl", 6144, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sliding_window_causal_sm80_kernel_nl", 6144, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm80_kernel_nl", 12288, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm80_kernel_nl", 12288, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sliding_window_causal_sm80_kernel_nl", 12288, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel_nl", 16384, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm80_kernel_nl", 16384, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel_nl", 16384, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm80_kernel_nl", 16384, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel_nl", 32768, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm80_kernel_nl", 32768, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel_nl", 32768, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm80_kernel_nl", 32768, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl", 49152, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel_nl", 49152, 128, 64, false, true, false, 0, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, 1, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, 2, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_16_sliding_window_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_32_sliding_window_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_40_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_128_128_S_64_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_80_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_128_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_160_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_128_S_256_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_sm80_kernel_nl", 6144, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_causal_sm80_kernel_nl", 6144, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_16_sliding_window_causal_sm80_kernel_nl", 6144, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_sm80_kernel_nl", 12288, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_causal_sm80_kernel_nl", 12288, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_64_S_32_sliding_window_causal_sm80_kernel_nl", 12288, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_sm80_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_40_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_sm80_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_64_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_sm80_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_80_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_sm80_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_32_S_128_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_sm80_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_160_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_sm80_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_BF16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_bf16_64_16_S_256_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sliding_window_causal_sm80_kernel_nl_tiled", 16384, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sliding_window_causal_sm80_kernel_nl_tiled", 32768, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sliding_window_causal_sm80_kernel_nl_tiled", 65536, 128, 128, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_80_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 0, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 1, true, true},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sliding_window_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, 2, true, true},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_kernel_nl", 6144, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm80_kernel_nl", 6144, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sliding_window_causal_sm80_kernel_nl", 6144, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_kernel_nl", 12288, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm80_kernel_nl", 12288, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sliding_window_causal_sm80_kernel_nl", 12288, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_kernel_nl", 16384, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sliding_window_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_kernel_nl", 32768, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sliding_window_causal_sm80_kernel_nl", 32768, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_kernel_nl", 49152, 128, 64, false, true, true, 0, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 1, true, false},
{ DATA_TYPE_FP16, 0, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sliding_window_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, 2, true, false},
};

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm