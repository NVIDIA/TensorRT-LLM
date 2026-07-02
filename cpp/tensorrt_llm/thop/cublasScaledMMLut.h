/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace cublas_lut
{

struct HashTuple
{
    size_t operator()(std::tuple<int32_t, int32_t, int32_t> const& x) const
    {
        return std::get<0>(x) ^ std::get<1>(x) ^ std::get<2>(x);
    }
};

// {mp2, k, n}: {algo, m_tile, m_stages, m_numsK, m_reduction, m_swizzle, m_custom, m_cga}
using AlgoListType = std::unordered_map<std::tuple<int32_t, int32_t, int32_t>, std::array<int, 8>, HashTuple>;

inline const AlgoListType spark_bf16_algo_list = {
    // llama 8b instruct fp16 decode
    // [-algo67 -m_tile6 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom130 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 4096}, {67, 6, 35, 1, 0, 0, 130, 2}},
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 6144}, {67, 393, 35, 1, 0, 0, 142, 2}},
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 4096, 128256}, {67, 393, 35, 1, 0, 0, 142, 2}},

    // gpt-oss mxfp4-fp16 decode
    // [-algo67 -m_tile393 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 2880, 201088}, {67, 393, 35, 1, 0, 0, 142, 2}},
    // [-algo14 -m_tile0 -m_stages35 -m_numsK10 -m_reduction2 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 2880, 32}, {14, 0, 0, 10, 2, 0, 0, 0}},
    // [-algo21 -m_tile11 -m_stages13 -m_numsK9 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-k2880
    {{2048, 2880, 32}, {21, 11, 13, 9, 1, 0, 0, 0}},
    // [-algo21 -m_tile11 -m_stages19 -m_numsK11 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-m_workmem1024 -k2880
    {{4096, 2880, 32}, {21, 11, 19, 11, 1, 0, 0, 0}},
    // [-algo23 -m_tile11 -m_stages8 -m_numsK2 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    //-m_workmem1024 -k2880
    {{8, 2880, 5120}, {23, 11, 8, 2, 1, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{2048, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{4096, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo23 -m_tile11 -m_stages14 -m_numsK24 -m_reduction1 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 2880}, {23, 11, 14, 24, 1, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{2048, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},
    // [-algo21 -m_tile20 -m_stages15 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{4096, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},

    // nemotron super v3
    // [-algo67 -m_tile6 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom16 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 1024, 4096}, {67, 6, 37, 1, 0, 0, 16, 0}},
    // [-algo67 -m_tile6 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom93 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 512}, {67, 6, 37, 1, 0, 0, 93, 0}},
    // [-algo67 -m_tile10 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom142 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 1024}, {67, 10, 37, 1, 0, 0, 142, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 4096}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 4608}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 5376}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom75 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 18560}, {13, 0, 0, 1, 0, 0, 75, 0}},
    // [-algo13 -m_tile0 -m_stages0 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom83 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 131072}, {13, 0, 0, 1, 0, 0, 83, 0}},
    // [-algo67 -m_tile3 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom76 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 5376, 4096}, {67, 3, 37, 1, 0, 0, 76, 0}},
    // [-algo67 -m_tile3 -m_stages38 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom0 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 8192, 4096}, {67, 3, 38, 1, 0, 0, 0, 0}},
};

// bf16*bf16->fp32->bf16
inline const AlgoListType bf16_algo_list = {
    // Deepseek v3/R1 router gemm
    // [-algo66 -m_tile10 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom3 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 7168, 256}, {66, 10, 35, 1, 0, 0, 3, 2}},
    {{512, 7168, 256}, {66, 48, 35, 1, 0, 0, 0, 2}},
    {{1024, 7168, 256}, {66, 13, 35, 1, 0, 0, 1, 3}},
};

// fp8*fp8->fp32->fp16
inline const AlgoListType fp8_algo_list = {
    // Llama-3.1-70B
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom5 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 8192}, {66, 393, 36, 1, 0, 0, 5, 2}},
    // [-algo66 -m_tile10 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 57344}, {66, 10, 36, 1, 0, 0, 1, 2}},
    // Llama-3.3-70B TP4 (this is the default algo on B200. Here we aim to use the same algo on GB200.)
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga4 -m_scheduling1]
    {{8, 8192, 14336}, {66, 393, 36, 1, 0, 1, 1, 4}},

    // Qwen3.6-35B-A3B-NVFP4 on GB10 (sm_121), FP8 per-tensor projections. Output is BF16; the FP8
    // branch of find_special_algo is output-dtype-agnostic so these match. Shapes are the ACTUAL
    // runtime cuBLASLt inventory from the DEBUG log (1330 calls, all previously missing) -- NOT
    // derived from weight headers (TRT-LLM FUSES qkv+z and gate+up, so derived dims were wrong).
    // NVJET algos chosen via cublasTest matmulFind + -T200 -Tme20p retime, validated at M=1/4/8.
    // Swept against the RUNTIME cuBLASLt (CUDA 13.1, libcublasLt.so.13.2.1.1) -- NVJET tile/custom
    // configs are version-specific, valid only for that cuBLASLt generation. Qwen3.6-specific keys.
    // [-algo67 -m_tile6 -m_stages38 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom16 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 2048, 12288},
        {67, 6, 38, 1, 0, 0, 16, 0}}, // GDN in_proj qkv+z fused (x210 in log): heur 61.5->51.3us (robust across M)
    // [-algo67 -m_tile3 -m_stages38 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom114 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 2048, 256}, {67, 3, 38, 1, 0, 0, 114, 0}}, // k/v proj n=256 (x280 in log): heur 8.35->6.08us (M1)
    // [-algo67 -m_tile10 -m_stages37 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom28 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 2048, 9216},
        {67, 10, 37, 1, 0, 0, 28, 0}}, // shared_expert gate+up fused (x70 in log): heur 49/110->24.8us (M1/-80% M4)
    // [-algo67 -m_tile10 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom29 -m_mma0 -m_cga0 -m_scheduling1]
    {{8, 4096, 2048},
        {67, 10, 36, 1, 0, 0, 29, 0}}, // out_proj/down_proj K=4096 (x280 in log): heur 22.5->12.3us (-45%)
};

} // namespace cublas_lut
} // namespace torch_ext

TRTLLM_NAMESPACE_END
