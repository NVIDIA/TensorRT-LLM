/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// NVFP4 MoE experts stage: C ABI entry points (included inside the library's extern "C" block).

// The layout constants the host (experts.py) asserts against.
int msoMkMoeWsBytes()
{
    return mso_mk::kMoeWsBytes;
}

int msoMkMoePartBytes()
{
    return mso_mk::kMoePartBytes;
}

int msoMkMoeDbgBytes()
{
    return mso_mk::kMoeDbgBytes;
}

int msoMkMoeDescWords()
{
    return mso_mk::kMdNumWords;
}

int64_t msoMkMoeMagic()
{
    return mso_mk::kMoeMagic;
}

int msoMkMoeStages()
{
    return mso_mk::kMoeStages;
}

// The 128-byte CUtensorMap of an expert stack [rows][2048 B] (u8) with SWIZZLE_128B boxes of 128 B x 128 rows, written
// to out (host memory). The driver entry point is resolved through the runtime (no -lcuda); returns a CUresult, or
// 1000 + the cudaError_t of the lookup.
int msoMkMoeTmap(void* out, void* base, unsigned long long rows)
{
    using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, cuuint64_t const*,
        cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static Encode encode = nullptr;
    if (encode == nullptr)
    {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult q{};
        cudaError_t const e
            = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
        if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || fn == nullptr)
        {
            return 1000 + static_cast<int>(e);
        }
        encode = reinterpret_cast<Encode>(fn);
    }
    cuuint64_t dims[2] = {2048, static_cast<cuuint64_t>(rows)};
    cuuint64_t strides[1] = {2048};
    cuuint32_t box[2] = {128, 128};
    cuuint32_t elem[2] = {1, 1};
    CUresult const r = encode(static_cast<CUtensorMap*>(out), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides,
        box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return static_cast<int>(r);
}
