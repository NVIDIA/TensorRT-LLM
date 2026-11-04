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

// BF16 MoE experts stage: C ABI entry points (included inside the library's extern "C" block).

int msoMkBfWsBytes()
{
    return mso_mk::kBfWsBytes;
}

int msoMkBfDbgBytes()
{
    return mso_mk::kBfDbgBytes;
}

int64_t msoMkBfMagic()
{
    return mso_mk::kBfMagic;
}

int msoMkBfStages()
{
    return mso_mk::kBfStages;
}

int msoMkBfN()
{
    return mso_mk::kBfN;
}

int msoMkBfUnitKb()
{
    return mso_mk::kBfUkb;
}

// 1 when the build contains the stage (MSO_MK_LANES bits 16 and 4).
int msoMkBfCompiled()
{
#if (MSO_MK_LANES & 16) && (MSO_MK_LANES & 4)
    return 1;
#else
    return 0;
#endif
}

// The 128-byte CUtensorMap of a BlockMajorK BF16 stack [E][K / 64][R][64] seen as [E (K / 64) planes][R rows][128 B]
// (u8), SWIZZLE_128B boxes of 128 B x 128 rows x kBfUkb planes (one unit), written to out (host memory).
int msoMkBfTmap(void* out, void* base, unsigned long long rows, unsigned long long planes)
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
    cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(planes)};
    cuuint64_t strides[2] = {128, 128ull * rows};
    cuuint32_t box[3] = {128, 128, static_cast<cuuint32_t>(mso_mk::kBfUkb)};
    cuuint32_t elem[3] = {1, 1, 1};
    CUresult const r = encode(static_cast<CUtensorMap*>(out), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, base, dims, strides,
        box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return static_cast<int>(r);
}
