/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off

#define TLLM_GEN_VERSION "571f61a5-dirty"
#include "kernelMetaInfoForwardDecl.def"
} // namespace kernels
} // namespace tensorrt_llm

namespace tensorrt_llm {
namespace _v1 {
namespace kernels {
using namespace tensorrt_llm::kernels;

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
constexpr int32_t kSM_100f = 10100;
constexpr int32_t kSM_103 = 103;
constexpr int32_t kSM_120 = 120;
constexpr int32_t kSM_121 = 121;
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E2M1,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};
struct TllmGenFmhaKernelMetaInfo
{
    Data_type mDataTypeQ;
    Data_type mDataTypeKv;
    Data_type mDataTypeO;
    int mTileSizeQ;
    int mTileSizeKv;
    int mStepQ;
    int mStepKv;
    int mHeadDimPerCtaV;
    int mHeadDimQk;
    int mHeadDimV;
    int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    int mSharedMemBytes;
    int mThreadsPerCTA;
    int mQkvLayout;
    int mNumTokensPerPage;
    int mMaskType;
    int mKernelType;
    int mMaxNumHeadsQPerKvInCta;
    int mTileScheduler;
    int mMultiCtasKvMode;
    bool mGroupsHeadsQ;
    bool mReuseSmemKForV;
    bool m2CtaMma;
    bool mSparseMla;
    bool mReservedParam;
    const char* sha256;
};

extern const TllmGenFmhaKernelMetaInfo sTllmGenFmhaKernelMetaInfos[] = {
#include "kernelMetaInfo1.def"
#include "kernelMetaInfo2.def"
};

extern const size_t sTllmGenFmhaKernelMetaInfosSize = sizeof(sTllmGenFmhaKernelMetaInfos) / sizeof(sTllmGenFmhaKernelMetaInfos[0]);

// clang-format on
} // namespace kernels
} // namespace _v1
} // namespace tensorrt_llm
