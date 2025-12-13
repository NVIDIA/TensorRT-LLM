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
#pragma once

#include "../kernelParams.h"
#include "tensorrt_llm/common/config.h"

#include <cstddef>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

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
    unsigned char const* mCubin;
    unsigned int mCubinSize;
    char const* mFuncName;
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
    bool mSkipsSoftmaxWhenPossible;
    char const* sha256;
};

extern TllmGenFmhaKernelMetaInfo const sTllmGenFmhaKernelMetaInfos[];
extern size_t const sTllmGenFmhaKernelMetaInfosSize;
// clang-format on
} // namespace kernels

TRTLLM_NAMESPACE_END
