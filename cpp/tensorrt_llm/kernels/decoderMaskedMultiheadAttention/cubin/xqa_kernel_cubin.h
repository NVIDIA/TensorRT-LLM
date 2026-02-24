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

#include "tensorrt_llm/common/config.h"
#include <cstddef>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
struct XQAKernelMetaInfo
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
    const unsigned long long* mCubin;
    unsigned int mCubinSize;
    char const* mFuncName;
};

extern XQAKernelMetaInfo const sXqaKernelMetaInfo[];
extern size_t const sXqaKernelMetaInfoSize;

// clang-format on
} // namespace kernels

TRTLLM_NAMESPACE_END
