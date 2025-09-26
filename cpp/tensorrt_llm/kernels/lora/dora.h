/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include <NvInferRuntime.h>

namespace tensorrt_llm::kernels
{
class DoraImpl
{
public:
    DoraImpl() = delete;

    DoraImpl(std::vector<int> const& outHiddenSizes, nvinfer1::DataType type);

    ~DoraImpl() = default;

    size_t getWorkspaceElemCount(int64_t const numTokens) const;
    size_t getWorkspaceSize(int64_t const numTokens) const;

    int run(int64_t numTokens, void const* input, void const* const* loraWeightsPtr, void* const* outputs,
        void* workspace, cudaStream_t stream);

private:
    std::vector<int64_t> mCumModuleSizes;
    std::vector<int64_t> mHostBuf;
    nvinfer1::DataType mType;
};
} // namespace tensorrt_llm::kernels
