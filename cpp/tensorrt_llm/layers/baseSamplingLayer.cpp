/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/fillBuffers.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator, cudaDeviceProp* prop)
    : BaseLayer(stream, std::move(allocator), prop)
    , mMaxBatchSize(maxBatchSize)
    , mVocabSize(vocabSize)
    , mVocabSizePadded(vocabSizePadded)
{
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
