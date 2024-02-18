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

#pragma once

#include <curand_kernel.h>

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

//! \brief Base class for sampling layers.
//! Layer modifies logits in-place.
template <typename T>
class BaseSamplingLayer : public BaseLayer
{
public:
    class SetupParams : public DecodingSetupParams
    {
    public:
        std::optional<std::vector<std::uint32_t>> runtime_top_k;  // [1] or [batchSize] on cpu
        std::optional<std::vector<float>> runtime_top_p;          // [1] or [batchSize] on cpu
        std::optional<std::vector<uint64_t>> randomSeed;          // [1] or [batchSize] on cpu
        std::optional<std::vector<float>> top_p_decay;            // [batchSize], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;              // [batchSize], must between [0, 1]
        std::optional<std::vector<std::int32_t>> top_p_reset_ids; // [batchSize]
        std::optional<bool> normalize_log_probs;
    };

    class ForwardParams : public DecodingParams
    {
    public:
        ForwardParams(int step, int ite, tc::Tensor logits, tc::Tensor end_ids, int max_seq_len)
            : DecodingParams{step, ite, std::move(logits), std::move(end_ids)}
            , max_seq_len{max_seq_len}
        {
        }

        // mandatory parameters
        int max_seq_len;

        // optional parameters
        std::optional<tc::Tensor> input_lengths; // [localBatchSize]
        curandState_t* curand_states;            // [localBatchSize]
        // Pointer to the workspace for sampling computation
        void* sampling_workspace;
        // Flag to mark that logits tensor contains probabilities
        bool probs_computed;
    };

    // clang-format off
    //! \brief Constructor.
    //!
    //! \param maxBatchSize Maximum batch size configured in the system
    //! \param vocabSize Unpadded size of the vocabulary
    //! \param vocabSizePadded Padded size of the vocabulary
    //! \param stream cuda stream
    //! \param allocator shared pointer to IAllocator object that will be use to alloc and free tensors
    //! \param prop [optional] cudaDeviceProp
    // clang-format on
    BaseSamplingLayer(size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, cudaDeviceProp* prop);

    ~BaseSamplingLayer() override = default;

    // clang-format off
    //! \brief Executes sampling layer.
    //! Applies temperature, repetition/presence penalties and minLength penalty.
    //! Then calls runSampling.
    //! It exits early if mSkipDecodeHost is set to skip this layer for all requests in the batch
    //!
    //! \param outputs DecodingOutputParams struct with output tensors
    //! \param inputs ForwardParams struct with input tensors and params
    //! \param curandStatesDevice Properly initialized curand states buffer on device
    // clang-format on
    virtual void forward(DecodingOutputParams& outputs, ForwardParams& inputs) = 0;

    // clang-format off
    //! \brief Virtual function that setups internal tensors of the layer with sampling params
    //! specified in setupParams for the entries specified by batchSlots.
    //! It updates data for new requests in internal tensors inplace.
    //! Thus, it must be called only once for new requests.
    //!
    //! \param batchSize Maximum batch size configured in the system
    //! \param batchSlots input tensor [batchSize], address map of the new requests, in pinned memory
    //! \param setupParams setup sampling parameters per request
    // clang-format on
    virtual void setup(size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams) = 0;

    size_t getWorkspaceSize() const
    {
        return mSamplingWorkspaceSize;
    }

    size_t getAllocatedSize() const
    {
        return mAllocatedSize;
    }

protected:
    size_t mMaxBatchSize;
    size_t mVocabSize;
    size_t mVocabSizePadded;

    size_t mSamplingWorkspaceSize = 0;
    size_t mAllocatedSize = 0;
};

} // namespace layers
} // namespace tensorrt_llm
