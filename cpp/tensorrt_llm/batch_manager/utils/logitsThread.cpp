/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "logitsThread.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::batch_manager::utils
{

enum class FastLogitsMpiId : uint64_t
{
    ASK_TENSOR = 1,
    SEND_TENSOR = 2,
};

void draftModelSendLogitsThread(int device, std::atomic<bool>* draftModelThreadShouldExit,
    RequestVector* draftRequestsWaitingToSendLogits, std::shared_ptr<SequenceSlotManager> const& seqSlotManager,
    SizeType32 maxInputLen, std::shared_ptr<kv_cache_manager::BaseKVCacheManager> const& kvCacheManager,
    std::shared_ptr<kv_cache_manager::BaseKVCacheManager> const& crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> const& peftCacheManager)
{
#if ENABLE_MULTI_DEVICE
    TLLM_CUDA_CHECK(cudaSetDevice(device));
    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();

    bool msgReady{false};
    MPI_Message msg = nullptr;
    MPI_Status status;

    while (true)
    {
        msgReady = worldComm.improbe(MPI_ANY_SOURCE, mpi::MpiTag::kSpecDecLogitsId, &msg, &status);

        if (!msgReady)
        {
            if (*draftModelThreadShouldExit)
            {
                TLLM_LOG_INFO("Draft model sender thread exiting");
                break;
            }

            continue;
        }

        int const source_rank = status.MPI_SOURCE;

        int32_t count = 0;
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        TLLM_CHECK(count == 1);

        FastLogitsMpiId mpiId{};
        MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status));

        TLLM_CHECK(mpiId == FastLogitsMpiId::ASK_TENSOR);

        worldComm.mprobe(MPI_ANY_SOURCE, mpi::MpiTag::kSpecDecLogitsData, &msg, &status);
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        TLLM_CHECK(count == 1);

        uint64_t draftRequestId = 0;
        MPICHECK(MPI_Mrecv(&draftRequestId, count, MPI_UINT64_T, &msg, &status));

        auto const findDraftRequest
            = [draftRequestId, draftRequestsWaitingToSendLogits]() -> std::shared_ptr<LlmRequest>
        {
            for (auto it = draftRequestsWaitingToSendLogits->begin(); it != draftRequestsWaitingToSendLogits->end();
                 ++it)
            {
                auto req = *it;
                if (req->mRequestId == draftRequestId)
                {
                    draftRequestsWaitingToSendLogits->erase(it);
                    return req;
                }
            }
            return nullptr;
        };

        std::shared_ptr<LlmRequest> draftRequest = findDraftRequest();
        TLLM_CHECK(draftRequest != nullptr);

        auto draftLogits = runtime::ITensor::slice(draftRequest->getGenerationLogitsHost(), {0, 0});

        auto const shape = draftLogits->getShape();
        TLLM_CHECK(shape.nbDims == 2);

        FastLogitsMpiId constexpr id{FastLogitsMpiId::SEND_TENSOR};
        worldComm.send(&id, 1, mpi::MpiType::kUINT64, source_rank, mpi::MpiTag::kSpecDecLogitsId);

        worldComm.send(shape.d, 2, mpi::MpiType::kINT64, source_rank, mpi::MpiTag::kSpecDecLogitsData);
        worldComm.send(draftLogits->data(), draftLogits->getSizeInBytes(), mpi::MpiType::kUINT8, source_rank,
            mpi::MpiTag::kSpecDecLogitsData);

        terminateRequest(
            *seqSlotManager, *draftRequest, maxInputLen, kvCacheManager, crossKvCacheManager, peftCacheManager);
    }
#endif // ENABLE_MULTI_DEVICE
}

void targetModelReceiveLogits(runtime::ITensor::SharedPtr& draftLogitsHost,
    executor::SpeculativeDecodingFastLogitsInfo const& fastLogitsInfo, nvinfer1::DataType logitsDtype)
{
#if ENABLE_MULTI_DEVICE
    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();

    FastLogitsMpiId mpiId{FastLogitsMpiId::ASK_TENSOR};
    worldComm.send(&mpiId, 1, mpi::MpiType::kUINT64, fastLogitsInfo.draftParticipantId, mpi::MpiTag::kSpecDecLogitsId);
    worldComm.send(&fastLogitsInfo.draftRequestId, 1, mpi::MpiType::kUINT64, fastLogitsInfo.draftParticipantId,
        mpi::MpiTag::kSpecDecLogitsData);

    MPI_Message msg;
    MPI_Status status;
    worldComm.mprobe(fastLogitsInfo.draftParticipantId, mpi::MpiTag::kSpecDecLogitsId, &msg, &status);

    int32_t count;
    MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
    TLLM_CHECK(count == 1);

    MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status));
    TLLM_CHECK(mpiId == FastLogitsMpiId::SEND_TENSOR);

    worldComm.mprobe(fastLogitsInfo.draftParticipantId, mpi::MpiTag::kSpecDecLogitsData, &msg, &status);

    MPICHECK(MPI_Get_count(&status, MPI_INT64_T, &count));
    TLLM_CHECK(count == 2);

    int64_t dims[2];
    MPICHECK(MPI_Mrecv(&dims, count, MPI_INT64_T, &msg, &status));

    draftLogitsHost->reshape(runtime::ITensor::makeShape({dims[0], dims[1]}));

    worldComm.mprobe(fastLogitsInfo.draftParticipantId, mpi::MpiTag::kSpecDecLogitsData, &msg, &status);

    MPICHECK(MPI_Get_count(&status, MPI_UINT8_T, &count));

    uint64_t const expectedSize = static_cast<uint64_t>(dims[0]) * dims[1] * tc::getDTypeSize(logitsDtype);
    TLLM_CHECK((uint64_t) count == expectedSize);

    MPICHECK(MPI_Mrecv(draftLogitsHost->data(), count, MPI_UINT8_T, &msg, &status));
#endif // ENABLE_MULTI_DEVICE
}

} // namespace tensorrt_llm::batch_manager::utils
