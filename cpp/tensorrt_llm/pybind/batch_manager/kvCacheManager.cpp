/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/pybind/common/bindTypes.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace tb = tensorrt_llm::batch_manager;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace py = pybind11;
using BlockKey = tbk::BlockKey;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;

namespace
{
std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

class PyKvCacheManager : public tbk::BaseKVCacheManager
{
public:
    // using BaseKVCacheManager::BaseKVCacheManager; // Inherit constructors
    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, allocatePools, dtype, useUvm);
    }

    void releasePools() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, releasePools);
    }

    void startScheduling() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, startScheduling);
    }

    SizeType32 getTokensPerBlock() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getTokensPerBlock);
    }

    SizeType32 getMaxNumBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getMaxNumBlocks);
    }

    SizeType32 getNumPools() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getNumPools);
    }

    tbk::KvCacheStats getKvCacheStats() const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::KvCacheStats, tbk::BaseKVCacheManager, getKvCacheStats);
    }

    SizeType32 getMaxBlocksPerSeq() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getMaxBlocksPerSeq);
    }

    SizeType32 getNeededBlocksOneStep(tb::LlmRequest const& req, bool twoStepsLookAhead) const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getNeededBlocksOneStep, req, twoStepsLookAhead);
    }

    SizeType32 getRemainingBlocksToCompletion(tb::LlmRequest const& req) const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getRemainingBlocksToCompletion, req);
    }

    void addToken(tb::LlmRequest::RequestIdType requestId) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, addToken, requestId);
    }

    void addSequence(tb::LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest> llmRequest = std::nullopt) override
    {
        PYBIND11_OVERLOAD_PURE(
            void, tbk::BaseKVCacheManager, addSequence, requestId, inputLength, beamWidth, llmRequest);
    }

    void removeSequence(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest = std::nullopt) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, removeSequence, requestId, llmRequest);
    }

    tbk::GenerationRequest const& getSequence(tb::LlmRequest::RequestIdType requestId) const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::GenerationRequest const&, tbk::BaseKVCacheManager, getSequence, requestId);
    }

    void schedulingRemoveSequence(tb::LlmRequest::RequestIdType requestId) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, schedulingRemoveSequence, requestId);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::UniquePtr, tbk::BaseKVCacheManager, getBlockPoolPointers);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::UniquePtr, tbk::BaseKVCacheManager, getLayerToPoolMapping);
    }

    void getBlockOffsetsOfBatch(tensorrt_llm::runtime::ITensor& output, SizeType32 firstBatchSlotIdx,
        SizeType32 batchSize, SizeType32 beamWidth) const override
    {
        PYBIND11_OVERLOAD_PURE(
            void, tbk::BaseKVCacheManager, getBlockOffsetsOfBatch, output, firstBatchSlotIdx, batchSize, beamWidth);
    }

    SizeType32 copyBlockOffsets(tensorrt_llm::runtime::ITensor& output, SizeType32 outputSlotOffset,
        tb::LlmRequest::RequestIdType requestId) const override
    {
        PYBIND11_OVERLOAD_PURE(
            SizeType32, tbk::BaseKVCacheManager, copyBlockOffsets, output, outputSlotOffset, requestId);
    }

    bool isEnableBlockReuse() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, isEnableBlockReuse);
    }

    bool isUseOneMoreBlock() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, isUseOneMoreBlock);
    }

    void rewindKVCache(tb::LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, rewindKVCache, requestId, rewindLengths);
    }

    bool isCrossKv() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, isCrossKv);
    }

    std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, tb::LlmRequest const& llmRequest) const override
    {
        PYBIND11_OVERLOAD_PURE(
            std::optional<BlockKey>, tbk::BaseKVCacheManager, findNewContextBlock, uniqueTokens, llmRequest);
    }

    void storeContextBlocks(tb::LlmRequest const& llmRequest) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, storeContextBlocks, llmRequest);
    }

    bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, schedulingHasFreeBlocks, numRequired);
    }

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(tb::LlmRequest::RequestIdType requestId) const override
    {
        PYBIND11_OVERLOAD_PURE(
            std::vector<std::vector<SizeType32>> const&, tbk::BaseKVCacheManager, getCacheBlockIds, requestId);
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<tb::LlmRequest::RequestIdType> const& requestIds) const override
    {
        PYBIND11_OVERLOAD_PURE(std::vector<std::vector<std::vector<SizeType32>>>, tbk::BaseKVCacheManager,
            getBatchCacheBlockIds, requestIds);
    }

    SizeType32 getUsedNumBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getUsedNumBlocks);
    }

    SizeType32 getNumFreeBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getNumFreeBlocks);
    }

    tbk::BlockManager const& getBlockManager() const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::BlockManager const&, tbk::BaseKVCacheManager, getBlockManager);
    }

    std::deque<tensorrt_llm::executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        PYBIND11_OVERLOAD_PURE(
            std::deque<tensorrt_llm::executor::KVCacheEvent>, tbk::BaseKVCacheManager, getLatestEvents, timeout);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getPrimaryPool, layer_idx);
    }

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getPoolLayerIdx, layer_idx);
    }

    void refreshBlocks() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, refreshBlocks);
    }

    void flushIterationEvents() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, flushIterationEvents);
    }
};

// TODO: Deduplicate executor bindings KvCacheStats
class PyBasePeftCacheManager : public tb::BasePeftCacheManager
{
public:
    ~PyBasePeftCacheManager() override = default;

    void addRequestPeft(tb::BasePeftCacheManager::LlmRequestPtr llmRequest, bool tryGpuCache = true) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, addRequestPeft, llmRequest, tryGpuCache);
    }

    tb::BasePeftCacheManager::PeftTable ensureBatch(tb::RequestVector const& contextRequests,
        tb::RequestVector const& generationRequests, bool resetGpuCache = false) override
    {
        PYBIND11_OVERLOAD_PURE(tb::BasePeftCacheManager::PeftTable, tb::BasePeftCacheManager, ensureBatch,
            contextRequests, generationRequests, resetGpuCache);
    }

    void resetDeviceCache() override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, resetDeviceCache);
    }

    void markRequestDone(tb::LlmRequest const& llmReq, bool pause = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, markRequestDone, llmReq, pause);
    }

    tr::SizeType32 getMaxDevicePages() const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, getMaxDevicePages);
    }

    tr::SizeType32 getMaxHostPages() const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, getMaxHostPages);
    }

    tr::SizeType32 determineNumPages(std::shared_ptr<tb::LlmRequest> llmRequest) const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, determineNumPages, llmRequest);
    }

    bool enabled() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tb::BasePeftCacheManager, enabled);
    }
};
} // namespace

void tb::kv_cache_manager::KVCacheManagerBindings::initBindings(py::module_& m)
{
    py::class_<tbk::KvCacheStats>(m, "KvCacheStats")
        .def(py::init<>())
        .def_readwrite("max_num_blocks", &tbk::KvCacheStats::maxNumBlocks)
        .def_readwrite("free_num_blocks", &tbk::KvCacheStats::freeNumBlocks)
        .def_readwrite("used_num_blocks", &tbk::KvCacheStats::usedNumBlocks)
        .def_readwrite("tokens_per_block", &tbk::KvCacheStats::toksPerBlock)
        .def_readwrite("alloc_total_blocks", &tbk::KvCacheStats::allocTotalBlocks)
        .def_readwrite("alloc_new_blocks", &tbk::KvCacheStats::allocNewBlocks)
        .def_readwrite("reused_blocks", &tbk::KvCacheStats::reusedBlocks);

    py::classh<tbk::BaseKVCacheManager, PyKvCacheManager>(m, "BaseKVCacheManager")
        .def_static("calculate_max_num_blocks", &tbk::BaseKVCacheManager::calculateMaxNumBlocks, py::arg("config"),
            py::arg("dtype"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"))
        .def("allocate_pools", &BaseKVCacheManager::allocatePools)
        .def("release_pools", &BaseKVCacheManager::releasePools)
        .def("start_scheduling", &BaseKVCacheManager::startScheduling)
        .def_property_readonly("tokens_per_block", &BaseKVCacheManager::getTokensPerBlock)
        .def_property_readonly("max_num_blocks", &BaseKVCacheManager::getMaxNumBlocks)
        .def_property_readonly("num_pools", &BaseKVCacheManager::getNumPools)
        .def("get_kv_cache_stats", &BaseKVCacheManager::getKvCacheStats)
        .def_property_readonly("max_blocks_per_seq", &BaseKVCacheManager::getMaxBlocksPerSeq)
        .def("get_needed_blocks_one_step", &BaseKVCacheManager::getNeededBlocksOneStep)
        .def("get_remaining_blocks_to_completion", &BaseKVCacheManager::getRemainingBlocksToCompletion)
        .def("add_token", &BaseKVCacheManager::addToken)
        .def("add_sequence", &BaseKVCacheManager::addSequence)
        .def("remove_sequence", &BaseKVCacheManager::removeSequence)
        .def("scheduling_remove_sequence", &BaseKVCacheManager::schedulingRemoveSequence)
        .def("get_block_pool_pointers",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> block_pool_pointers{std::nullopt};
                auto tensor = self.getBlockPoolPointers();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    block_pool_pointers = tr::Torch::tensor(_tensor);
                }
                return block_pool_pointers;
            })
        .def("get_layer_to_pool_mapping",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> layer_to_pool_mapping{std::nullopt};
                auto tensor = self.getLayerToPoolMapping();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    layer_to_pool_mapping = tr::Torch::tensor(_tensor);
                }
                return layer_to_pool_mapping;
            })
        .def("get_primary_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getPrimaryPool(layer_idx));
                auto pool_layer_idx = self.getPoolLayerIdx(layer_idx);
                return pool.index({torch::indexing::Slice(), pool_layer_idx});
            })
        .def("get_block_offsets_of_batch",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                SizeType32 beamWidth)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
            })
        .def("copy_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 outputSlotOffset,
                tb::LlmRequest::RequestIdType requestId)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                auto maxBlockCount = self.copyBlockOffsets(*(_output.value()), outputSlotOffset, requestId);
                return maxBlockCount;
            })
        .def("copy_batch_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output,
                std::vector<tb::LlmRequest::RequestIdType> const& requestIds)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                for (size_t i = 0; i < requestIds.size(); ++i)
                {
                    self.copyBlockOffsets(*(_output.value()), i, requestIds[i]);
                }
            })
        .def_property_readonly("enable_block_reuse", &BaseKVCacheManager::isEnableBlockReuse)
        .def_property_readonly("use_one_more_block", &BaseKVCacheManager::isUseOneMoreBlock)
        .def("rewind_kv_cache", &BaseKVCacheManager::rewindKVCache)
        .def_property_readonly("cross_kv", &BaseKVCacheManager::isCrossKv)
        .def("store_context_blocks", &BaseKVCacheManager::storeContextBlocks)
        .def("scheduling_has_free_blocks", &BaseKVCacheManager::schedulingHasFreeBlocks)
        .def("get_cache_block_ids", &BaseKVCacheManager::getCacheBlockIds)
        .def("get_batch_cache_block_ids", &BaseKVCacheManager::getBatchCacheBlockIds);

    py::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS);

    py::classh<tbk::KVCacheManager, tbk::BaseKVCacheManager>(m, "KVCacheManager")
        .def(py::init<std::vector<SizeType32> const&, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, SizeType32, SizeType32, SizeType32, bool, int64_t, bool, bool, tbk::CacheType>(),
            py::arg("num_kv_heads_per_layer"), py::arg("size_per_head"), py::arg("tokens_per_block"),
            py::arg("blocks_in_primary_pool"), py::arg("blocks_in_secondary_pool"), py::arg("max_num_sequences"),
            py::arg("max_beam_width"), py::arg("max_attention_window"), py::arg("temporary_attention_window"),
            py::arg("sink_token_length"), py::arg("stream"), py::arg("max_sequence_length"),
            py::arg("enable_block_reuse") = false, py::arg("onboard_blocks") = true,
            py::arg_v("cache_type", tbk::CacheType::kSELF, "bindings.internal.batch_manager.CacheType.SELF"));
}

void tb::BasePeftCacheManagerBindings::initBindings(py::module_& m)
{
    py::classh<tb::BasePeftCacheManager, PyBasePeftCacheManager>(m, "BasePeftCacheManager")
        .def("add_request_peft", &tb::BasePeftCacheManager::addRequestPeft, py::arg("request"),
            py::arg("try_gpu_cache") = true)
        .def("ensure_batch", &tb::BasePeftCacheManager::ensureBatch, py::arg("context_requests"),
            py::arg("generation_requests"), py::arg("reset_gpu_cache") = false)
        .def("reset_device_cache", &tb::BasePeftCacheManager::resetDeviceCache)
        .def("mark_request_done", &tb::BasePeftCacheManager::markRequestDone, py::arg("request"),
            py::arg("pause") = false)
        .def_property_readonly("max_device_pages", &tb::BasePeftCacheManager::getMaxDevicePages)
        .def_property_readonly("max_host_pages", &tb::BasePeftCacheManager::getMaxHostPages)
        .def("determine_num_pages", &tb::BasePeftCacheManager::determineNumPages, py::arg("request"))
        .def_property_readonly("enabled", &tb::BasePeftCacheManager::enabled);

    py::classh<tb::PeftCacheManager, tb::BasePeftCacheManager>(m, "PeftCacheManager")
        .def(py::init<tb::PeftCacheManagerConfig, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            py::arg("config"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"));

    py::classh<tb::NoOpPeftCacheManager, tb::BasePeftCacheManager>(m, "NoOpPeftCacheManager").def(py::init());
}
