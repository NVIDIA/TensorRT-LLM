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
namespace tbc = tensorrt_llm::batch_manager::kv_connector;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace py = pybind11;
using BlockKey = tbk::BlockKey;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;

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
    void allocatePools(bool useUvm = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, allocatePools, useUvm);
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

    std::optional<tbk::KVCacheBlock::IdType> removeSequence(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest = std::nullopt,
        bool pinOnRelease = false) override
    {
        PYBIND11_OVERLOAD_PURE(std::optional<tbk::KVCacheBlock::IdType>, tbk::BaseKVCacheManager, removeSequence,
            requestId, llmRequest, pinOnRelease);
    }

    std::optional<tbk::KVCacheBlock::IdType> storeBlocksForReuse(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest, bool pinBlocks) override
    {
        PYBIND11_OVERLOAD_PURE(std::optional<tbk::KVCacheBlock::IdType>, tbk::BaseKVCacheManager, storeBlocksForReuse,
            requestId, llmRequest, pinBlocks);
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

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        tb::LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override
    {
        PYBIND11_OVERLOAD_PURE(std::vector<std::vector<SizeType32>> const&, tbk::BaseKVCacheManager, getCacheBlockIds,
            requestId, windowSize);
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override
    {
        PYBIND11_OVERLOAD_PURE(std::vector<std::vector<std::vector<SizeType32>>>, tbk::BaseKVCacheManager,
            getBatchCacheBlockIds, requestIds, windowSize);
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

    tensorrt_llm::runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getPrimaryPool, poolIdx);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getUniquePrimaryPool() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getUniquePrimaryPool);
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
        .def_readwrite("reused_blocks", &tbk::KvCacheStats::reusedBlocks)
        .def_readwrite("missed_blocks", &tbk::KvCacheStats::missedBlocks)
        .def_readwrite("cache_hit_rate", &tbk::KvCacheStats::cacheHitRate)
        .def_readwrite("num_free_blocks_per_window_size", &tbk::KvCacheStats::numFreeBlocksPerWindowSize)
        .def_readonly("allocated_bytes", &tbk::KvCacheStats::allocatedBytes);

    py::class_<tbk::TempAttentionWindowInputs>(m, "TempAttentionWindowInputs")
        .def(py::init<>())
        .def_readwrite("paged_context_fmha", &tbk::TempAttentionWindowInputs::pagedContextFMHA)
        .def_readwrite("max_input_len", &tbk::TempAttentionWindowInputs::maxInputLen)
        .def_readwrite("max_num_tokens", &tbk::TempAttentionWindowInputs::maxNumTokens);

    py::class_<tbk::BlockKey>(m, "BlockKey")
        .def(py::init<>())
        .def(py::init<VecTokens const&, std::optional<tr::LoraTaskIdType>>(), py::arg("tokens"),
            py::arg("lora_task_id") = std::nullopt)
        .def(py::init<bool, std::optional<tr::LoraTaskIdType>, VecUniqueTokens const&>(), py::arg("uses_extra_ids"),
            py::arg("lora_task_id"), py::arg("unique_tokens"))
        .def_readonly("uses_extra_ids", &tbk::BlockKey::usesExtraIds)
        .def_readonly("lora_task_id", &tbk::BlockKey::loraTaskId)
        .def_readonly("unique_tokens", &tbk::BlockKey::uniqueTokens);

    py::class_<tbk::BlockKeyHasher>(m, "BlockKeyHasher")
        .def_static("hash", &tbk::BlockKeyHasher::hash, py::arg("block_key"), py::arg("parent_hash") = 0);

    py::class_<tbk::KVCacheEventManager, std::shared_ptr<tbk::KVCacheEventManager>>(m, "KVCacheEventManager")
        .def(py::init<size_t, std::optional<SizeType32>, std::optional<SizeType32>, SizeType32>(),
            py::arg("max_kv_event_entries"), py::arg("attention_dp_rank") = std::nullopt,
            py::arg("attention_dp_size") = std::nullopt, py::arg("attention_dp_events_gather_period_ms") = 5);

    py::classh<tbk::BaseKVCacheManager, PyKvCacheManager>(m, "BaseKVCacheManager")
        .def_static("calculate_max_num_blocks", &tbk::BaseKVCacheManager::calculateMaxNumBlocks, py::arg("config"),
            py::arg("is_cross_attention"), py::arg("dtype"), py::arg("model_config"), py::arg("world_config"),
            py::arg("window_size_to_layers"), py::arg("allotted_primary_mem_bytes"),
            py::arg("allotted_secondary_mem_bytes"), py::arg("extra_cost_memory"), py::arg("kv_factor"),
            py::call_guard<py::gil_scoped_release>())
        .def("allocate_pools", &BaseKVCacheManager::allocatePools, py::call_guard<py::gil_scoped_release>())
        .def("release_pools", &BaseKVCacheManager::releasePools, py::call_guard<py::gil_scoped_release>())
        .def("start_scheduling", &BaseKVCacheManager::startScheduling, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("tokens_per_block", &BaseKVCacheManager::getTokensPerBlock)
        .def_property_readonly("max_num_blocks", &BaseKVCacheManager::getMaxNumBlocks)
        .def_property_readonly("num_pools", &BaseKVCacheManager::getNumPools)
        .def("get_kv_cache_stats", &BaseKVCacheManager::getKvCacheStats, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("max_blocks_per_seq",
            [](tbk::BaseKVCacheManager& self) { return self.getOffsetTableDimensions().maxBlocksPerSeq; })
        .def("get_needed_blocks_one_step", &BaseKVCacheManager::getNeededBlocksOneStep,
            py::call_guard<py::gil_scoped_release>())
        .def("get_remaining_blocks_to_completion", &BaseKVCacheManager::getRemainingBlocksToCompletion,
            py::call_guard<py::gil_scoped_release>())
        .def("add_token", &BaseKVCacheManager::addToken, py::call_guard<py::gil_scoped_release>())
        .def("add_sequence", &BaseKVCacheManager::addSequence, py::call_guard<py::gil_scoped_release>())
        .def("remove_sequence", &BaseKVCacheManager::removeSequence, py::call_guard<py::gil_scoped_release>())
        .def("pin_blocks", &BaseKVCacheManager::pinBlocks, py::call_guard<py::gil_scoped_release>())
        .def("scheduling_remove_sequence", &BaseKVCacheManager::schedulingRemoveSequence,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_block_pool_pointers",
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
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_block_scale_pool_pointers",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> block_scale_pool_pointers{std::nullopt};
                auto tensor = self.getBlockScalePoolPointers();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    block_scale_pool_pointers = tr::Torch::tensor(_tensor);
                }
                return block_scale_pool_pointers;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_layer_to_pool_mapping",
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
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_primary_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getPrimaryPool(layer_idx));
                auto pool_layer_idx = self.getPoolLayerIdx(layer_idx);
                return pool.index({torch::indexing::Slice(), pool_layer_idx});
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_unique_primary_pool", [](tbk::BaseKVCacheManager& self) { return self.getUniquePrimaryPool(); },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_block_offsets_of_batch",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                SizeType32 beamWidth)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "copy_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 outputSlotOffset,
                tb::LlmRequest::RequestIdType requestId)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                auto maxBlockCount = self.copyBlockOffsets(*(_output.value()), outputSlotOffset, requestId);
                return maxBlockCount;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "copy_batch_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output,
                std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 const beamWidth,
                SizeType32 const offset)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                for (size_t i = 0; i < requestIds.size(); ++i)
                {
                    self.copyBlockOffsets(*(_output.value()), i * beamWidth + offset, requestIds[i]);
                }
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_latest_events",
            [](tbk::BaseKVCacheManager& self, std::optional<double> timeout_ms = std::nullopt)
            {
                if (timeout_ms)
                {
                    return self.getLatestEvents(std::chrono::milliseconds(static_cast<int64_t>(*timeout_ms)));
                }
                return self.getLatestEvents(std::nullopt);
            },
            py::arg("timeout_ms") = std::nullopt, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("enable_block_reuse", &BaseKVCacheManager::isEnableBlockReuse)
        .def("rewind_kv_cache", &BaseKVCacheManager::rewindKVCache, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("cross_kv", &BaseKVCacheManager::isCrossKv)
        .def("store_context_blocks", &BaseKVCacheManager::storeContextBlocks, py::call_guard<py::gil_scoped_release>())
        .def("store_blocks_for_reuse", &BaseKVCacheManager::storeBlocksForReuse,
            py::call_guard<py::gil_scoped_release>())
        .def("get_cache_block_ids", &BaseKVCacheManager::getCacheBlockIds, py::call_guard<py::gil_scoped_release>())
        .def("get_batch_cache_block_ids", &BaseKVCacheManager::getBatchCacheBlockIds,
            py::call_guard<py::gil_scoped_release>())
        .def("flush_iteration_events", &BaseKVCacheManager::flushIterationEvents,
            py::call_guard<py::gil_scoped_release>())
        .def("get_last_block_id", &BaseKVCacheManager::getLastBlockId, py::call_guard<py::gil_scoped_release>())
        .def("unpin_blocks_by_id", &BaseKVCacheManager::unpinBlocksById, py::call_guard<py::gil_scoped_release>());

    py::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS)
        .value("SELFKONLY", tbk::CacheType::kSELFKONLY);

    py::classh<tbk::KVCacheManager, tbk::BaseKVCacheManager>(m, "KVCacheManager")
        .def(py::init<std::vector<SizeType32> const&, SizeType32, SizeType32,
                 std::map<SizeType32, std::tuple<SizeType32, SizeType32>> const&, SizeType32, SizeType32,
                 std::vector<SizeType32> const&, std::optional<tbk::TempAttentionWindowInputs> const&,
                 nvinfer1::DataType, SizeType32, bool, int64_t, bool, bool, tbk::CacheType,
                 std::optional<tensorrt_llm::executor::RetentionPriority>, std::shared_ptr<tbk::KVCacheEventManager>,
                 bool, bool, std::shared_ptr<tbc::KvCacheConnectorManager>>(),
            py::arg("num_kv_heads_per_layer"), py::arg("size_per_head"), py::arg("tokens_per_block"),
            py::arg("blocks_per_window"), py::arg("max_num_sequences"), py::arg("max_beam_width"),
            py::arg("max_attention_window_vec"), py::arg("temp_attention_window_inputs"), py::arg("dtype"),
            py::arg("sink_token_length"), py::arg("stream"), py::arg("max_sequence_length"),
            py::arg("enable_block_reuse") = false, py::arg("onboard_blocks") = true,
            py::arg_v("cache_type", tbk::CacheType::kSELF, "bindings.internal.batch_manager.CacheType.SELF"),
            py::arg("secondary_offload_min_priority") = std::nullopt, py::arg("event_manager") = nullptr,
            py::arg("enable_partial_reuse") = true, py::arg("copy_on_partial_reuse") = true,
            py::arg("kv_connector_manager") = nullptr, py::call_guard<py::gil_scoped_release>());
}

void tb::BasePeftCacheManagerBindings::initBindings(py::module_& m)
{
    py::classh<tb::BasePeftCacheManager, PyBasePeftCacheManager>(m, "BasePeftCacheManager")
        .def("add_request_peft", &tb::BasePeftCacheManager::addRequestPeft, py::arg("request"),
            py::arg("try_gpu_cache") = true, py::call_guard<py::gil_scoped_release>())
        .def(
            "ensure_batch",
            [](tb::BasePeftCacheManager& self, tb::RequestVector const& contextRequests,
                tb::RequestVector const& generationRequests, bool resetGpuCache)
            { return self.ensureBatch(contextRequests, generationRequests, resetGpuCache); },
            py::arg("context_requests"), py::arg("generation_requests"), py::arg("reset_gpu_cache") = false,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "reset_device_cache", &tb::BasePeftCacheManager::resetDeviceCache, py::call_guard<py::gil_scoped_release>())
        .def("mark_request_done", &tb::BasePeftCacheManager::markRequestDone, py::arg("request"),
            py::arg("pause") = false, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("max_device_pages", &tb::BasePeftCacheManager::getMaxDevicePages)
        .def_property_readonly("max_host_pages", &tb::BasePeftCacheManager::getMaxHostPages)
        .def("determine_num_pages", &tb::BasePeftCacheManager::determineNumPages, py::arg("request"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("enabled", &tb::BasePeftCacheManager::enabled);

    py::classh<tb::PeftCacheManager, tb::BasePeftCacheManager>(m, "PeftCacheManager")
        .def(py::init<tb::PeftCacheManagerConfig, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            py::arg("config"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"),
            py::call_guard<py::gil_scoped_release>())
        .def("is_task_cached", &tb::PeftCacheManager::isTaskCached, py::arg("taskId"),
            py::call_guard<py::gil_scoped_release>());

    py::classh<tb::NoOpPeftCacheManager, tb::BasePeftCacheManager>(m, "NoOpPeftCacheManager")
        .def(py::init<>(), py::call_guard<py::gil_scoped_release>());
}
