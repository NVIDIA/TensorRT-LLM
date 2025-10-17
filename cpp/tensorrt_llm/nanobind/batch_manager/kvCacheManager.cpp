/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/nanobind/common/bindTypes.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>

namespace tb = tensorrt_llm::batch_manager;
namespace tbc = tensorrt_llm::batch_manager::kv_connector;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;
using BlockKey = tbk::BlockKey;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using CacheBlockIds = std::vector<std::vector<SizeType32>>;

NB_MAKE_OPAQUE(CacheBlockIds);

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
    NB_TRAMPOLINE(tbk::BaseKVCacheManager, 30);

    // using BaseKVCacheManager::BaseKVCacheManager; // Inherit constructors
    void allocatePools(bool useUvm = false) override
    {
        NB_OVERRIDE_PURE(allocatePools, useUvm);
    }

    void releasePools() override
    {
        NB_OVERRIDE_PURE(releasePools);
    }

    void startScheduling() override
    {
        NB_OVERRIDE_PURE(startScheduling);
    }

    SizeType32 getTokensPerBlock() const override
    {
        NB_OVERRIDE_PURE(getTokensPerBlock);
    }

    SizeType32 getMaxNumBlocks() const override
    {
        NB_OVERRIDE_PURE(getMaxNumBlocks);
    }

    SizeType32 getNumPools() const override
    {
        NB_OVERRIDE_PURE(getNumPools);
    }

    tbk::KvCacheStats getKvCacheStats() const override
    {
        NB_OVERRIDE_PURE(getKvCacheStats);
    }

    void addToken(tb::LlmRequest::RequestIdType requestId) override
    {
        NB_OVERRIDE_PURE(addToken, requestId);
    }

    void addSequence(tb::LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest> llmRequest = std::nullopt) override
    {
        NB_OVERRIDE_PURE(addSequence, requestId, inputLength, beamWidth, llmRequest);
    }

    std::optional<tbk::KVCacheBlock::IdType> removeSequence(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest = std::nullopt,
        bool pinOnRelease = false) override
    {
        NB_OVERRIDE_PURE(removeSequence, requestId, llmRequest, pinOnRelease);
    }

    std::optional<tbk::KVCacheBlock::IdType> storeBlocksForReuse(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest, bool pinBlocks) override
    {
        NB_OVERRIDE_PURE(storeBlocksForReuse, requestId, llmRequest, pinBlocks);
    }

    tbk::GenerationRequest const& getSequence(tb::LlmRequest::RequestIdType requestId) const override
    {
        NB_OVERRIDE_PURE(getSequence, requestId);
    }

    void schedulingRemoveSequence(tb::LlmRequest::RequestIdType requestId) override
    {
        NB_OVERRIDE_PURE(schedulingRemoveSequence, requestId);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        NB_OVERRIDE_PURE(getBlockPoolPointers);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        NB_OVERRIDE_PURE(getLayerToPoolMapping);
    }

    void getBlockOffsetsOfBatch(tensorrt_llm::runtime::ITensor& output, SizeType32 firstBatchSlotIdx,
        SizeType32 batchSize, SizeType32 beamWidth) const override
    {
        NB_OVERRIDE_PURE(getBlockOffsetsOfBatch, output, firstBatchSlotIdx, batchSize, beamWidth);
    }

    SizeType32 copyBlockOffsets(tensorrt_llm::runtime::ITensor& output, SizeType32 outputSlotOffset,
        tb::LlmRequest::RequestIdType requestId) const override
    {
        NB_OVERRIDE_PURE(copyBlockOffsets, output, outputSlotOffset, requestId);
    }

    bool isEnableBlockReuse() const override
    {
        NB_OVERRIDE_PURE(isEnableBlockReuse);
    }

    void rewindKVCache(tb::LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override
    {
        NB_OVERRIDE_PURE(rewindKVCache, requestId, rewindLengths);
    }

    bool isCrossKv() const override
    {
        NB_OVERRIDE_PURE(isCrossKv);
    }

    std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, tb::LlmRequest const& llmRequest) const override
    {
        NB_OVERRIDE_PURE(findNewContextBlock, uniqueTokens, llmRequest);
    }

    void storeContextBlocks(tb::LlmRequest const& llmRequest) override
    {
        NB_OVERRIDE_PURE(storeContextBlocks, llmRequest);
    }

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        tb::LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override
    {
        NB_OVERRIDE_PURE(getCacheBlockIds, requestId, windowSize);
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override
    {
        NB_OVERRIDE_PURE(getBatchCacheBlockIds, requestIds, windowSize);
    }

    SizeType32 getUsedNumBlocks() const override
    {
        NB_OVERRIDE_PURE(getUsedNumBlocks);
    }

    SizeType32 getNumFreeBlocks() const override
    {
        NB_OVERRIDE_PURE(getNumFreeBlocks);
    }

    tbk::BlockManager const& getBlockManager() const override
    {
        NB_OVERRIDE_PURE(getBlockManager);
    }

    std::deque<tensorrt_llm::executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        NB_OVERRIDE_PURE(getLatestEvents, timeout);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override
    {
        NB_OVERRIDE_PURE(getPrimaryPool, layer_idx);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getIndexerKCachePool() const override
    {
        NB_OVERRIDE_PURE(getIndexerKCachePool);
    }

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        NB_OVERRIDE_PURE(getPoolLayerIdx, layer_idx);
    }

    void refreshBlocks() override
    {
        NB_OVERRIDE_PURE(refreshBlocks);
    }

    void flushIterationEvents() override
    {
        NB_OVERRIDE_PURE(flushIterationEvents);
    }
};

// TODO: Deduplicate executor bindings KvCacheStats
class PyBasePeftCacheManager : public tb::BasePeftCacheManager
{
public:
    ~PyBasePeftCacheManager() override = default;

    NB_TRAMPOLINE(tb::BasePeftCacheManager, 8);

    void addRequestPeft(tb::BasePeftCacheManager::LlmRequestPtr llmRequest, bool tryGpuCache = true) override
    {
        NB_OVERRIDE_PURE(addRequestPeft, llmRequest, tryGpuCache);
    }

    tb::BasePeftCacheManager::PeftTable ensureBatch(tb::RequestVector const& contextRequests,
        tb::RequestVector const& generationRequests, bool resetGpuCache = false) override
    {
        NB_OVERRIDE_PURE(ensureBatch, contextRequests, generationRequests, resetGpuCache);
    }

    void resetDeviceCache() override
    {
        NB_OVERRIDE_PURE(resetDeviceCache);
    }

    void markRequestDone(tb::LlmRequest const& llmReq, bool pause = false) override
    {
        NB_OVERRIDE_PURE(markRequestDone, llmReq, pause);
    }

    tr::SizeType32 getMaxDevicePages() const override
    {
        NB_OVERRIDE_PURE(getMaxDevicePages);
    }

    tr::SizeType32 getMaxHostPages() const override
    {
        NB_OVERRIDE_PURE(getMaxHostPages);
    }

    tr::SizeType32 determineNumPages(std::shared_ptr<tb::LlmRequest> llmRequest) const override
    {
        NB_OVERRIDE_PURE(determineNumPages, llmRequest);
    }

    bool enabled() const override
    {
        NB_OVERRIDE_PURE(enabled);
    }
};
} // namespace

void tb::kv_cache_manager::KVCacheManagerBindings::initBindings(nb::module_& m)
{
    nb::class_<tbk::KvCacheStats>(m, "KvCacheStats")
        .def(nb::init<>())
        .def_rw("max_num_blocks", &tbk::KvCacheStats::maxNumBlocks)
        .def_rw("free_num_blocks", &tbk::KvCacheStats::freeNumBlocks)
        .def_rw("used_num_blocks", &tbk::KvCacheStats::usedNumBlocks)
        .def_rw("tokens_per_block", &tbk::KvCacheStats::toksPerBlock)
        .def_rw("alloc_total_blocks", &tbk::KvCacheStats::allocTotalBlocks)
        .def_rw("alloc_new_blocks", &tbk::KvCacheStats::allocNewBlocks)
        .def_rw("reused_blocks", &tbk::KvCacheStats::reusedBlocks)
        .def_rw("missed_blocks", &tbk::KvCacheStats::missedBlocks)
        .def_rw("cache_hit_rate", &tbk::KvCacheStats::cacheHitRate)
        .def_rw("num_free_blocks_per_window_size", &tbk::KvCacheStats::numFreeBlocksPerWindowSize)
        .def_ro("allocated_bytes", &tbk::KvCacheStats::allocatedBytes);

    nb::class_<tbk::TempAttentionWindowInputs>(m, "TempAttentionWindowInputs")
        .def(nb::init<>())
        .def_rw("paged_context_fmha", &tbk::TempAttentionWindowInputs::pagedContextFMHA)
        .def_rw("max_input_len", &tbk::TempAttentionWindowInputs::maxInputLen)
        .def_rw("max_num_tokens", &tbk::TempAttentionWindowInputs::maxNumTokens);

    nb::class_<tbk::BlockKey>(m, "BlockKey")
        .def(nb::init<>())
        .def(nb::init<VecTokens const&, std::optional<tr::LoraTaskIdType>>(), nb::arg("tokens"),
            nb::arg("lora_task_id") = std::nullopt)
        .def(nb::init<bool, std::optional<tr::LoraTaskIdType>, VecUniqueTokens const&>(), nb::arg("uses_extra_ids"),
            nb::arg("lora_task_id"), nb::arg("unique_tokens"))
        .def_ro("uses_extra_ids", &tbk::BlockKey::usesExtraIds)
        .def_ro("lora_task_id", &tbk::BlockKey::loraTaskId)
        .def_ro("unique_tokens", &tbk::BlockKey::uniqueTokens);

    nb::class_<tbk::BlockKeyHasher>(m, "BlockKeyHasher")
        .def_static("hash", &tbk::BlockKeyHasher::hash, nb::arg("block_key"), nb::arg("parent_hash") = 0);

    nb::class_<tbk::KVCacheEventManager>(m, "KVCacheEventManager")
        .def(nb::init<size_t, std::optional<SizeType32>, std::optional<SizeType32>, SizeType32>(),
            nb::arg("max_kv_event_entries"), nb::arg("attention_dp_rank") = std::nullopt,
            nb::arg("attention_dp_size") = std::nullopt, nb::arg("attention_dp_events_gather_period_ms") = 5);

    nb::class_<tbk::BaseKVCacheManager, PyKvCacheManager>(m, "BaseKVCacheManager")
        .def_static("calculate_max_num_blocks", &tbk::BaseKVCacheManager::calculateMaxNumBlocks, nb::arg("config"),
            nb::arg("is_cross_attention"), nb::arg("dtype"), nb::arg("model_config"), nb::arg("world_config"),
            nb::arg("window_size_to_layers"), nb::arg("allotted_primary_mem_bytes"),
            nb::arg("allotted_secondary_mem_bytes"), nb::arg("extra_cost_memory"), nb::arg("kv_factor"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("allocate_pools", &BaseKVCacheManager::allocatePools, nb::call_guard<nb::gil_scoped_release>())
        .def("release_pools", &BaseKVCacheManager::releasePools, nb::call_guard<nb::gil_scoped_release>())
        .def("start_scheduling", &BaseKVCacheManager::startScheduling, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("tokens_per_block", &BaseKVCacheManager::getTokensPerBlock)
        .def_prop_ro("max_num_blocks", &BaseKVCacheManager::getMaxNumBlocks)
        .def_prop_ro("num_pools", &BaseKVCacheManager::getNumPools)
        .def("get_kv_cache_stats", &BaseKVCacheManager::getKvCacheStats, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_blocks_per_seq",
            [](tbk::BaseKVCacheManager& self) { return self.getOffsetTableDimensions().maxBlocksPerSeq; })
        .def("get_needed_blocks_one_step", &BaseKVCacheManager::getNeededBlocksOneStep,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_remaining_blocks_to_completion", &BaseKVCacheManager::getRemainingBlocksToCompletion,
            nb::call_guard<nb::gil_scoped_release>())
        .def("add_token", &BaseKVCacheManager::addToken, nb::call_guard<nb::gil_scoped_release>())
        .def("add_sequence", &BaseKVCacheManager::addSequence, nb::call_guard<nb::gil_scoped_release>())
        .def("remove_sequence", &BaseKVCacheManager::removeSequence, nb::call_guard<nb::gil_scoped_release>())
        .def("pin_blocks", &BaseKVCacheManager::pinBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def("scheduling_remove_sequence", &BaseKVCacheManager::schedulingRemoveSequence,
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_primary_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getPrimaryPool(layer_idx));
                auto pool_layer_idx = self.getPoolLayerIdx(layer_idx);
                return pool.index({torch::indexing::Slice(), pool_layer_idx});
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_indexer_k_cache_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getIndexerKCachePool());
                return pool.index({torch::indexing::Slice(), layer_idx});
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_unique_primary_pool", [](tbk::BaseKVCacheManager& self) { return self.getUniquePrimaryPool(); },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_block_offsets_of_batch",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                SizeType32 beamWidth)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
            },
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::arg("timeout_ms") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("enable_block_reuse", &BaseKVCacheManager::isEnableBlockReuse)
        .def("rewind_kv_cache", &BaseKVCacheManager::rewindKVCache, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("cross_kv", &BaseKVCacheManager::isCrossKv)
        .def("store_context_blocks", &BaseKVCacheManager::storeContextBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def("store_blocks_for_reuse", &BaseKVCacheManager::storeBlocksForReuse,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_cache_block_ids", &BaseKVCacheManager::getCacheBlockIds, nb::call_guard<nb::gil_scoped_release>())
        .def("get_batch_cache_block_ids", &BaseKVCacheManager::getBatchCacheBlockIds,
            nb::call_guard<nb::gil_scoped_release>())
        .def("flush_iteration_events", &BaseKVCacheManager::flushIterationEvents,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_last_block_id", &BaseKVCacheManager::getLastBlockId, nb::call_guard<nb::gil_scoped_release>())
        .def("unpin_blocks_by_id", &BaseKVCacheManager::unpinBlocksById, nb::call_guard<nb::gil_scoped_release>());

    nb::bind_vector<CacheBlockIds>(m, "CacheBlockIds")
        .def("__getstate__", [](CacheBlockIds const& v) { return nb::make_tuple(v); })
        .def("__setstate__",
            [](CacheBlockIds& self, nb::tuple const& t)
            {
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                new (&self) CacheBlockIds(nb::cast<std::vector<std::vector<SizeType32>>>(t[0]));
            });

    nb::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS)
        .value("SELFKONLY", tbk::CacheType::kSELFKONLY);

    nb::class_<tbk::KVCacheManager, tbk::BaseKVCacheManager>(m, "KVCacheManager")
        .def(nb::init<std::vector<SizeType32> const&, SizeType32, SizeType32,
                 std::map<SizeType32, std::tuple<SizeType32, SizeType32>> const&, SizeType32, SizeType32,
                 std::vector<SizeType32> const&, std::optional<tbk::TempAttentionWindowInputs> const&,
                 nvinfer1::DataType, SizeType32, int64_t, runtime::SizeType32, bool, bool, tbk::CacheType,
                 std::optional<tensorrt_llm::executor::RetentionPriority>, std::shared_ptr<tbk::KVCacheEventManager>,
                 bool, bool, std::shared_ptr<tbc::KvCacheConnectorManager>, bool, SizeType32, SizeType32>(),
            nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"), nb::arg("tokens_per_block"),
            nb::arg("blocks_per_window"), nb::arg("max_num_sequences"), nb::arg("max_beam_width"),
            nb::arg("max_attention_window_vec"), nb::arg("temp_attention_window_inputs").none(), nb::arg("dtype"),
            nb::arg("sink_token_length"), nb::arg("stream"), nb::arg("max_sequence_length").none(),
            nb::arg("enable_block_reuse") = false, nb::arg("onboard_blocks") = true,
            nb::arg("cache_type") = tbk::CacheType::kSELF, nb::arg("secondary_offload_min_priority") = std::nullopt,
            nb::arg("event_manager") = nullptr, nb::arg("enable_partial_reuse") = true,
            nb::arg("copy_on_partial_reuse") = true, nb::arg("kv_connector_manager") = nullptr,
            nb::arg("enable_indexer_k_cache") = false, nb::arg("indexer_k_cache_quant_block_size") = 128,
            nb::arg("indexer_k_cache_index_head_dim") = 0,
            nb::call_guard<nb::gil_scoped_release>());
}

void tb::BasePeftCacheManagerBindings::initBindings(nb::module_& m)
{
    nb::class_<tb::BasePeftCacheManager, PyBasePeftCacheManager>(m, "BasePeftCacheManager")
        .def("add_request_peft", &tb::BasePeftCacheManager::addRequestPeft, nb::arg("request"),
            nb::arg("try_gpu_cache") = true, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "ensure_batch",
            [](tb::BasePeftCacheManager& self, tb::RequestVector const& contextRequests,
                tb::RequestVector const& generationRequests, bool resetGpuCache)
            { return self.ensureBatch(contextRequests, generationRequests, resetGpuCache); },
            nb::arg("context_requests"), nb::arg("generation_requests"), nb::arg("reset_gpu_cache") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "reset_device_cache", &tb::BasePeftCacheManager::resetDeviceCache, nb::call_guard<nb::gil_scoped_release>())
        .def("mark_request_done", &tb::BasePeftCacheManager::markRequestDone, nb::arg("request"),
            nb::arg("pause") = false, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_device_pages", &tb::BasePeftCacheManager::getMaxDevicePages)
        .def_prop_ro("max_host_pages", &tb::BasePeftCacheManager::getMaxHostPages)
        .def("determine_num_pages", &tb::BasePeftCacheManager::determineNumPages, nb::arg("request"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("enabled", &tb::BasePeftCacheManager::enabled);

    nb::class_<tb::PeftCacheManager, tb::BasePeftCacheManager>(m, "PeftCacheManager")
        .def(nb::init<tb::PeftCacheManagerConfig, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            nb::arg("config"), nb::arg("model_config"), nb::arg("world_config"), nb::arg("buffer_manager"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("is_task_cached", &tb::PeftCacheManager::isTaskCached, nb::arg("taskId"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tb::NoOpPeftCacheManager, tb::BasePeftCacheManager>(m, "NoOpPeftCacheManager")
        .def(nb::init<>(), nb::call_guard<nb::gil_scoped_release>());
}
