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

class PyBasePeftCacheManager : public tb::BasePeftCacheManager
{
public:
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

    py::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS);

    py::classh<tbk::KVCacheManager>(m, "KVCacheManager")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, SizeType32, bool, tbk::KVCacheManager::CudaStreamPtr, bool, bool, tbk::CacheType>(),
            py::arg("num_layers"), py::arg("num_kv_heads"), py::arg("size_per_head"), py::arg("tokens_per_block"),
            py::arg("blocks_in_primary_pool"), py::arg("blocks_in_secondary_pool"), py::arg("max_num_sequences"),
            py::arg("max_beam_width"), py::arg("max_attention_window"), py::arg("sink_token_length"),
            py::arg("use_one_more_block"), py::arg("stream_ptr"), py::arg("enable_block_reuse") = false,
            py::arg("onboard_blocks") = true, py::arg_v("cache_type", tbk::CacheType::kSELF, "CacheType.SELF"))
        .def_static("get_max_attention_window_upper_bound", &tbk::KVCacheManager::getMaxAttentionWindowUpperBound,
            py::arg("blocks_in_primary_pool"), py::arg("tokens_per_block"), py::arg("max_beam_width"),
            py::arg("sink_token_len"), py::arg("use_one_more_block"))
        .def_static("calculate_max_num_blocks", &tbk::KVCacheManager::calculateMaxNumBlocks, py::arg("config"),
            py::arg("dtype"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"))
        .def("allocate_pools", &tbk::KVCacheManager::allocatePools)
        .def("start_scheduling", &tbk::KVCacheManager::startScheduling)
        .def_property_readonly("tokens_per_block", &tbk::KVCacheManager::getTokensPerBlock)
        .def_property_readonly("max_num_blocks", &tbk::KVCacheManager::getMaxNumBlocks)
        .def("get_kv_cache_stats", &tbk::KVCacheManager::getKvCacheStats)
        .def_property_readonly("max_blocks_per_seq", &tbk::KVCacheManager::getMaxBlocksPerSeq)
        .def("get_needed_blocks_one_step", &tbk::KVCacheManager::getNeededBlocksOneStep)
        .def("add_token", &tbk::KVCacheManager::addToken)
        .def("add_sequence", &tbk::KVCacheManager::addSequence)
        .def("remove_sequence", &tbk::KVCacheManager::removeSequence)
        .def("scheduling_remove_sequence", &tbk::KVCacheManager::schedulingRemoveSequence)
        .def("get_block_pool_pointers",
            [](tbk::KVCacheManager& self)
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
        .def("get_block_offsets_of_batch",
            [](tbk::KVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                SizeType32 beamWidth)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
            })
        .def("copy_block_offsets",
            [](tbk::KVCacheManager& self, at::Tensor output, SizeType32 outputSlotOffset, SizeType32 seqSlotIdx)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                auto maxBlockCount = self.copyBlockOffsets(*(_output.value()), outputSlotOffset, seqSlotIdx);
                return maxBlockCount;
            })
        .def_property_readonly("enable_block_reuse", &tbk::KVCacheManager::isEnableBlockReuse)
        .def("rewind_kv_cache", &tbk::KVCacheManager::rewindKVCache)
        .def_property_readonly("cross_kv", &tbk::KVCacheManager::isCrossKv)
        .def("store_context_blocks", &tbk::KVCacheManager::storeContextBlocks);
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
