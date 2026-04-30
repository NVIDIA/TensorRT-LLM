/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/nanobind/batch_manager/kvCacheManagerV2.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <cassert>
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace tensorrt_llm::nanobind::batch_manager
{

// Helper: convert Python list of int|bytes to vector<TokenIdExt>.
// nanobind's variant caster can't auto-convert bytes → DigestToken.
static std::vector<kv::TokenIdExt> castTokenList(nb::list tokens)
{
    std::vector<kv::TokenIdExt> vec;
    vec.reserve(nb::len(tokens));
    for (auto item : tokens)
    {
        if (nb::isinstance<nb::bytes>(item))
        {
            auto b = nb::cast<nb::bytes>(item);
            assert(nb::len(b) == kv::kDIGEST_LEN);
            kv::Digest d;
            std::memcpy(d.data(), b.c_str(), kv::kDIGEST_LEN);
            vec.emplace_back(kv::DigestToken(d));
        }
        else
        {
            vec.emplace_back(nb::cast<kv::TokenId>(item));
        }
    }
    return vec;
}

void KvCacheManagerV2Bindings::initBindings(nb::module_& m)
{
    // ---- Exceptions --------------------------------------------------------
    static nb::object sOutOfMemoryError = nb::exception<kv::OutOfMemoryError>(m, "OutOfMemoryError");
    static nb::object sHostOOMError = nb::exception<kv::HostOOMError>(m, "HostOOMError");
    static nb::object sDiskOOMError = nb::exception<kv::DiskOOMError>(m, "DiskOOMError");
    static nb::object sCuOOMError = nb::exception<kv::CuOOMError>(m, "CuOOMError");
    static nb::object sLogicError = nb::exception<kv::LogicError>(m, "LogicError");
    static nb::object sResourceBusyError = nb::exception<kv::ResourceBusyError>(m, "ResourceBusyError");
    static nb::object sOutOfPagesError = nb::exception<kv::OutOfPagesError>(m, "OutOfPagesError");

    // ---- Enums -------------------------------------------------------------
    nb::enum_<kv::PageStatus>(m, "PageStatus")
        .value("LOCKED", kv::PageStatus::LOCKED)
        .value("HELD", kv::PageStatus::HELD)
        .value("DROPPABLE", kv::PageStatus::DROPPABLE)
        .export_values();

    nb::enum_<kv::CacheTier>(m, "CacheTier")
        .value("GPU_MEM", kv::CacheTier::GPU_MEM)
        .value("HOST_MEM", kv::CacheTier::HOST_MEM)
        .value("DISK", kv::CacheTier::DISK)
        .export_values();

    // ---- KvCache::CommitState enum (also accessible as _KVCache.CommitState)
    auto kvCacheCommitState = nb::enum_<kv::KvCache::CommitState>(m, "KvCacheCommitState")
                                  .value("ALLOWED", kv::KvCache::CommitState::ALLOWED)
                                  .value("VIRTUAL_STOP", kv::KvCache::CommitState::VIRTUAL_STOP)
                                  .value("USER_STOP", kv::KvCache::CommitState::USER_STOP);

    // ---- KvCache::Status enum (also accessible as _KVCache.Status) ---------
    auto kvCacheStatus = nb::enum_<kv::KvCache::Status>(m, "KvCacheStatus")
                             .value("ACTIVE", kv::KvCache::Status::ACTIVE)
                             .value("SUSPENDED", kv::KvCache::Status::SUSPENDED)
                             .value("CLOSED", kv::KvCache::Status::CLOSED);

    // ---- BufferId ----------------------------------------------------------
    nb::class_<kv::BufferId>(m, "BufferId")
        .def(nb::init<kv::LayerId, kv::DataRole>(), nb::arg("layer_id"), nb::arg("role"))
        .def_rw("layer_id", &kv::BufferId::layerId)
        .def_rw("role", &kv::BufferId::role)
        .def("__eq__", &kv::BufferId::operator==);

    // ---- Config structs ----------------------------------------------------
    // Helper: add __copy__ and __deepcopy__ for aggregate config types.
    // All config structs are simple aggregates — default copy construction works.
#define DEF_COPY(cls)                                                                                                  \
    .def("__copy__", [](cls const& self) { return cls(self); })                                                        \
        .def(                                                                                                          \
            "__deepcopy__", [](cls const& self, nb::dict) { return cls(self); }, nb::arg("memo"))

    nb::class_<kv::GpuCacheTierConfig>(m, "GpuCacheTierConfig")
        .def(nb::init<size_t>(), nb::arg("quota"))
        .def_rw("quota", &kv::GpuCacheTierConfig::quota) DEF_COPY(kv::GpuCacheTierConfig);

    nb::class_<kv::HostCacheTierConfig>(m, "HostCacheTierConfig")
        .def(nb::init<size_t>(), nb::arg("quota"))
        .def_rw("quota", &kv::HostCacheTierConfig::quota) DEF_COPY(kv::HostCacheTierConfig);

    nb::class_<kv::DiskCacheTierConfig>(m, "DiskCacheTierConfig")
        .def(nb::init<size_t, std::string>(), nb::arg("quota"), nb::arg("path"))
        .def_rw("quota", &kv::DiskCacheTierConfig::quota)
        .def_rw("path", &kv::DiskCacheTierConfig::path) DEF_COPY(kv::DiskCacheTierConfig);

    nb::class_<kv::BufferConfig>(m, "BufferConfig")
        .def(nb::init<kv::DataRole, size_t, std::optional<int>>(), nb::arg("role"), nb::arg("size"),
            nb::arg("tokens_per_block_override") = std::nullopt)
        .def_rw("role", &kv::BufferConfig::role)
        .def_rw("size", &kv::BufferConfig::size)
        .def_rw("tokens_per_block_override", &kv::BufferConfig::tokensPerBlockOverride) DEF_COPY(kv::BufferConfig);

    nb::class_<kv::AttentionLayerConfig>(m, "AttentionLayerConfig")
        .def(nb::init<kv::LayerId, std::vector<kv::BufferConfig>, std::optional<int>, std::optional<int>>(),
            nb::arg("layer_id"), nb::arg("buffers"), nb::arg("sliding_window_size") = std::nullopt,
            nb::arg("num_sink_tokens") = std::nullopt)
        .def_rw("layer_id", &kv::AttentionLayerConfig::layerId)
        .def_rw("buffers", &kv::AttentionLayerConfig::buffers)
        .def_rw("sliding_window_size", &kv::AttentionLayerConfig::slidingWindowSize)
        .def_rw("num_sink_tokens", &kv::AttentionLayerConfig::numSinkTokens)
        .def_prop_ro("window_size", &kv::AttentionLayerConfig::windowSize) DEF_COPY(kv::AttentionLayerConfig);

    nb::enum_<kv::LayerType>(m, "LayerType")
        .value("ATTENTION", kv::LayerType::ATTENTION)
        .value("SSM", kv::LayerType::SSM);

    nb::class_<kv::SsmLayerConfig>(m, "SsmLayerConfig")
        .def(nb::init<kv::LayerId, std::vector<kv::BufferConfig>>(), nb::arg("layer_id"), nb::arg("buffers"))
        .def_rw("layer_id", &kv::SsmLayerConfig::layerId)
        .def_rw("buffers", &kv::SsmLayerConfig::buffers) DEF_COPY(kv::SsmLayerConfig);

    nb::class_<kv::KVCacheDesc>(m, "KVCacheDesc")
        .def(nb::init<int, int>(), nb::arg("capacity"), nb::arg("history_length"))
        .def_rw("capacity", &kv::KVCacheDesc::capacity)
        .def_rw("history_length", &kv::KVCacheDesc::historyLength) DEF_COPY(kv::KVCacheDesc);

    nb::class_<kv::BatchDesc>(m, "BatchDesc")
        .def(
            "__init__",
            [](kv::BatchDesc* bd, std::vector<kv::KVCacheDesc> kvCaches, int systemPromptLength)
            {
                new (bd) kv::BatchDesc();
                bd->kvCaches = std::move(kvCaches);
                bd->systemPromptLength = systemPromptLength;
            },
            nb::arg("kv_caches"), nb::arg("system_prompt_length") = 0)
        .def_rw("kv_caches", &kv::BatchDesc::kvCaches)
        .def_rw("system_prompt_length", &kv::BatchDesc::systemPromptLength) DEF_COPY(kv::BatchDesc);

    nb::class_<kv::KVCacheManagerConfig>(m, "KVCacheManagerConfig")
        .def(
            "__init__",
            [](kv::KVCacheManagerConfig* cfg, int tokensPerBlock, int vocabSize,
                std::vector<kv::CacheTierConfig> cacheTiers, nb::list layers, float maxUtilForResume,
                bool enablePartialReuse, std::optional<kv::BatchDesc> typicalStep,
                std::vector<kv::BatchDesc> constraints, int ssmReuseInterval)
            {
                new (cfg) kv::KVCacheManagerConfig();
                cfg->tokensPerBlock = tokensPerBlock;
                cfg->vocabSize = vocabSize;
                cfg->cacheTiers = std::move(cacheTiers);
                // Convert Python list of AttentionLayerConfig|SsmLayerConfig to vector<LayerConfig>.
                for (auto item : layers)
                {
                    if (nb::isinstance<kv::SsmLayerConfig>(item))
                        cfg->layers.push_back(nb::cast<kv::SsmLayerConfig>(item));
                    else
                        cfg->layers.push_back(nb::cast<kv::AttentionLayerConfig>(item));
                }
                cfg->maxUtilForResume = maxUtilForResume;
                cfg->enablePartialReuse = enablePartialReuse;
                cfg->typicalStep = std::move(typicalStep);
                cfg->constraints = std::move(constraints);
                cfg->ssmReuseInterval = ssmReuseInterval;
            },
            nb::arg("tokens_per_block"), nb::arg("vocab_size"), nb::arg("cache_tiers"), nb::arg("layers"),
            nb::arg("max_util_for_resume") = 0.97f, nb::arg("enable_partial_reuse") = true,
            nb::arg("typical_step") = std::nullopt, nb::arg("constraints") = std::vector<kv::BatchDesc>{},
            nb::arg("ssm_reuse_interval") = 512)
        .def_rw("tokens_per_block", &kv::KVCacheManagerConfig::tokensPerBlock)
        .def_rw("vocab_size", &kv::KVCacheManagerConfig::vocabSize)
        .def_rw("cache_tiers", &kv::KVCacheManagerConfig::cacheTiers)
        .def_rw("layers", &kv::KVCacheManagerConfig::layers)
        .def_rw("max_util_for_resume", &kv::KVCacheManagerConfig::maxUtilForResume)
        .def_rw("enable_partial_reuse", &kv::KVCacheManagerConfig::enablePartialReuse)
        .def_rw("typical_step", &kv::KVCacheManagerConfig::typicalStep)
        .def_rw("constraints", &kv::KVCacheManagerConfig::constraints)
        .def_rw("ssm_reuse_interval", &kv::KVCacheManagerConfig::ssmReuseInterval)
        .def("validate", &kv::KVCacheManagerConfig::validate) DEF_COPY(kv::KVCacheManagerConfig);

#undef DEF_COPY

    // ---- KvCache -----------------------------------------------------------
    nb::class_<kv::KvCache>(m, "_KVCache")
        .def(
            "resume",
            [](kv::KvCache& self, nb::object stream)
            {
                std::optional<CUstream> optStream;
                if (!stream.is_none())
                    optStream = reinterpret_cast<CUstream>(nb::cast<intptr_t>(stream));
                nb::gil_scoped_release rel;
                return self.resume(optStream);
            },
            nb::arg("cuda_stream") = nb::none())
        .def("suspend", &kv::KvCache::suspend, nb::call_guard<nb::gil_scoped_release>())
        .def("close", &kv::KvCache::close, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "resize",
            [](kv::KvCache& self, std::optional<int> capacity, std::optional<int> historyLength) -> bool
            {
                try
                {
                    self.resize(capacity, historyLength);
                    return true;
                }
                catch (kv::OutOfPagesError const&)
                {
                    return false;
                }
            },
            nb::arg("capacity"), nb::arg("history_length") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "commit",
            [](kv::KvCache& self, nb::list tokens)
            {
                auto vec = castTokenList(tokens);
                nb::gil_scoped_release release;
                self.commit(vec);
            },
            nb::arg("tokens"))
        .def("stop_committing", &kv::KvCache::stopCommitting, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_base_page_indices",
            [](kv::KvCache const& self, kv::LayerGroupId lgId, kv::BeamIndex beamIdx)
            {
                kv::Span<int const> span;
                {
                    nb::gil_scoped_release release;
                    span = self.getBasePageIndices(lgId, beamIdx);
                }
                // Zero-copy: return a read-only numpy ndarray referencing the internal buffer.
                // nb::handle() = no owner; the returned array does not own the data.
                // TODO(yaoy): switch to nb::ndarray<nb::memview> when we have nanobind >= 2.9.0,
                // or nb::memoryview for nanobind >= 2.12.0.
                return nb::ndarray<nb::numpy, int const, nb::ndim<1>>(
                    span.ptr, {static_cast<size_t>(span.len)}, nb::handle());
            },
            nb::arg("layer_group_id"), nb::arg("beam_idx") = 0)
        .def("get_ssm_block_base_index", &kv::KvCache::getSsmBlockBaseIndex, nb::arg("layer_group_id"),
            nb::arg("beam_id") = 0, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("status", [](kv::KvCache const& kvc) { return kvc.status(); })
        .def_prop_ro("num_blocks", &kv::KvCache::numBlocks)
        .def_prop_ro("num_committed_blocks", &kv::KvCache::numCommittedBlocks)
        .def_prop_ro("num_committed_tokens", &kv::KvCache::numCommittedTokens)
        .def_prop_rw("history_length", &kv::KvCache::historyLength,
            [](kv::KvCache& self, int hist) { self.setHistoryLength(hist); })
        .def_prop_rw("capacity", &kv::KvCache::capacity, [](kv::KvCache& self, int cap) { self.setCapacity(cap); })
        .def_prop_ro("tokens_per_block", &kv::KvCache::tokensPerBlock)
        .def_prop_ro("beam_width", &kv::KvCache::beamWidth)
        .def_prop_rw(
            "cuda_stream",
            [](kv::KvCache const& self) -> intptr_t { return reinterpret_cast<intptr_t>(self.cudaStream()); },
            [](kv::KvCache& self, intptr_t stream) { self.setCudaStream(reinterpret_cast<CUstream>(stream)); })
        .def_prop_ro("_commit_state", &kv::KvCache::commitState)
        .def_rw("id", &kv::KvCache::id)
        .def_prop_ro(
            "manager", [](kv::KvCache& self) -> kv::KvCacheManager& { return self.manager(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("_committed_tokens",
            [](kv::KvCache const& self)
            {
                nb::list result;
                for (auto const& tok : self.committedTokens())
                {
                    if (auto* id = std::get_if<kv::TokenId>(&tok))
                        result.append(*id);
                    else
                    {
                        auto const& d = std::get<kv::DigestToken>(tok);
                        result.append(nb::bytes(reinterpret_cast<char const*>(d.data()), d.size()));
                    }
                }
                return result;
            })
        .def(
            "get_aggregated_page_indices",
            [](kv::KvCache const& self, kv::LayerGroupId lgId, kv::BeamIndex beamIdx, bool validOnly)
            { return self.getAggregatedPageIndices(lgId, beamIdx, validOnly); },
            nb::arg("layer_group_id"), nb::arg("beam_idx") = 0, nb::arg("valid_only") = false)
        .def(
            "set_base_page_index_buf",
            [](kv::KvCache& self, kv::BeamIndex beamIdx, kv::LayerGroupId lgId, nb::object bufObj)
            {
                if (bufObj.is_none())
                {
                    self.setBasePageIndexBuf(beamIdx, lgId, nullptr, 0);
                    return;
                }
                // Accept any object exporting a 1-D writable int32 ('i') buffer.
                // PyBUF_ND is required so that memoryview exports shape + format together;
                // without it, Python refuses to present a non-byte format as a flat buffer.
                Py_buffer view;
                if (PyObject_GetBuffer(bufObj.ptr(), &view, PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND) != 0)
                    throw nb::python_error();
                struct Cleanup
                {
                    Py_buffer* v;
                    ~Cleanup()
                    {
                        PyBuffer_Release(v);
                    }
                } cleanup{&view};
                if (std::string(view.format) != "i" || view.ndim != 1)
                    throw std::invalid_argument("set_base_page_index_buf: buffer must be 1-D int32 ('i')");
                self.setBasePageIndexBuf(
                    beamIdx, lgId, static_cast<int32_t*>(view.buf), static_cast<int>(view.len / sizeof(int32_t)));
            },
            nb::arg("beam_idx"), nb::arg("layer_group_id"), nb::arg("buf"));

    // Make Status and CommitState accessible as _KVCache.Status / _KVCache.CommitState
    m.attr("_KVCache").attr("Status") = kvCacheStatus;
    m.attr("_KVCache").attr("CommitState") = kvCacheCommitState;

    // ---- StorageStatistics ---------------------------------------------------
    nb::class_<kv::StorageStatistics>(m, "StorageStatistics")
        .def_ro("slot_sizes", &kv::StorageStatistics::slotSizes)
        .def_ro("total", &kv::StorageStatistics::total)
        .def_ro("free", &kv::StorageStatistics::free)
        .def_ro("evictable", &kv::StorageStatistics::evictable)
        .def_prop_ro("available", &kv::StorageStatistics::available)
        .def_prop_ro("unavailable", &kv::StorageStatistics::unavailable);

    // ---- StorageManager (partial — for test introspection) -----------------
    nb::class_<kv::StorageManager>(m, "StorageManager")
        .def(
            "get_statistics",
            [](kv::StorageManager const& self, kv::CacheLevel level)
            {
                std::vector<kv::StorageStatistics> result;
                int numPg = self.numPoolGroups();
                for (int pg = 0; pg < numPg; ++pg)
                    result.push_back(self.getStatistics(level, static_cast<kv::PoolGroupIndex>(pg)));
                return result;
            },
            nb::arg("level") = kv::kGpuLevel, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_ratio_list",
            [](kv::StorageManager const& self, kv::CacheLevel level) { return self.getRatioList(level); },
            nb::arg("level") = kv::kGpuLevel, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("num_pool_groups", [](kv::StorageManager const& self) { return self.numPoolGroups(); })
        .def_prop_ro("num_cache_levels", [](kv::StorageManager const& self) { return self.numCacheLevels(); });

    // ---- KvCacheManager ----------------------------------------------------
    nb::class_<kv::KvCacheManager>(m, "KVCacheManager")
        .def(nb::init<kv::KVCacheManagerConfig const&>(), nb::arg("config"), nb::call_guard<nb::gil_scoped_release>())
        .def("shutdown", &kv::KvCacheManager::shutdown, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "clear_reusable_blocks", &kv::KvCacheManager::clearReusableBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_kv_cache",
            [](kv::KvCacheManager& self, std::optional<int64_t> loraTaskId, nb::object inputTokens,
                std::optional<int64_t> id, kv::KvCache::PriorityCb priorityCb)
            {
                std::vector<kv::TokenIdExt> tokens;
                if (!inputTokens.is_none() && nb::isinstance<nb::list>(inputTokens))
                    tokens = castTokenList(nb::cast<nb::list>(inputTokens));
                nb::gil_scoped_release release;
                return self.createKvCache(loraTaskId, tokens, id, std::move(priorityCb));
            },
            nb::arg("lora_task_id") = std::nullopt, nb::arg("input_tokens") = nb::none(), nb::arg("id") = std::nullopt,
            nb::arg("priority_cb") = kv::KvCache::PriorityCb{})
        .def("get_mem_pool_base_address", &kv::KvCacheManager::getMemPoolBaseAddress, nb::arg("layer_id"),
            nb::arg("data_role"), nb::call_guard<nb::gil_scoped_release>())
        .def("get_page_stride", &kv::KvCacheManager::getPageStride, nb::arg("layer_id"), nb::arg("data_role"))
        .def("get_page_index_scale", &kv::KvCacheManager::getPageIndexScale, nb::arg("layer_id"), nb::arg("data_role"))
        .def("get_page_index_upper_bound", &kv::KvCacheManager::getPageIndexUpperBound, nb::arg("layer_id"),
            nb::arg("data_role"))
        .def("resize", &kv::KvCacheManager::resize, nb::arg("cache_level"), nb::arg("quota"),
            nb::arg("best_efforts") = false, nb::call_guard<nb::gil_scoped_release>())
        .def("get_quota", &kv::KvCacheManager::getQuota, nb::arg("cache_level"))
        .def_prop_ro("tokens_per_block", &kv::KvCacheManager::tokensPerBlock)
        .def_prop_ro("cache_tier_list", &kv::KvCacheManager::cacheTierList)
        .def_prop_ro("all_buffer_ids", &kv::KvCacheManager::allBufferIds)
        .def("clamp_max_seq_len_for_mem", &kv::KvCacheManager::clampMaxSeqLenForMem, nb::arg("batch_size"),
            nb::arg("token_num_upper_bound"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("enable_partial_match", &kv::KvCacheManager::enablePartialMatch)
        .def_prop_ro("ssm_reuse_interval", &kv::KvCacheManager::ssmReuseInterval)
        .def_prop_ro("num_layers", &kv::KvCacheManager::numLayers)
        .def_prop_ro("layer_ids", &kv::KvCacheManager::layerIds)
        .def_prop_ro("layer_grouping", &kv::KvCacheManager::layerGrouping)
        .def("get_layer_group_id", &kv::KvCacheManager::getLayerGroupId, nb::arg("layer_id"))
        .def("get_page_index_converter", &kv::KvCacheManager::getPageIndexConverter, nb::arg("layer_id"),
            nb::arg("data_role"))
        .def(
            "get_aggregated_pages",
            [](kv::KvCacheManager const& self, nb::object buffers)
            {
                // Convert Python iterable → C++ vector (GIL held).
                std::vector<kv::BufferId> ids;
                for (auto item : nb::cast<nb::iterable>(buffers))
                    ids.push_back(nb::cast<kv::BufferId>(item));
                // Release GIL for the pure C++ computation.
                nb::gil_scoped_release rel;
                return self.getAggregatedPages(ids);
            },
            nb::arg("buffers"))
        .def_prop_ro(
            "_radix_tree", [](kv::KvCacheManager& self) -> kv::BlockRadixTree& { return self.radixTree(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "_current_gpu_ratio", [](kv::KvCacheManager& self) { return self.storage().getRatioList(kv::kGpuLevel); })
        .def_prop_ro(
            "_storage", [](kv::KvCacheManager& self) -> kv::StorageManager& { return self.storage(); },
            nb::rv_policy::reference_internal);

    // ---- ExpandedBuffer / AggregatedPageDesc --------------------------------
    nb::class_<kv::ExpandedBuffer>(m, "ExpandedBuffer")
        .def_ro("id", &kv::ExpandedBuffer::id)
        .def_ro("expansion", &kv::ExpandedBuffer::expansion)
        .def("__eq__",
            [](kv::ExpandedBuffer const& a, kv::ExpandedBuffer const& b)
            { return a.id == b.id && a.expansion == b.expansion; });

    nb::class_<kv::AggregatedPageDesc>(m, "AggregatedPageDesc")
        .def_ro("base", &kv::AggregatedPageDesc::base)
        .def_ro("size", &kv::AggregatedPageDesc::size)
        .def_ro("stride", &kv::AggregatedPageDesc::stride)
        .def_ro("layer_group_id", &kv::AggregatedPageDesc::layerGroupId)
        .def_ro("buffers", &kv::AggregatedPageDesc::buffers);

    // ---- PageIndexConverter ------------------------------------------------
    nb::class_<kv::PageIndexConverter>(m, "PageIndexConverter")
        .def_ro("scale", &kv::PageIndexConverter::scale)
        .def_ro("expansion", &kv::PageIndexConverter::expansion)
        .def("__call__", &kv::PageIndexConverter::operator(), nb::arg("base_index"));

    // ---- Page (read-only introspection, used via block.storage) ---------------
    // status is returned as int so it compares equal with the Python IntEnum PageStatus.
    nb::class_<kv::Page>(m, "Page")
        .def_prop_ro("status", [](kv::Page const& self) { return static_cast<int>(self.status()); })
        .def_prop_ro("scheduled_for_eviction", &kv::Page::scheduledForEviction);

    // ---- Block (radix-tree node) --------------------------------------------
    nb::class_<kv::Block>(m, "Block")
        .def_prop_ro("next",
            [](kv::Block const& self)
            {
                nb::dict d;
                for (auto const& [key, sp] : self.next)
                    d[nb::bytes(reinterpret_cast<char const*>(key.data()), key.size())] = sp;
                return d;
            })
        .def_prop_ro("prev",
            [](kv::Block& self) -> nb::object
            {
                if (self.parentRoot)
                    return nb::cast(self.parentRoot, nb::rv_policy::reference);
                if (self.parentBlock)
                    return nb::cast(self.parentBlock->shared_from_this());
                return nb::none();
            })
        .def_prop_ro("storage",
            [](kv::Block const& self)
            {
                nb::list lst;
                for (auto const& wp : self.storage)
                {
                    auto sp = wp.lock();
                    if (!sp)
                        lst.append(nb::none());
                    else
                    {
                        // Return a callable: page() returns the Page object.
                        // This matches the rawref.ref[Page] protocol used in Python.
                        auto page = std::static_pointer_cast<kv::Page>(sp);
                        lst.append(nb::cpp_function([page]() { return page; }));
                    }
                }
                return lst;
            });

    // ---- RootBlock (radix-tree root node) -----------------------------------
    nb::class_<kv::RootBlock>(m, "RootBlock")
        .def_prop_ro("next",
            [](kv::RootBlock const& self)
            {
                nb::dict d;
                for (auto const& [key, sp] : self.next)
                    d[nb::bytes(reinterpret_cast<char const*>(key.data()), key.size())] = sp;
                return d;
            });

    // ---- BlockRadixTree (read-only introspection) ---------------------------
    nb::class_<kv::BlockRadixTree>(m, "BlockRadixTree")
        .def_prop_ro("tokens_per_block", &kv::BlockRadixTree::tokensPerBlock)
        .def_prop_ro("num_life_cycles", &kv::BlockRadixTree::numLifeCycles)
        .def_prop_ro("next",
            [](kv::BlockRadixTree const& self)
            {
                nb::dict d;
                for (auto const& [key, rb] : self.roots())
                {
                    auto* ptr = const_cast<kv::RootBlock*>(&rb);
                    d[nb::bytes(reinterpret_cast<char const*>(key.data()), key.size())]
                        = nb::cast(ptr, nb::rv_policy::reference);
                }
                return d;
            });
}

} // namespace tensorrt_llm::nanobind::batch_manager
