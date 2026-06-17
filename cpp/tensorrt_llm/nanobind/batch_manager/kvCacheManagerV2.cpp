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
#include "kv_cache_manager_v2/introspection.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;
namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace tensorrt_llm::nanobind::batch_manager
{

// Helper: convert a Python iterable of int|bytes to vector<TokenIdExt>.
// nanobind's variant caster can't auto-convert bytes → DigestToken.
static std::vector<kv::TokenIdExt> castTokenIterable(nb::handle tokens)
{
    std::vector<kv::TokenIdExt> vec;
    for (auto item : nb::cast<nb::iterable>(tokens))
    {
        if (nb::isinstance<nb::bytes>(item))
        {
            auto b = nb::cast<nb::bytes>(item);
            if (nb::len(b) != kv::kDIGEST_LEN)
            {
                throw std::invalid_argument("Token bytes must have length kDIGEST_LEN");
            }
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

static kv::TypedVec<kv::PoolIndex, size_t> typedPoolSizeList(std::vector<size_t> const& slotSizeList)
{
    return kv::TypedVec<kv::PoolIndex, size_t>{slotSizeList};
}

static kv::TypedVec<kv::PoolGroupIndex, kv::TypedVec<kv::PoolIndex, size_t>> typedSlotSizeLists(
    std::vector<std::vector<size_t>> const& slotSizeLists)
{
    kv::TypedVec<kv::PoolGroupIndex, kv::TypedVec<kv::PoolIndex, size_t>> result;
    for (auto const& slotSizeList : slotSizeLists)
    {
        result.push_back(typedPoolSizeList(slotSizeList));
    }
    return result;
}

static kv::TypedVec<kv::PoolGroupIndex, kv::SlotCount> typedSlotCounts(std::vector<kv::SlotCount> const& slotCounts)
{
    kv::TypedVec<kv::PoolGroupIndex, kv::SlotCount> result;
    for (kv::SlotCount slotCount : slotCounts)
    {
        if (slotCount < 0)
        {
            throw std::invalid_argument("slot counts must be non-negative");
        }
        result.push_back(slotCount);
    }
    return result;
}

static nb::object castLifeCycle(kv::LifeCycle const& lifeCycle)
{
    return std::visit([](auto const& concreteLifeCycle) { return nb::cast(concreteLifeCycle); }, lifeCycle);
}

static kv::KvCache::PriorityCb castPriorityCallback(kv::KvCacheManager const& manager, nb::object callback)
{
    if (callback.is_none())
    {
        return {};
    }
    if (!PyCallable_Check(callback.ptr()))
    {
        throw std::invalid_argument("custom_priority_callback must be callable");
    }

    kv::LifeCycleRegistry const* lifeCycles = &manager.lifeCycles();
    return [callback = std::move(callback), lifeCycles](kv::BlockOrdinal ordinal, kv::LifeCycleId lifeCycleId)
    {
        nb::gil_scoped_acquire acquire;
        return nb::cast<kv::Priority>(callback(ordinal.value(), castLifeCycle(lifeCycles->getLifeCycle(lifeCycleId))));
    };
}

static nb::tuple bufferIdTuple(kv::BufferId const& self)
{
    return nb::make_tuple(self.layerId, self.role);
}

static nb::object optionalIntToObject(std::optional<int64_t> value)
{
    if (!value.has_value())
    {
        return nb::none();
    }
    return nb::cast(*value);
}

static nb::tuple reuseScopeTuple(kv::ReuseScope const& self)
{
    return nb::make_tuple(optionalIntToObject(self.loraId), optionalIntToObject(self.salt));
}

static nb::list committedTokensList(kv::KvCache const& self)
{
    nb::list result;
    for (auto const& tok : self.committedTokens())
    {
        if (auto* id = std::get_if<kv::TokenId>(&tok))
        {
            result.append(*id);
        }
        else
        {
            auto const& d = std::get<kv::DigestToken>(tok);
            result.append(nb::bytes(reinterpret_cast<char const*>(d.data()), d.size()));
        }
    }
    return result;
}

static std::optional<int64_t> castOptionalIntAttr(nb::handle obj, char const* attrName)
{
    nb::object attr = nb::steal(PyObject_GetAttrString(obj.ptr(), attrName));
    if (!attr)
    {
        throw nb::python_error();
    }
    if (attr.is_none())
    {
        return std::nullopt;
    }
    return nb::cast<int64_t>(attr);
}

static kv::ReuseScope castReuseScope(nb::object reuseScope)
{
    if (reuseScope.is_none())
    {
        return {};
    }
    if (nb::isinstance<kv::ReuseScope>(reuseScope))
    {
        return nb::cast<kv::ReuseScope>(reuseScope);
    }
    // Backward-compatible bridge for old callers that still pass lora_task_id as the first argument.
    if (PyLong_Check(reuseScope.ptr()))
    {
        return {nb::cast<int64_t>(reuseScope), std::nullopt};
    }
    if (PyObject_HasAttrString(reuseScope.ptr(), "lora_id") && PyObject_HasAttrString(reuseScope.ptr(), "salt"))
    {
        return {castOptionalIntAttr(reuseScope, "lora_id"), castOptionalIntAttr(reuseScope, "salt")};
    }
    throw std::invalid_argument(
        "reuse_scope must be None, ReuseScope, an int lora_task_id, or an object with lora_id and salt");
}

void KvCacheManagerV2Bindings::initBindings(nb::module_& m)
{
    // Export the C++ debug mode as an immutable Python bool snapshot.
    m.attr("NDEBUG") = nb::bool_(!kv::gDebug);

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

    // ---- KvCache::Status enum (also accessible as _KVCache.Status) ---------
    auto kvCacheStatus = nb::enum_<kv::KvCache::Status>(m, "KvCacheStatus")
                             .value("ACTIVE", kv::KvCache::Status::ACTIVE)
                             .value("SUSPENDED", kv::KvCache::Status::SUSPENDED)
                             .value("CLOSED", kv::KvCache::Status::CLOSED);

    // ---- Life cycle helpers ------------------------------------------------
    using BlockRange = kv::HalfOpenRange<kv::BlockOrdinal>;
    nb::class_<BlockRange>(m, "HalfOpenRange")
        .def(nb::init<int, int>(), nb::arg("beg"), nb::arg("end"))
        .def_prop_ro("beg", [](BlockRange const& self) { return self.beg.value(); })
        .def_prop_ro("end", [](BlockRange const& self) { return self.end.value(); })
        .def("__bool__", [](BlockRange const& self) { return static_cast<bool>(self); })
        .def("__len__", &BlockRange::length)
        .def("__eq__", [](BlockRange const& self, BlockRange const& other) { return self == other; })
        .def(
            "__contains__", [](BlockRange const& self, int item) { return self.contains(kv::BlockOrdinal{item}); },
            nb::arg("item"));

    nb::enum_<kv::PageIndexMode>(m, "PageIndexMode")
        .value("SHARED", kv::PageIndexMode::SHARED)
        .value("PER_LAYER", kv::PageIndexMode::PER_LAYER);

    nb::class_<kv::ScratchDesc>(m, "ScratchDesc")
        .def_ro("range", &kv::ScratchDesc::range)
        .def_ro("slot_ids", &kv::ScratchDesc::slotIds)
        .def("__bool__", [](kv::ScratchDesc const& self) { return static_cast<bool>(self); });

    nb::class_<kv::AttnLifeCycle>(m, "AttnLifeCycle")
        .def(nb::init<std::optional<int>, int>(), nb::arg("window_size"), nb::arg("num_sink_blocks"))
        .def_prop_ro("window_size", [](kv::AttnLifeCycle const& self) { return self.windowSize; })
        .def_ro("num_sink_blocks", &kv::AttnLifeCycle::numSinkBlocks)
        .def("get_stale_range", &kv::AttnLifeCycle::getStaleRange, nb::arg("history_length"),
            nb::arg("tokens_per_block"))
        .def("__eq__", &kv::AttnLifeCycle::operator==);

    nb::class_<kv::SsmLifeCycle>(m, "SsmLifeCycle")
        .def(nb::init<>())
        .def(
            "get_stale_range", &kv::SsmLifeCycle::getStaleRange, nb::arg("history_length"), nb::arg("tokens_per_block"))
        .def("__eq__", &kv::SsmLifeCycle::operator==);

    // ---- CUDA event --------------------------------------------------------
    auto cachedCudaEvent
        = nb::class_<kv::CachedCudaEvent>(m, "CachedCudaEvent")
              .def(nb::init<kv::CudaStream>(), nb::arg("stream"))
              .def("query_complete", &kv::CachedCudaEvent::queryComplete, nb::call_guard<nb::gil_scoped_release>())
              .def("synchronize", &kv::CachedCudaEvent::synchronize, nb::call_guard<nb::gil_scoped_release>())
              .def(
                  "wait_in_stream",
                  [](kv::CachedCudaEvent const& self, kv::CudaStream stream) { self.waitInStream(stream); },
                  nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>())
              .def("close", &kv::CachedCudaEvent::close)
              .def("is_closed", &kv::CachedCudaEvent::isClosed);
    cachedCudaEvent.attr("NULL") = kv::CachedCudaEvent::makeNull();

    // ---- ReuseScope --------------------------------------------------------
    nb::class_<kv::ReuseScope>(m, "ReuseScope")
        .def(nb::init<std::optional<int64_t>, std::optional<int64_t>>(), nb::arg("lora_id").none() = std::nullopt,
            nb::arg("salt").none() = std::nullopt)
        .def_ro("lora_id", &kv::ReuseScope::loraId)
        .def_ro("salt", &kv::ReuseScope::salt)
        .def("to_bytes",
            [](kv::ReuseScope const& self)
            {
                auto bytes = self.toBytes();
                return nb::bytes(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            })
        .def("__len__", [](kv::ReuseScope const&) { return 2; })
        .def(
            "__getitem__",
            [](kv::ReuseScope const& self, int index) -> nb::object
            {
                if (index < 0)
                {
                    index += 2;
                }
                if (index == 0)
                {
                    return optionalIntToObject(self.loraId);
                }
                if (index == 1)
                {
                    return optionalIntToObject(self.salt);
                }
                throw nb::index_error("ReuseScope index out of range");
            },
            nb::arg("index"))
        .def("__iter__",
            [](kv::ReuseScope const& self)
            { return nb::steal<nb::iterator>(PyObject_GetIter(reuseScopeTuple(self).ptr())); })
        .def("__hash__", [](kv::ReuseScope const& self) { return PyObject_Hash(reuseScopeTuple(self).ptr()); })
        .def(
            "__eq__",
            [](kv::ReuseScope const& self, nb::object other) -> nb::object
            {
                if (nb::isinstance<kv::ReuseScope>(other))
                {
                    return nb::bool_(self == nb::cast<kv::ReuseScope>(other));
                }
                if (PyTuple_Check(other.ptr()) && PyTuple_GET_SIZE(other.ptr()) == 2)
                {
                    return nb::bool_(PyObject_RichCompareBool(reuseScopeTuple(self).ptr(), other.ptr(), Py_EQ) == 1);
                }
                return nb::not_implemented();
            },
            nb::arg("other"))
        .def("__repr__",
            [](kv::ReuseScope const& self)
            {
                std::string repr = "ReuseScope(lora_id=";
                repr += self.loraId.has_value() ? std::to_string(*self.loraId) : "None";
                repr += ", salt=";
                repr += self.salt.has_value() ? std::to_string(*self.salt) : "None";
                repr += ")";
                return repr;
            });

    // ---- BufferId ----------------------------------------------------------
    nb::class_<kv::BufferId>(m, "BufferId")
        .def(nb::init<kv::LayerId, kv::DataRole>(), nb::arg("layer_id"), nb::arg("role"))
        .def_ro("layer_id", &kv::BufferId::layerId)
        .def_ro("role", &kv::BufferId::role)
        .def("__len__", [](kv::BufferId const&) { return 2; })
        .def(
            "__getitem__",
            [](kv::BufferId const& self, int index) -> nb::object
            {
                if (index < 0)
                {
                    index += 2;
                }
                if (index == 0)
                {
                    return nb::cast(self.layerId);
                }
                if (index == 1)
                {
                    return nb::cast(self.role);
                }
                throw nb::index_error("BufferId index out of range");
            },
            nb::arg("index"))
        .def("__iter__",
            [](kv::BufferId const& self)
            { return nb::steal<nb::iterator>(PyObject_GetIter(bufferIdTuple(self).ptr())); })
        .def("__hash__", [](kv::BufferId const& self) { return PyObject_Hash(bufferIdTuple(self).ptr()); })
        .def(
            "__eq__",
            [](kv::BufferId const& self, nb::object other) -> nb::object
            {
                if (nb::isinstance<kv::BufferId>(other))
                {
                    return nb::bool_(self == nb::cast<kv::BufferId>(other));
                }
                if (PyTuple_Check(other.ptr()) && PyTuple_GET_SIZE(other.ptr()) == 2)
                {
                    return nb::bool_(PyObject_RichCompareBool(bufferIdTuple(self).ptr(), other.ptr(), Py_EQ) == 1);
                }
                return nb::not_implemented();
            },
            nb::arg("other"));

    // ---- Storage layout structs -------------------------------------------
    nb::class_<kv::CoalescedBuffer>(m, "CoalescedBuffer")
        .def_ro("single_buffer_size", &kv::CoalescedBuffer::singleBufferSize)
        .def_ro("buffer_ids", &kv::CoalescedBuffer::bufferIds)
        .def_prop_ro("size", &kv::CoalescedBuffer::size)
        .def_prop_ro("num_buffers", &kv::CoalescedBuffer::numBuffers);

    nb::class_<kv::SlotDescVariant>(m, "SlotDescVariant")
        .def_prop_ro("layer_group_id", [](kv::SlotDescVariant const& self) { return self.lifeCycleId.value(); })
        .def_prop_ro("coalesced_buffers", [](kv::SlotDescVariant const& self) { return self.coalescedBuffers.raw(); })
        .def_prop_ro("slot_size_list",
            [](kv::SlotDescVariant const& self)
            {
                auto sizes = self.slotSizeList();
                return std::vector<int>(sizes.begin(), sizes.end());
            });

    nb::class_<kv::SlotDesc>(m, "SlotDesc")
        .def_ro("variants", &kv::SlotDesc::variants)
        .def_prop_ro("slot_size_list",
            [](kv::SlotDesc const& self)
            {
                auto sizes = self.slotSizeList();
                return std::vector<int>(sizes.begin(), sizes.end());
            });

    nb::class_<kv::PoolDesc>(m, "PoolDesc")
        .def_prop_ro("pool_index", [](kv::PoolDesc const& self) { return self.poolIndex.value(); })
        .def_ro("base_address", &kv::PoolDesc::baseAddress)
        .def_ro("slot_bytes", &kv::PoolDesc::slotBytes);

    nb::class_<kv::PoolGroupDesc>(m, "PoolGroupDesc")
        .def_prop_ro("pool_group_index", [](kv::PoolGroupDesc const& self) { return self.poolGroupIndex.value(); })
        .def_ro("num_slots", &kv::PoolGroupDesc::numSlots)
        .def_ro("slot_desc", &kv::PoolGroupDesc::slotDesc)
        .def_prop_ro("pools", [](kv::PoolGroupDesc const& self) { return self.pools.raw(); });

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
        .def_prop_ro("layer_group_id", [](kv::AggregatedPageDesc const& self) { return self.layerGroupId.value(); })
        .def_ro("buffers", &kv::AggregatedPageDesc::buffers);

    // ---- Config structs ----------------------------------------------------
    // Helper: add __copy__ and __deepcopy__ for aggregate config types.
    // All config structs are simple aggregates — default copy construction works.
#define DEF_COPY(cls)                                                                                                  \
    .def("__copy__", [](cls const& self) { return cls(self); })                                                        \
        .def(                                                                                                          \
            "__deepcopy__", [](cls const& self, nb::dict) { return cls(self); }, nb::arg("memo"))

    nb::class_<kv::GpuCacheTierConfig>(m, "GpuCacheTierConfig")
        .def(nb::init<size_t>(), nb::arg("quota"))
        .def_rw("quota", &kv::GpuCacheTierConfig::quota)
        .def_prop_ro("tier", &kv::GpuCacheTierConfig::tier)
        .def("assert_valid", &kv::GpuCacheTierConfig::assertValid) DEF_COPY(kv::GpuCacheTierConfig);

    nb::class_<kv::HostCacheTierConfig>(m, "HostCacheTierConfig")
        .def(nb::init<size_t>(), nb::arg("quota"))
        .def_rw("quota", &kv::HostCacheTierConfig::quota)
        .def_prop_ro("tier", &kv::HostCacheTierConfig::tier)
        .def("assert_valid", &kv::HostCacheTierConfig::assertValid) DEF_COPY(kv::HostCacheTierConfig);

    nb::class_<kv::DiskCacheTierConfig>(m, "DiskCacheTierConfig")
        .def(nb::init<size_t, std::string>(), nb::arg("quota"), nb::arg("path"))
        .def_rw("quota", &kv::DiskCacheTierConfig::quota)
        .def_rw("path", &kv::DiskCacheTierConfig::path)
        .def_prop_ro("tier", &kv::DiskCacheTierConfig::tier)
        .def("assert_valid", &kv::DiskCacheTierConfig::assertValid) DEF_COPY(kv::DiskCacheTierConfig);

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

    nb::class_<kv::HelixConfig>(m, "HelixConfig")
        .def(nb::init<int, int, int, int>(), nb::arg("helix_group_size"), nb::arg("helix_gpu_rank"),
            nb::arg("helix_shard_size"), nb::arg("shared_comm_port"))
        .def_rw("helix_group_size", &kv::HelixConfig::helixGroupSize)
        .def_rw("helix_gpu_rank", &kv::HelixConfig::helixGpuRank)
        .def_rw("helix_shard_size", &kv::HelixConfig::helixShardSize)
        .def_rw("shared_comm_port", &kv::HelixConfig::sharedCommPort) DEF_COPY(kv::HelixConfig);

    nb::class_<kv::SwaScratchReuseConfig>(m, "SwaScratchReuseConfig")
        .def(
            "__init__",
            [](kv::SwaScratchReuseConfig* cfg, int maxRewindLen)
            {
                new (cfg) kv::SwaScratchReuseConfig();
                cfg->maxRewindLen = maxRewindLen;
                cfg->validate();
            },
            nb::arg("max_rewind_len") = 0)
        .def_rw("max_rewind_len", &kv::SwaScratchReuseConfig::maxRewindLen) DEF_COPY(kv::SwaScratchReuseConfig);

    nb::class_<kv::KVCacheManagerConfig>(m, "KVCacheManagerConfig")
        .def(
            "__init__",
            [](kv::KVCacheManagerConfig* cfg, int tokensPerBlock, std::vector<kv::CacheTierConfig> cacheTiers,
                nb::list layers, float maxUtilForResume, bool enablePartialReuse,
                std::optional<kv::BatchDesc> typicalStep, std::vector<kv::BatchDesc> constraints, int ssmReuseInterval,
                std::optional<kv::SwaScratchReuseConfig> swaScratchReuse, std::optional<kv::HelixConfig> helixConfig)
            {
                new (cfg) kv::KVCacheManagerConfig();
                cfg->tokensPerBlock = tokensPerBlock;
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
                cfg->swaScratchReuse = std::move(swaScratchReuse);
                cfg->helixConfig = std::move(helixConfig);
            },
            nb::arg("tokens_per_block"), nb::arg("cache_tiers"), nb::arg("layers"),
            nb::arg("max_util_for_resume") = 0.97f, nb::arg("enable_partial_reuse") = true,
            nb::arg("typical_step") = std::nullopt, nb::arg("constraints") = std::vector<kv::BatchDesc>{},
            nb::arg("ssm_reuse_interval") = 512, nb::arg("swa_scratch_reuse").none() = std::nullopt,
            nb::arg("helix_config").none() = std::nullopt)
        .def_rw("tokens_per_block", &kv::KVCacheManagerConfig::tokensPerBlock)
        .def_rw("cache_tiers", &kv::KVCacheManagerConfig::cacheTiers)
        .def_rw("layers", &kv::KVCacheManagerConfig::layers)
        .def_rw("max_util_for_resume", &kv::KVCacheManagerConfig::maxUtilForResume)
        .def_rw("enable_partial_reuse", &kv::KVCacheManagerConfig::enablePartialReuse)
        .def_rw("typical_step", &kv::KVCacheManagerConfig::typicalStep)
        .def_rw("constraints", &kv::KVCacheManagerConfig::constraints)
        .def_rw("ssm_reuse_interval", &kv::KVCacheManagerConfig::ssmReuseInterval)
        .def_rw("swa_scratch_reuse", &kv::KVCacheManagerConfig::swaScratchReuse)
        .def_prop_ro("enable_swa_scratch_reuse", &kv::KVCacheManagerConfig::enableSwaScratchReuse)
        .def_prop_rw(
            "helix_config",
            [](kv::KVCacheManagerConfig const& self) -> nb::object
            {
                if (!self.helixConfig.has_value())
                {
                    return nb::none();
                }
                return nb::cast(*self.helixConfig);
            },
            [](kv::KVCacheManagerConfig& self, nb::handle value)
            {
                if (value.is_none())
                {
                    self.helixConfig.reset();
                    return;
                }
                self.helixConfig = nb::cast<kv::HelixConfig>(value);
            },
            nb::arg("helix_config").none())
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
        .def(
            "prefetch", [](kv::KvCache& self, int target) { return self.prefetch(kv::CacheLevel{target}); },
            nb::arg("target"), nb::call_guard<nb::gil_scoped_release>())
        .def("close", &kv::KvCache::close, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "resize",
            [](kv::KvCache& self, std::optional<int> capacity, std::optional<int> historyLength) -> bool
            { return self.resize(capacity, historyLength); },
            nb::arg("capacity") = std::nullopt, nb::arg("history_length") = std::nullopt,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "commit",
            [](kv::KvCache& self, nb::object acceptedInputTokens, nb::object beamSearchIndices)
            {
                auto vec = castTokenIterable(acceptedInputTokens);
                if (vec.empty())
                {
                    return;
                }
                if (!beamSearchIndices.is_none())
                {
                    PyErr_SetString(PyExc_AssertionError, "beam_search_indices must be None");
                    throw nb::python_error();
                }
                nb::gil_scoped_release release;
                self.commit(vec);
            },
            nb::arg("accepted_input_tokens"), nb::arg("beam_search_indices").none() = nb::none())
        .def("stop_committing", &kv::KvCache::stopCommitting, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_base_page_indices",
            [](kv::KvCache const& self, int layerGroupId, int beamIdx)
            {
                kv::Span<int const> span;
                {
                    nb::gil_scoped_release release;
                    span = self.getBasePageIndices(kv::LayerGroupId{layerGroupId}, kv::BeamIndex{beamIdx});
                }
                // Zero-copy: return a read-only numpy ndarray referencing the internal buffer.
                // nb::handle() = no owner; the returned array does not own the data.
                // Contract: callers must keep the KvCache alive and must not mutate/resize it
                // while using this view. The view is intended for read-only use, matching the
                // practical use of Python's array.array/memoryview index buffers.
                // TODO(yaoy): switch to nb::ndarray<nb::memview> when we have nanobind >= 2.9.0,
                // or nb::memoryview for nanobind >= 2.12.0.
                return nb::ndarray<nb::numpy, int const, nb::ndim<1>>(
                    span.ptr, {static_cast<size_t>(span.len)}, nb::handle());
            },
            nb::arg("layer_group_id"), nb::arg("beam_idx") = 0)
        .def(
            "get_ssm_block_base_index",
            [](kv::KvCache const& self, int layerGroupId, int beamId)
            { return self.getSsmBlockBaseIndex(kv::LayerGroupId{layerGroupId}, kv::BeamIndex{beamId}); },
            nb::arg("layer_group_id"), nb::arg("beam_id") = 0, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_scratch_desc",
            [](kv::KvCache const& self, int layerGroupId)
            { return self.getScratchDesc(kv::LayerGroupId{layerGroupId}); },
            nb::arg("layer_group_id"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("has_scratch_slots", &kv::KvCache::hasScratchSlots)
        .def_prop_rw("enable_swa_scratch_reuse", &kv::KvCache::isSwaScratchReuseEnabled,
            [](kv::KvCache& self, bool enable) { self.setEnableSwaScratchReuse(enable); })
        .def("supports_index_mode", &kv::KvCache::supportsIndexMode, nb::arg("mode"))
        .def_prop_ro("status", [](kv::KvCache const& kvc) { return kvc.status(); })
        .def_prop_ro("is_active", &kv::KvCache::isActive)
        .def_prop_ro("finish_event",
            [](kv::KvCache const& self)
            {
                try
                {
                    return self.finishEvent();
                }
                catch (std::bad_optional_access const&)
                {
                    // Python unwrap_optional(None) raises ValueError with this message.
                    PyErr_SetString(PyExc_ValueError, "Expected non-None value");
                    throw nb::python_error();
                }
            })
        .def_prop_ro("num_blocks", [](kv::KvCache const& self) { return self.numBlocks().value(); })
        .def_prop_ro("num_committed_blocks", &kv::KvCache::numCommittedBlocks)
        .def_prop_ro("num_committed_tokens", &kv::KvCache::numCommittedTokens)
        .def_prop_rw("history_length", &kv::KvCache::historyLength,
            [](kv::KvCache& self, int hist) { self.setHistoryLength(hist); })
        .def_prop_rw("capacity", &kv::KvCache::capacity, [](kv::KvCache& self, int cap) { self.setCapacity(cap); })
        .def_prop_ro("tokens_per_block", &kv::KvCache::tokensPerBlock)
        .def_prop_ro("beam_width", [](kv::KvCache const& self) { return self.beamWidth().value(); })
        .def_prop_rw(
            "cuda_stream",
            [](kv::KvCache const& self) -> intptr_t { return reinterpret_cast<intptr_t>(self.cudaStream()); },
            [](kv::KvCache& self, intptr_t stream) { self.setCudaStream(reinterpret_cast<CUstream>(stream)); })
        .def_rw("id", &kv::KvCache::id)
        .def_prop_ro(
            "manager", [](kv::KvCache& self) -> kv::KvCacheManager& { return self.manager(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("committed_tokens", &committedTokensList)
        .def_prop_ro("reuse_scope", &kv::KvCache::reuseScope)
        .def(
            "get_aggregated_page_indices",
            [](kv::KvCache const& self, int layerGroupId, int beamIdx, bool validOnly) {
                return self.getAggregatedPageIndices(kv::LayerGroupId{layerGroupId}, kv::BeamIndex{beamIdx}, validOnly);
            },
            nb::arg("layer_group_id"), nb::arg("beam_idx") = 0, nb::arg("valid_only") = false)
        .def(
            "set_base_page_index_buf",
            [](kv::KvCache& self, int beamIdx, int layerGroupId, nb::object bufObj)
            {
                kv::BeamIndex const typedBeamIdx{beamIdx};
                kv::LayerGroupId const typedLayerGroupId{layerGroupId};
                if (bufObj.is_none())
                {
                    self.setBasePageIndexBuf(typedBeamIdx, typedLayerGroupId, nullptr, 0);
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
                self.setBasePageIndexBuf(typedBeamIdx, typedLayerGroupId, static_cast<int32_t*>(view.buf),
                    static_cast<int>(view.len / sizeof(int32_t)));
            },
            nb::arg("beam_idx"), nb::arg("layer_group_id"), nb::arg("buf"));

    // Make Status accessible as _KVCache.Status.
    m.attr("_KVCache").attr("Status") = kvCacheStatus;

    // ---- Introspection -------------------------------------------------------
    auto mIntrospection = m.def_submodule("_introspection", "KV cache manager v2 introspection helpers");
    nb::class_<kv::StorageStatistics>(mIntrospection, "StorageStatistics")
        .def_prop_ro("slot_sizes", [](kv::StorageStatistics const& self) { return self.slotSizes.raw(); })
        .def_ro("total", &kv::StorageStatistics::total)
        .def_ro("free", &kv::StorageStatistics::free)
        .def_ro("evictable", &kv::StorageStatistics::evictable)
        .def_prop_ro("available", &kv::StorageStatistics::available)
        .def_prop_ro("unavailable", &kv::StorageStatistics::unavailable);
    mIntrospection.def(
        "active_page_stats",
        [](kv::KvCache const& kvCache)
        {
            auto [counts, unscheduledEvictable] = kv::KvCacheIntrospection::activePageStats(kvCache);
            return std::make_tuple(std::move(counts.raw()), std::move(unscheduledEvictable.raw()));
        },
        nb::arg("kv_cache"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def("all_tree_pages_droppable", &kv::KvCacheIntrospection::allTreePagesDroppable, nb::arg("manager"),
        nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "is_commit_allowed",
        [](kv::KvCache const& kvCache) { return kvCache.commitState() == kv::KvCache::CommitState::ALLOWED; },
        nb::arg("kv_cache"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "current_gpu_ratio",
        [](kv::KvCacheManager& manager)
        {
            auto ratio = manager.storage().getRatioList(kv::kGpuLevel);
            return std::move(ratio.raw());
        },
        nb::arg("manager"), nb::call_guard<nb::gil_scoped_release>());
    // White-box test hooks: mutate auto-tuner state so accuracy tests can force
    // a pool rebalance. Mirror the Python manager's internal attributes.
    mIntrospection.def(
        "set_num_sampled_kv_caches",
        [](kv::KvCacheManager& manager, int value) { kv::KvCacheIntrospection::setNumSampledKvCaches(manager, value); },
        nb::arg("manager"), nb::arg("value"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "set_last_adjustment_time",
        [](kv::KvCacheManager& manager, double value)
        { kv::KvCacheIntrospection::setLastAdjustmentTime(manager, value); },
        nb::arg("manager"), nb::arg("value"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "set_target_ratio_list_gpu",
        [](kv::KvCacheManager& manager, std::vector<float> ratios)
        {
            kv::KvCacheIntrospection::setTargetRatioListGpu(
                manager, kv::TypedVec<kv::PoolGroupIndex, float>{std::move(ratios)});
        },
        nb::arg("manager"), nb::arg("ratios"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "storage_statistics",
        [](kv::KvCacheManager& manager, int cacheLevel)
        {
            auto stats = kv::KvCacheIntrospection::storageStatistics(manager, kv::CacheLevel{cacheLevel});
            return std::move(stats.raw());
        },
        nb::arg("manager"), nb::arg("cache_level") = kv::kGpuLevel.value(), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "storage_utilization",
        [](kv::KvCacheManager& manager, int cacheLevel)
        {
            auto utilization = manager.storage().getUtilization(kv::CacheLevel{cacheLevel});
            return std::move(utilization.raw());
        },
        nb::arg("manager"), nb::arg("cache_level") = kv::kGpuLevel.value(), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "grains_for_slots",
        [](kv::SlotCount numSlots, std::vector<size_t> const& slotSizeList, size_t granularity)
        {
            if (numSlots < 0)
            {
                throw std::invalid_argument("num_slots must be non-negative");
            }
            return kv::CacheLevelStorage::grainsForSlots(numSlots, typedPoolSizeList(slotSizeList), granularity);
        },
        nb::arg("num_slots"), nb::arg("slot_size_list"), nb::arg("granularity"),
        nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "grains_to_slots",
        [](size_t pgGrains, std::vector<size_t> const& slotSizeList, size_t granularity)
        {
            auto [slots, used]
                = kv::CacheLevelStorage::grainsToSlots(pgGrains, typedPoolSizeList(slotSizeList), granularity);
            return std::make_tuple(slots, used);
        },
        nb::arg("pg_grains"), nb::arg("slot_size_list"), nb::arg("granularity"),
        nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "ratio_to_slot_count_list",
        [](size_t quota, std::vector<std::vector<size_t>> const& slotSizeLists, std::vector<float> const& ratioList,
            size_t granularity, std::vector<kv::SlotCount> const& minSlots)
        {
            auto slotCountList = kv::CacheLevelStorage::ratioToSlotCountList(quota, typedSlotSizeLists(slotSizeLists),
                kv::TypedVec<kv::PoolGroupIndex, float>{ratioList}, granularity, typedSlotCounts(minSlots));
            std::vector<kv::SlotCount> result;
            result.reserve(slotCountList.stdSize());
            for (kv::SlotCount slotCount : slotCountList)
            {
                result.push_back(slotCount);
            }
            return result;
        },
        nb::arg("quota"), nb::arg("slot_size_lists"), nb::arg("ratio_list"), nb::arg("granularity"),
        nb::arg("min_slots"), nb::call_guard<nb::gil_scoped_release>());

    // ---- KvCacheManager ----------------------------------------------------
    nb::class_<kv::KvCacheManager>(m, "KVCacheManager")
        .def(nb::init<kv::KVCacheManagerConfig const&>(), nb::arg("config"), nb::call_guard<nb::gil_scoped_release>())
        .def("shutdown", &kv::KvCacheManager::shutdown, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "clear_reusable_blocks", &kv::KvCacheManager::clearReusableBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_kv_cache",
            [](std::shared_ptr<kv::KvCacheManager> self, nb::object reuseScopeObj, nb::object inputTokens,
                std::optional<int64_t> id, nb::object customPriorityCallback)
            {
                kv::ReuseScope reuseScope = castReuseScope(std::move(reuseScopeObj));
                std::vector<kv::TokenIdExt> tokens;
                if (!inputTokens.is_none())
                {
                    tokens = castTokenIterable(inputTokens);
                }
                kv::KvCache::PriorityCb priorityCb = castPriorityCallback(*self, std::move(customPriorityCallback));
                nb::gil_scoped_release release;
                return self->createKvCache(std::move(reuseScope), tokens, id, std::move(priorityCb));
            },
            nb::arg("reuse_scope") = nb::none(), nb::arg("input_tokens") = nb::none(), nb::arg("id") = std::nullopt,
            nb::arg("custom_priority_callback") = nb::none())
        .def(
            "probe_reuse",
            [](std::shared_ptr<kv::KvCacheManager> self, nb::object reuseScopeObj, nb::object inputTokens)
            {
                kv::ReuseScope reuseScope = castReuseScope(std::move(reuseScopeObj));
                std::vector<kv::TokenIdExt> tokens;
                if (!inputTokens.is_none())
                {
                    tokens = castTokenIterable(inputTokens);
                }
                nb::gil_scoped_release release;
                return self->probeReuse(std::move(reuseScope), tokens);
            },
            nb::arg("reuse_scope") = nb::none(), nb::arg("input_tokens") = nb::none())
        .def("get_mem_pool_base_address", &kv::KvCacheManager::getMemPoolBaseAddress, nb::arg("layer_id"),
            nb::arg("data_role"), nb::arg("index_mode") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def("get_page_stride", &kv::KvCacheManager::getPageStride, nb::arg("layer_id"), nb::arg("data_role"))
        .def("get_page_index_scale", &kv::KvCacheManager::getPageIndexScale, nb::arg("layer_id"), nb::arg("data_role"))
        .def("get_page_index_upper_bound", &kv::KvCacheManager::getPageIndexUpperBound, nb::arg("layer_id"),
            nb::arg("data_role"))
        .def(
            "resize",
            [](kv::KvCacheManager& self, int cacheLevel, size_t quota, bool bestEfforts)
            { return self.resize(kv::CacheLevel{cacheLevel}, quota, bestEfforts); },
            nb::arg("cache_level"), nb::arg("quota"), nb::arg("best_efforts") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_quota",
            [](kv::KvCacheManager const& self, int cacheLevel) { return self.getQuota(kv::CacheLevel{cacheLevel}); },
            nb::arg("cache_level"))
        .def_prop_ro("tokens_per_block", &kv::KvCacheManager::tokensPerBlock)
        .def_prop_ro("init_config", [](kv::KvCacheManager const& self) { return self.config(); })
        .def_prop_ro("cache_tier_list", [](kv::KvCacheManager const& self) { return self.cacheTierList().raw(); })
        .def_prop_ro("all_buffer_ids", &kv::KvCacheManager::allBufferIds)
        .def_prop_ro("pool_group_descs", [](kv::KvCacheManager const& self) { return self.poolGroupDescs().raw(); })
        .def("clamp_max_seq_len_for_mem", &kv::KvCacheManager::clampMaxSeqLenForMem, nb::arg("batch_size"),
            nb::arg("token_num_upper_bound"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("allow_seq_rebasing", &kv::KvCacheManager::allowSeqRebasing)
        .def_prop_ro("enable_partial_match", &kv::KvCacheManager::enablePartialMatch)
        .def_prop_ro("enable_swa_scratch_reuse", &kv::KvCacheManager::isSwaScratchReuseEnabled)
        .def(
            "supports_index_mode",
            [](kv::KvCacheManager const& self, kv::PageIndexMode mode) -> nb::object
            {
                auto result = self.supportsIndexMode(mode);
                if (!result.has_value())
                    return nb::none();
                return nb::cast(*result);
            },
            nb::arg("mode"))
        .def_prop_ro("ssm_reuse_interval", &kv::KvCacheManager::ssmReuseInterval)
        .def_prop_ro("num_layers", &kv::KvCacheManager::numLayers)
        .def_prop_ro("layer_ids", &kv::KvCacheManager::layerIds)
        .def_prop_ro("layer_grouping", [](kv::KvCacheManager const& self) { return self.layerGrouping().raw(); })
        .def(
            "get_layer_group_id",
            [](kv::KvCacheManager const& self, kv::LayerId layerId) { return self.getLayerGroupId(layerId).value(); },
            nb::arg("layer_id"))
        .def("get_page_index_converter", &kv::KvCacheManager::getPageIndexConverter, nb::arg("layer_id"),
            nb::arg("data_role"))
        .def(
            "get_aggregated_pages",
            [](kv::KvCacheManager const& self, nb::object buffers)
            {
                std::vector<kv::BufferId> ids;
                for (auto item : nb::cast<nb::iterable>(buffers))
                    ids.push_back(nb::cast<kv::BufferId>(item));
                nb::gil_scoped_release release;
                return self.getAggregatedPages(ids);
            },
            nb::arg("buffers"))
        .def("adjust", &kv::KvCacheManager::adjust, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("need_adjustment", &kv::KvCacheManager::needAdjustment);

    // ---- PageIndexConverter ------------------------------------------------
    nb::class_<kv::PageIndexConverter>(m, "PageIndexConverter")
        .def_ro("scale", &kv::PageIndexConverter::scale)
        .def_ro("expansion", &kv::PageIndexConverter::expansion)
        .def_ro("layer_offset", &kv::PageIndexConverter::layerOffset)
        .def_ro("scratch_pages_per_block", &kv::PageIndexConverter::scratchPagesPerBlock)
        .def(
            "__call__",
            [](kv::PageIndexConverter const& self, std::vector<int> const& baseIndices,
                std::optional<kv::PageIndexMode> indexMode, nb::object scratchObj) -> std::vector<int>
            {
                kv::ScratchDesc const* scratch = nullptr;
                std::optional<kv::ScratchDesc> scratchHolder;
                if (!scratchObj.is_none())
                {
                    scratchHolder = nb::cast<kv::ScratchDesc>(scratchObj);
                    scratch = &*scratchHolder;
                }
                return self(baseIndices, indexMode, scratch);
            },
            nb::arg("base_indices"), nb::arg("index_mode") = nb::none(), nb::arg("scratch") = nb::none());
}

} // namespace tensorrt_llm::nanobind::batch_manager
