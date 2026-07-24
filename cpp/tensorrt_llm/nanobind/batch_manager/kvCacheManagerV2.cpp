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
#include "kv_cache_manager_v2/eventManager.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/introspection.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/stats.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

static nb::object optionalIntToObject(std::optional<std::uint64_t> value)
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

static nb::list tokenList(std::vector<kv::TokenIdExt> const& tokens)
{
    nb::list result;
    for (auto const& tok : tokens)
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

static nb::list committedTokensList(kv::KvCache const& self)
{
    return tokenList(self.committedTokens());
}

static nb::bytes digestBytes(kv::Digest const& digest)
{
    return nb::bytes(reinterpret_cast<char const*>(digest.data()), digest.size());
}

static kv::Digest castDigest(nb::handle value)
{
    if (!nb::isinstance<nb::bytes>(value))
    {
        throw std::invalid_argument("block hash must be bytes");
    }
    auto bytes = nb::cast<nb::bytes>(value);
    if (nb::len(bytes) != kv::kDIGEST_LEN)
    {
        throw std::invalid_argument("block hash bytes must have length kDIGEST_LEN");
    }
    kv::Digest digest;
    std::memcpy(digest.data(), bytes.c_str(), digest.size());
    return digest;
}

static bool hasDigestSize(nb::handle value)
{
    return nb::len(value) == kv::kDIGEST_LEN;
}

static kv::EventBlockHash castEventBlockHash(nb::handle value)
{
    if (PyLong_Check(value.ptr()))
    {
        return nb::cast<uint64_t>(value);
    }
    return nb::cast<std::string>(value);
}

static std::optional<kv::EventBlockHash> castOptionalEventBlockHash(nb::handle value)
{
    return value.is_none() ? std::nullopt : std::optional<kv::EventBlockHash>{castEventBlockHash(value)};
}

static std::vector<kv::KVCacheStoredBlockData> castStoredBlocks(nb::handle values)
{
    std::vector<kv::KVCacheStoredBlockData> result;
    for (nb::handle value : nb::cast<nb::iterable>(values))
    {
        result.push_back(nb::cast<kv::KVCacheStoredBlockData>(value));
    }
    return result;
}

static std::vector<kv::MmKey> castMmKeys(nb::handle values)
{
    std::vector<kv::MmKey> result;
    for (nb::handle value : nb::cast<nb::iterable>(values))
    {
        nb::tuple tuple = nb::cast<nb::tuple>(value);
        if (tuple.size() != 2 && tuple.size() != 3)
        {
            throw std::invalid_argument("mm_key must have two or three entries");
        }
        std::optional<std::string> uuid;
        if (tuple.size() == 3 && !tuple[2].is_none())
        {
            uuid = nb::cast<std::string>(tuple[2]);
        }
        if (!nb::isinstance<nb::bytes>(tuple[0]))
        {
            throw std::invalid_argument("mm_key hash must be bytes");
        }
        auto hash = nb::cast<nb::bytes>(tuple[0]);
        result.push_back(kv::MmKey{std::string(hash.c_str(), static_cast<size_t>(nb::len(hash))),
            nb::cast<int>(tuple[1]), std::move(uuid), tuple.size() == 3});
    }
    return result;
}

static nb::list castMmKeys(kv::KVCacheStoredBlockData const& data)
{
    nb::list result;
    for (auto const& mmKey : data.mmKeys)
    {
        auto hash = nb::bytes(mmKey.hash.data(), mmKey.hash.size());
        if (mmKey.hasUuidField)
        {
            result.append(nb::make_tuple(std::move(hash), mmKey.startOffset, mmKey.uuid));
        }
        else
        {
            result.append(nb::make_tuple(std::move(hash), mmKey.startOffset));
        }
    }
    return result;
}

static nb::object castEventData(kv::KVCacheEventData const& data)
{
    return std::visit([](auto const& concreteData) { return nb::cast(concreteData); }, data);
}

static kv::KVCacheEventData castEventData(nb::handle data)
{
    if (nb::isinstance<kv::KVCacheCreatedData>(data))
    {
        return nb::cast<kv::KVCacheCreatedData>(data);
    }
    if (nb::isinstance<kv::KVCacheStoredData>(data))
    {
        return nb::cast<kv::KVCacheStoredData>(data);
    }
    if (nb::isinstance<kv::KVCacheRemovedData>(data))
    {
        return nb::cast<kv::KVCacheRemovedData>(data);
    }
    if (nb::isinstance<kv::KVCacheUpdatedData>(data))
    {
        return nb::cast<kv::KVCacheUpdatedData>(data);
    }
    throw std::invalid_argument("Unsupported KV cache event data type");
}

static std::string pythonHandleRepr(nb::handle value)
{
    nb::object result = nb::steal<nb::object>(PyObject_Repr(value.ptr()));
    if (!result)
    {
        throw nb::python_error();
    }
    return nb::cast<std::string>(result);
}

template <typename T>
static std::string pythonRepr(T const& value)
{
    return pythonHandleRepr(nb::cast(value));
}

class PythonAttentionDpGather
{
public:
    explicit PythonAttentionDpGather(nb::handle callback)
        : mCallback(callback.ptr())
    {
        Py_INCREF(mCallback);
    }

    ~PythonAttentionDpGather()
    {
        if (Py_IsInitialized())
        {
            nb::gil_scoped_acquire acquire;
            Py_DECREF(mCallback);
        }
    }

    PythonAttentionDpGather(PythonAttentionDpGather const&) = delete;
    PythonAttentionDpGather& operator=(PythonAttentionDpGather const&) = delete;

    std::vector<std::vector<kv::KVCacheEvent>> operator()(std::vector<kv::KVCacheEvent> const& events) const
    {
        nb::gil_scoped_acquire acquire;
        return nb::cast<std::vector<std::vector<kv::KVCacheEvent>>>(nb::borrow<nb::object>(mCallback)(events));
    }

private:
    PyObject* mCallback;
};

static kv::EventManager::AttentionDpGatherFn castAttentionDpGather(nb::handle callback)
{
    if (callback.is_none())
    {
        return {};
    }
    if (!PyCallable_Check(callback.ptr()))
    {
        throw std::invalid_argument("attention_dp_gather must be callable");
    }
    auto gather = std::make_shared<PythonAttentionDpGather>(callback);
    return [gather = std::move(gather)](std::vector<kv::KVCacheEvent> const& events) { return (*gather)(events); };
}

static nb::object castStatsDelta(kv::KVCacheStatsDelta const& stats)
{
    return nb::cast(stats);
}

static nb::object castIterationStatsDelta(kv::KVCacheIterationStatsDelta const& stats)
{
    return nb::cast(stats);
}

static nb::dict castIterationStatsByLifeCycle(kv::IterationStatsByLifeCycle const& statsByLifeCycle)
{
    nb::dict result;
    for (auto const& [lifeCycle, stats] : statsByLifeCycle)
    {
        result[nb::int_(lifeCycle.value())] = castIterationStatsDelta(stats);
    }
    return result;
}

static nb::list castPeakBlockStats(kv::PeakBlockStatsByPoolGroup const& statsByPoolGroup)
{
    nb::list result;
    for (auto const& stats : statsByPoolGroup)
    {
        result.append(nb::cast(stats));
    }
    return result;
}

static std::string statsDeltaRepr(kv::KVCacheStatsDelta const& stats)
{
    std::ostringstream stream;
    stream << "KVCacheStatsDelta(alloc_total_blocks=" << stats.allocTotalBlocks
           << ", alloc_new_blocks=" << stats.allocNewBlocks << ", reused_blocks=" << stats.reusedBlocks
           << ", missed_blocks=" << stats.missedBlocks << ')';
    return stream.str();
}

static std::string iterationStatsDeltaRepr(kv::KVCacheIterationStatsDelta const& stats)
{
    std::ostringstream stream;
    stream << "KVCacheIterationStatsDelta(iter_alloc_total_blocks=" << stats.iterAllocTotalBlocks
           << ", iter_alloc_new_blocks=" << stats.iterAllocNewBlocks
           << ", iter_reused_blocks=" << stats.iterReusedBlocks
           << ", iter_full_reused_blocks=" << stats.iterFullReusedBlocks
           << ", iter_partial_reused_blocks=" << stats.iterPartialReusedBlocks
           << ", iter_missed_blocks=" << stats.iterMissedBlocks
           << ", iter_gen_alloc_blocks=" << stats.iterGenAllocBlocks
           << ", iter_onboard_blocks=" << stats.iterOnboardBlocks << ", iter_onboard_bytes=" << stats.iterOnboardBytes
           << ", iter_offload_blocks=" << stats.iterOffloadBlocks << ", iter_offload_bytes=" << stats.iterOffloadBytes
           << ", iter_intra_device_copy_blocks=" << stats.iterIntraDeviceCopyBlocks
           << ", iter_intra_device_copy_bytes=" << stats.iterIntraDeviceCopyBytes
           << ", iter_host_dropped_blocks=" << stats.iterHostDroppedBlocks
           << ", iter_host_dropped_bytes=" << stats.iterHostDroppedBytes << ')';
    return stream.str();
}

static std::string peakBlockStatsRepr(kv::PoolGroupPeakBlockStats const& stats)
{
    std::ostringstream stream;
    stream << "PoolGroupPeakBlockStats(available=" << stats.available << ", unavailable=" << stats.unavailable
           << ", evictable=" << stats.evictable << ')';
    return stream.str();
}

static nb::object castRequestIds(std::unordered_set<kv::RequestIdType> const& requestIds)
{
    nb::object result = nb::module_::import_("builtins").attr("set")();
    for (kv::RequestIdType const requestId : requestIds)
    {
        result.attr("add")(requestId);
    }
    return result;
}

static std::optional<std::uint64_t> castOptionalIntAttr(nb::handle obj, char const* attrName)
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
    return nb::cast<std::uint64_t>(attr);
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
        return {nb::cast<kv::LoraTaskIdType>(reuseScope), std::nullopt};
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

    // Map kv::AssertionError to Python's builtin AssertionError so shared tests see
    // the same exception type as the pure-Python backend (which uses `assert`).
    // Registered last so it is tried before the generic std::exception fallback.
    nb::register_exception_translator(
        [](std::exception_ptr const& p, void*)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (kv::AssertionError const& e)
            {
                PyErr_SetString(PyExc_AssertionError, e.what());
            }
        });

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

    // ---- KV cache events ----------------------------------------------------
    nb::class_<kv::UniqueToken>(m, "UniqueToken")
        .def(nb::init<kv::EventTokenId, int64_t>(), nb::arg("token_id"), nb::arg("token_extra_id") = 0)
        .def_ro("token_id", &kv::UniqueToken::tokenId)
        .def_ro("token_extra_id", &kv::UniqueToken::tokenExtraId)
        .def("__eq__", &kv::UniqueToken::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::UniqueToken const& self)
            {
                return "UniqueToken(token_id=" + pythonRepr(self.tokenId)
                    + ", token_extra_id=" + pythonRepr(self.tokenExtraId) + ")";
            })
        .def("__reduce__",
            [](kv::UniqueToken const& self)
            { return nb::make_tuple(nb::type<kv::UniqueToken>(), nb::make_tuple(self.tokenId, self.tokenExtraId)); });

    nb::class_<kv::KVCacheCreatedData>(m, "KVCacheCreatedData")
        .def(nb::init<std::vector<int>>(), nb::arg("num_blocks_per_cache_level"))
        .def_ro("num_blocks_per_cache_level", &kv::KVCacheCreatedData::numBlocksPerCacheLevel)
        .def("__eq__", &kv::KVCacheCreatedData::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheCreatedData const& self) {
                return "KVCacheCreatedData(num_blocks_per_cache_level=" + pythonRepr(self.numBlocksPerCacheLevel) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheCreatedData const& self) {
                return nb::make_tuple(nb::type<kv::KVCacheCreatedData>(), nb::make_tuple(self.numBlocksPerCacheLevel));
            });

    nb::class_<kv::KVCacheStoredBlockData>(m, "KVCacheStoredBlockData")
        .def(
            "__init__",
            [](kv::KVCacheStoredBlockData* self, kv::EventBlockHash blockHash, std::vector<kv::UniqueToken> tokens,
                int cacheLevel, int priority, nb::handle mmKeys, std::optional<std::string> cacheSalt)
            {
                new (self) kv::KVCacheStoredBlockData{std::move(blockHash), std::move(tokens), cacheLevel, priority,
                    castMmKeys(mmKeys), std::move(cacheSalt)};
            },
            nb::arg("block_hash"), nb::arg("tokens"), nb::arg("cache_level"), nb::arg("priority"),
            nb::arg("mm_keys") = nb::make_tuple(), nb::arg("cache_salt") = std::nullopt)
        .def_ro("block_hash", &kv::KVCacheStoredBlockData::blockHash)
        .def_ro("tokens", &kv::KVCacheStoredBlockData::tokens)
        .def_ro("cache_level", &kv::KVCacheStoredBlockData::cacheLevel)
        .def_ro("priority", &kv::KVCacheStoredBlockData::priority)
        .def_prop_ro("mm_keys", [](kv::KVCacheStoredBlockData const& self) { return castMmKeys(self); })
        .def_ro("cache_salt", &kv::KVCacheStoredBlockData::cacheSalt)
        .def("__eq__", &kv::KVCacheStoredBlockData::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheStoredBlockData const& self)
            {
                return "KVCacheStoredBlockData(block_hash=" + pythonRepr(self.blockHash)
                    + ", tokens=" + pythonRepr(self.tokens) + ", cache_level=" + pythonRepr(self.cacheLevel)
                    + ", priority=" + pythonRepr(self.priority) + ", mm_keys=" + pythonHandleRepr(castMmKeys(self))
                    + ", cache_salt=" + pythonRepr(self.cacheSalt) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheStoredBlockData const& self)
            {
                return nb::make_tuple(nb::type<kv::KVCacheStoredBlockData>(),
                    nb::make_tuple(
                        self.blockHash, self.tokens, self.cacheLevel, self.priority, castMmKeys(self), self.cacheSalt));
            });

    nb::class_<kv::KVCacheStoredData>(m, "KVCacheStoredData")
        .def(
            "__init__",
            [](kv::KVCacheStoredData* self, nb::object parentHash, nb::object blocks) {
                new (self) kv::KVCacheStoredData{castOptionalEventBlockHash(parentHash), castStoredBlocks(blocks)};
            },
            nb::arg("parent_hash").none(), nb::arg("blocks"))
        .def_ro("parent_hash", &kv::KVCacheStoredData::parentHash)
        .def_ro("blocks", &kv::KVCacheStoredData::blocks)
        .def("__eq__", &kv::KVCacheStoredData::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheStoredData const& self)
            {
                return "KVCacheStoredData(parent_hash=" + pythonRepr(self.parentHash)
                    + ", blocks=" + pythonRepr(self.blocks) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheStoredData const& self) {
                return nb::make_tuple(nb::type<kv::KVCacheStoredData>(), nb::make_tuple(self.parentHash, self.blocks));
            });

    nb::class_<kv::KVCacheRemovedData>(m, "KVCacheRemovedData")
        .def(nb::init<std::vector<kv::EventBlockHash>>(), nb::arg("block_hashes"))
        .def_ro("block_hashes", &kv::KVCacheRemovedData::blockHashes)
        .def("__eq__", &kv::KVCacheRemovedData::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheRemovedData const& self)
            { return "KVCacheRemovedData(block_hashes=" + pythonRepr(self.blockHashes) + ")"; })
        .def("__reduce__",
            [](kv::KVCacheRemovedData const& self)
            { return nb::make_tuple(nb::type<kv::KVCacheRemovedData>(), nb::make_tuple(self.blockHashes)); });

    nb::class_<kv::KVCacheEventDiff>(m, "KVCacheEventDiff")
        .def(nb::init<int, int>(), nb::arg("old_value"), nb::arg("new_value"))
        .def_ro("old_value", &kv::KVCacheEventDiff::oldValue)
        .def_ro("new_value", &kv::KVCacheEventDiff::newValue)
        .def("__eq__", &kv::KVCacheEventDiff::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheEventDiff const& self)
            {
                return "KVCacheEventDiff(old_value=" + pythonRepr(self.oldValue)
                    + ", new_value=" + pythonRepr(self.newValue) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheEventDiff const& self)
            { return nb::make_tuple(nb::type<kv::KVCacheEventDiff>(), nb::make_tuple(self.oldValue, self.newValue)); });

    nb::class_<kv::KVCacheUpdatedData>(m, "KVCacheUpdatedData")
        .def(nb::init<kv::EventBlockHash, std::optional<kv::KVCacheEventDiff>, std::optional<kv::KVCacheEventDiff>>(),
            nb::arg("block_hash"), nb::arg("cache_level").none(), nb::arg("priority").none())
        .def_ro("block_hash", &kv::KVCacheUpdatedData::blockHash)
        .def_ro("cache_level", &kv::KVCacheUpdatedData::cacheLevel)
        .def_ro("priority", &kv::KVCacheUpdatedData::priority)
        .def("__eq__", &kv::KVCacheUpdatedData::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheUpdatedData const& self)
            {
                return "KVCacheUpdatedData(block_hash=" + pythonRepr(self.blockHash)
                    + ", cache_level=" + pythonRepr(self.cacheLevel) + ", priority=" + pythonRepr(self.priority) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheUpdatedData const& self)
            {
                return nb::make_tuple(
                    nb::type<kv::KVCacheUpdatedData>(), nb::make_tuple(self.blockHash, self.cacheLevel, self.priority));
            });

    nb::class_<kv::KVCacheEvent>(m, "KVCacheEvent")
        .def(
            "__init__",
            [](kv::KVCacheEvent* self, int64_t eventId, nb::handle data, int windowSize,
                std::optional<std::string> hashAlgo, std::optional<int> attentionDpRank,
                kv::EventLayerGroupId layerGroupId)
            {
                new (self) kv::KVCacheEvent{
                    eventId, castEventData(data), windowSize, std::move(hashAlgo), attentionDpRank, layerGroupId};
            },
            nb::arg("event_id"), nb::arg("data"), nb::arg("window_size"), nb::arg("hash_algo") = std::nullopt,
            nb::arg("attention_dp_rank") = std::nullopt, nb::arg("layer_group_id") = std::nullopt)
        .def_ro("event_id", &kv::KVCacheEvent::eventId)
        .def_prop_ro("data", [](kv::KVCacheEvent const& self) { return castEventData(self.data); })
        .def_ro("window_size", &kv::KVCacheEvent::windowSize)
        .def_ro("hash_algo", &kv::KVCacheEvent::hashAlgo)
        .def_ro("attention_dp_rank", &kv::KVCacheEvent::attentionDpRank)
        .def_ro("layer_group_id", &kv::KVCacheEvent::layerGroupId)
        .def("__eq__", &kv::KVCacheEvent::operator==, nb::arg("other"))
        .def("__repr__",
            [](kv::KVCacheEvent const& self)
            {
                return "KVCacheEvent(event_id=" + pythonRepr(self.eventId) + ", data="
                    + pythonHandleRepr(castEventData(self.data)) + ", window_size=" + pythonRepr(self.windowSize)
                    + ", hash_algo=" + pythonRepr(self.hashAlgo) + ", attention_dp_rank="
                    + pythonRepr(self.attentionDpRank) + ", layer_group_id=" + pythonRepr(self.layerGroupId) + ")";
            })
        .def("__reduce__",
            [](kv::KVCacheEvent const& self)
            {
                return nb::make_tuple(nb::type<kv::KVCacheEvent>(),
                    nb::make_tuple(self.eventId, castEventData(self.data), self.windowSize, self.hashAlgo,
                        self.attentionDpRank, self.layerGroupId));
            });

    nb::class_<kv::EventManager>(m, "KVCacheEventManager")
        .def(
            "__init__",
            [](kv::EventManager* self, int maxKvEventEntries, int windowSize, std::optional<int> attentionDpRank,
                nb::handle attentionDpGather, std::string hashAlgo, nb::handle windowSizeByLayerGroup)
            {
                std::map<int, int> windowSizes;
                if (!windowSizeByLayerGroup.is_none())
                {
                    windowSizes = nb::cast<std::map<int, int>>(windowSizeByLayerGroup);
                }
                new (self) kv::EventManager(maxKvEventEntries, windowSize, attentionDpRank,
                    castAttentionDpGather(attentionDpGather), std::move(hashAlgo), std::move(windowSizes));
            },
            nb::arg("max_kv_event_entries"), nb::kw_only(), nb::arg("window_size") = 0,
            nb::arg("attention_dp_rank") = std::nullopt, nb::arg("attention_dp_gather") = nb::none(),
            nb::arg("hash_algo") = "v2_sha256", nb::arg("window_size_by_layer_group").none() = nb::none())
        .def("add_created_event", &kv::EventManager::addCreatedEvent, nb::arg("num_blocks_per_cache_level"),
            nb::arg("layer_group_ids") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def("set_layer_group_window_sizes", &kv::EventManager::setLayerGroupWindowSizes, nb::arg("window_sizes"),
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "add_stored_event",
            [](kv::EventManager& self, nb::object parentHash, nb::object blocks, kv::EventLayerGroupId layerGroupId)
            {
                self.addStoredEvent(
                    kv::KVCacheStoredData{castOptionalEventBlockHash(parentHash), castStoredBlocks(blocks)},
                    layerGroupId);
            },
            nb::arg("parent_hash").none(), nb::arg("blocks"), nb::arg("layer_group_id") = std::nullopt)
        .def(
            "add_removed_event",
            [](kv::EventManager& self, nb::handle blockHashes)
            {
                if (nb::isinstance<nb::bytes>(blockHashes))
                {
                    if (hasDigestSize(blockHashes))
                    {
                        self.addRemovedBlock(castDigest(blockHashes));
                    }
                    return;
                }
                if (PyLong_Check(blockHashes.ptr()) || nb::isinstance<nb::str>(blockHashes))
                {
                    self.addRemovedEvent({castEventBlockHash(blockHashes)});
                    return;
                }
                for (nb::handle blockHash : nb::cast<nb::iterable>(blockHashes))
                {
                    if (nb::isinstance<nb::bytes>(blockHash))
                    {
                        if (hasDigestSize(blockHash))
                        {
                            self.addRemovedBlock(castDigest(blockHash));
                        }
                    }
                    else
                    {
                        self.addRemovedEvent({castEventBlockHash(blockHash)});
                    }
                }
            },
            nb::arg("block_hashes"))
        .def(
            "add_removed_life_cycle_event",
            [](kv::EventManager& self, nb::handle blockHash, int lifeCycleId)
            {
                if (!nb::isinstance<nb::bytes>(blockHash))
                {
                    throw std::invalid_argument("block hash must be bytes");
                }
                if (hasDigestSize(blockHash))
                {
                    self.addRemovedLifeCycle(castDigest(blockHash), kv::LifeCycleId{lifeCycleId});
                }
            },
            nb::arg("block_hash"), nb::arg("life_cycle_id"))
        .def(
            "add_updated_event",
            [](kv::EventManager& self, nb::handle blockHash, std::optional<kv::KVCacheEventDiff> cacheLevel,
                std::optional<kv::KVCacheEventDiff> priority, kv::EventLayerGroupId layerGroupId)
            {
                if (!cacheLevel.has_value() && !priority.has_value())
                {
                    return;
                }
                if (nb::isinstance<nb::bytes>(blockHash))
                {
                    if (!hasDigestSize(blockHash))
                    {
                        return;
                    }
                    auto digest = castDigest(blockHash);
                    nb::gil_scoped_release release;
                    self.addUpdatedEvent(digest, cacheLevel, priority, layerGroupId);
                    return;
                }
                auto eventBlockHash = castEventBlockHash(blockHash);
                nb::gil_scoped_release release;
                self.addUpdatedEvent(std::move(eventBlockHash), cacheLevel, priority, layerGroupId);
            },
            nb::arg("block_hash"), nb::kw_only(), nb::arg("cache_level") = std::nullopt,
            nb::arg("priority") = std::nullopt, nb::arg("layer_group_id") = std::nullopt)
        .def(
            "flush_iteration_events", &kv::EventManager::flushIterationEvents, nb::call_guard<nb::gil_scoped_release>())
        .def("get_latest_events", &kv::EventManager::getLatestEvents, nb::arg("timeout_ms") = std::nullopt,
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("hash_algo", &kv::EventManager::hashAlgorithm)
        .def_prop_ro("_hash_algo", &kv::EventManager::hashAlgorithm)
        .def_static("_hash_block_key", &kv::EventManager::hashV1BlockKey, nb::arg("tokens"), nb::arg("parent_hash") = 0,
            nb::arg("lora_task_id") = std::nullopt, nb::arg("cache_salt_id") = std::nullopt);

    // ---- Statistics --------------------------------------------------------
    nb::class_<kv::KVCacheStatsDelta>(m, "KVCacheStatsDelta")
        .def(
            "__init__",
            [](kv::KVCacheStatsDelta* self, int64_t allocTotalBlocks, int64_t allocNewBlocks, int64_t reusedBlocks,
                int64_t missedBlocks)
            {
                new (self) kv::KVCacheStatsDelta{};
                self->allocTotalBlocks = allocTotalBlocks;
                self->allocNewBlocks = allocNewBlocks;
                self->reusedBlocks = reusedBlocks;
                self->missedBlocks = missedBlocks;
            },
            nb::arg("alloc_total_blocks") = 0, nb::arg("alloc_new_blocks") = 0, nb::arg("reused_blocks") = 0,
            nb::arg("missed_blocks") = 0)
        .def_rw("alloc_total_blocks", &kv::KVCacheStatsDelta::allocTotalBlocks)
        .def_rw("alloc_new_blocks", &kv::KVCacheStatsDelta::allocNewBlocks)
        .def_rw("reused_blocks", &kv::KVCacheStatsDelta::reusedBlocks)
        .def_rw("missed_blocks", &kv::KVCacheStatsDelta::missedBlocks)
        .def("add", &kv::KVCacheStatsDelta::add, nb::arg("other"))
        .def("subtract", &kv::KVCacheStatsDelta::subtract, nb::arg("other"))
        .def("clear", &kv::KVCacheStatsDelta::clear)
        .def("copy", &kv::KVCacheStatsDelta::copy)
        .def_prop_ro("empty", &kv::KVCacheStatsDelta::empty)
        .def("__eq__", &kv::KVCacheStatsDelta::operator==, nb::arg("other"))
        .def("__repr__", &statsDeltaRepr);

    nb::class_<kv::KVCacheIterationStatsDelta>(m, "KVCacheIterationStatsDelta")
        .def(
            "__init__",
            [](kv::KVCacheIterationStatsDelta* self, int64_t iterAllocTotalBlocks, int64_t iterAllocNewBlocks,
                int64_t iterReusedBlocks, int64_t iterFullReusedBlocks, int64_t iterPartialReusedBlocks,
                int64_t iterMissedBlocks, int64_t iterGenAllocBlocks, int64_t iterOnboardBlocks,
                int64_t iterOnboardBytes, int64_t iterOffloadBlocks, int64_t iterOffloadBytes,
                int64_t iterIntraDeviceCopyBlocks, int64_t iterIntraDeviceCopyBytes, int64_t iterHostDroppedBlocks,
                int64_t iterHostDroppedBytes)
            {
                new (self) kv::KVCacheIterationStatsDelta{};
                self->iterAllocTotalBlocks = iterAllocTotalBlocks;
                self->iterAllocNewBlocks = iterAllocNewBlocks;
                self->iterReusedBlocks = iterReusedBlocks;
                self->iterFullReusedBlocks = iterFullReusedBlocks;
                self->iterPartialReusedBlocks = iterPartialReusedBlocks;
                self->iterMissedBlocks = iterMissedBlocks;
                self->iterGenAllocBlocks = iterGenAllocBlocks;
                self->iterOnboardBlocks = iterOnboardBlocks;
                self->iterOnboardBytes = iterOnboardBytes;
                self->iterOffloadBlocks = iterOffloadBlocks;
                self->iterOffloadBytes = iterOffloadBytes;
                self->iterIntraDeviceCopyBlocks = iterIntraDeviceCopyBlocks;
                self->iterIntraDeviceCopyBytes = iterIntraDeviceCopyBytes;
                self->iterHostDroppedBlocks = iterHostDroppedBlocks;
                self->iterHostDroppedBytes = iterHostDroppedBytes;
            },
            nb::arg("iter_alloc_total_blocks") = 0, nb::arg("iter_alloc_new_blocks") = 0,
            nb::arg("iter_reused_blocks") = 0, nb::arg("iter_full_reused_blocks") = 0,
            nb::arg("iter_partial_reused_blocks") = 0, nb::arg("iter_missed_blocks") = 0,
            nb::arg("iter_gen_alloc_blocks") = 0, nb::arg("iter_onboard_blocks") = 0, nb::arg("iter_onboard_bytes") = 0,
            nb::arg("iter_offload_blocks") = 0, nb::arg("iter_offload_bytes") = 0,
            nb::arg("iter_intra_device_copy_blocks") = 0, nb::arg("iter_intra_device_copy_bytes") = 0,
            nb::arg("iter_host_dropped_blocks") = 0, nb::arg("iter_host_dropped_bytes") = 0)
        .def_rw("iter_alloc_total_blocks", &kv::KVCacheIterationStatsDelta::iterAllocTotalBlocks)
        .def_rw("iter_alloc_new_blocks", &kv::KVCacheIterationStatsDelta::iterAllocNewBlocks)
        .def_rw("iter_reused_blocks", &kv::KVCacheIterationStatsDelta::iterReusedBlocks)
        .def_rw("iter_full_reused_blocks", &kv::KVCacheIterationStatsDelta::iterFullReusedBlocks)
        .def_rw("iter_partial_reused_blocks", &kv::KVCacheIterationStatsDelta::iterPartialReusedBlocks)
        .def_rw("iter_missed_blocks", &kv::KVCacheIterationStatsDelta::iterMissedBlocks)
        .def_rw("iter_gen_alloc_blocks", &kv::KVCacheIterationStatsDelta::iterGenAllocBlocks)
        .def_rw("iter_onboard_blocks", &kv::KVCacheIterationStatsDelta::iterOnboardBlocks)
        .def_rw("iter_onboard_bytes", &kv::KVCacheIterationStatsDelta::iterOnboardBytes)
        .def_rw("iter_offload_blocks", &kv::KVCacheIterationStatsDelta::iterOffloadBlocks)
        .def_rw("iter_offload_bytes", &kv::KVCacheIterationStatsDelta::iterOffloadBytes)
        .def_rw("iter_intra_device_copy_blocks", &kv::KVCacheIterationStatsDelta::iterIntraDeviceCopyBlocks)
        .def_rw("iter_intra_device_copy_bytes", &kv::KVCacheIterationStatsDelta::iterIntraDeviceCopyBytes)
        .def_rw("iter_host_dropped_blocks", &kv::KVCacheIterationStatsDelta::iterHostDroppedBlocks)
        .def_rw("iter_host_dropped_bytes", &kv::KVCacheIterationStatsDelta::iterHostDroppedBytes)
        .def("add", &kv::KVCacheIterationStatsDelta::add, nb::arg("other"))
        .def("subtract", &kv::KVCacheIterationStatsDelta::subtract, nb::arg("other"))
        .def("clear", &kv::KVCacheIterationStatsDelta::clear)
        .def("copy", &kv::KVCacheIterationStatsDelta::copy)
        .def_prop_ro("empty", &kv::KVCacheIterationStatsDelta::empty)
        .def_prop_ro("iter_cache_hit_rate", &kv::KVCacheIterationStatsDelta::iterCacheHitRate)
        .def("__eq__", &kv::KVCacheIterationStatsDelta::operator==, nb::arg("other"))
        .def("__repr__", &iterationStatsDeltaRepr);

    m.attr("KVCacheIterationStatsDelta").attr("_field_names") = nb::make_tuple("iter_alloc_total_blocks",
        "iter_alloc_new_blocks", "iter_reused_blocks", "iter_full_reused_blocks", "iter_partial_reused_blocks",
        "iter_missed_blocks", "iter_gen_alloc_blocks", "iter_onboard_blocks", "iter_onboard_bytes",
        "iter_offload_blocks", "iter_offload_bytes", "iter_intra_device_copy_blocks", "iter_intra_device_copy_bytes",
        "iter_host_dropped_blocks", "iter_host_dropped_bytes");

    nb::class_<kv::PoolGroupPeakBlockStats>(m, "PoolGroupPeakBlockStats")
        .def(
            "__init__",
            [](kv::PoolGroupPeakBlockStats* self, kv::SlotCount available, kv::SlotCount unavailable,
                kv::SlotCount evictable) {
                new (self) kv::PoolGroupPeakBlockStats{available, unavailable, evictable};
            },
            nb::arg("available"), nb::arg("unavailable"), nb::arg("evictable"))
        .def_ro("available", &kv::PoolGroupPeakBlockStats::available)
        .def_ro("unavailable", &kv::PoolGroupPeakBlockStats::unavailable)
        .def_ro("evictable", &kv::PoolGroupPeakBlockStats::evictable)
        .def("__eq__", &kv::PoolGroupPeakBlockStats::operator==, nb::arg("other"))
        .def("__repr__", &peakBlockStatsRepr);

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
        .def(nb::init<std::optional<kv::LoraTaskIdType>, std::optional<std::uint64_t>>(),
            nb::arg("lora_id").none() = std::nullopt, nb::arg("salt").none() = std::nullopt)
        .def_ro("lora_id", &kv::ReuseScope::loraId)
        .def_ro("salt", &kv::ReuseScope::salt)
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
        .def_rw("history_length", &kv::KVCacheDesc::historyLength)
        .def("__eq__",
            [](kv::KVCacheDesc const& self, nb::handle other)
            {
                if (!nb::isinstance<kv::KVCacheDesc>(other))
                {
                    return false;
                }
                return self == nb::cast<kv::KVCacheDesc>(other);
            })
        .def("__repr__",
            [](kv::KVCacheDesc const& self)
            {
                return "KVCacheDesc(capacity=" + std::to_string(self.capacity)
                    + ", history_length=" + std::to_string(self.historyLength) + ")";
            }) DEF_COPY(kv::KVCacheDesc);

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
        .def_rw("system_prompt_length", &kv::BatchDesc::systemPromptLength)
        .def("__eq__",
            [](kv::BatchDesc const& self, nb::handle other)
            {
                if (!nb::isinstance<kv::BatchDesc>(other))
                {
                    return false;
                }
                return self == nb::cast<kv::BatchDesc>(other);
            })
        .def("__repr__",
            [](kv::BatchDesc const& self)
            {
                std::string repr = "BatchDesc(kv_caches=[";
                for (size_t i = 0; i < self.kvCaches.size(); ++i)
                {
                    if (i != 0)
                    {
                        repr += ", ";
                    }
                    repr += "KVCacheDesc(capacity=" + std::to_string(self.kvCaches[i].capacity)
                        + ", history_length=" + std::to_string(self.kvCaches[i].historyLength) + ")";
                }
                repr += "], system_prompt_length=" + std::to_string(self.systemPromptLength) + ")";
                return repr;
            }) DEF_COPY(kv::BatchDesc);

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
                std::optional<kv::BatchDesc> typicalStep, std::vector<kv::BatchDesc> constraints,
                std::optional<std::vector<float>> initialPoolRatio,
                std::optional<kv::SwaScratchReuseConfig> swaScratchReuse, bool commitMinSnapshot, bool enableStats)
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
                cfg->initialPoolRatio = std::move(initialPoolRatio);
                cfg->swaScratchReuse = std::move(swaScratchReuse);
                cfg->commitMinSnapshot = commitMinSnapshot;
                cfg->enableStats = enableStats;
                // Mirror Python's __post_init__: validate at construction. Config-integrity
                // failures raise AssertionError (translated below).
                cfg->validate();
            },
            nb::arg("tokens_per_block"), nb::arg("cache_tiers"), nb::arg("layers"),
            nb::arg("max_util_for_resume") = 0.97f, nb::arg("enable_partial_reuse") = true,
            nb::arg("typical_step") = std::nullopt, nb::arg("constraints") = std::vector<kv::BatchDesc>{},
            nb::arg("initial_pool_ratio").none() = std::nullopt, nb::arg("swa_scratch_reuse").none() = std::nullopt,
            nb::arg("commit_min_snapshot") = false, nb::arg("enable_stats") = true)
        .def_rw("tokens_per_block", &kv::KVCacheManagerConfig::tokensPerBlock)
        .def_rw("cache_tiers", &kv::KVCacheManagerConfig::cacheTiers)
        .def_rw("layers", &kv::KVCacheManagerConfig::layers)
        .def_rw("max_util_for_resume", &kv::KVCacheManagerConfig::maxUtilForResume)
        .def_rw("enable_partial_reuse", &kv::KVCacheManagerConfig::enablePartialReuse)
        .def_rw("typical_step", &kv::KVCacheManagerConfig::typicalStep)
        .def_rw("constraints", &kv::KVCacheManagerConfig::constraints)
        .def_rw("initial_pool_ratio", &kv::KVCacheManagerConfig::initialPoolRatio)
        .def_rw("swa_scratch_reuse", &kv::KVCacheManagerConfig::swaScratchReuse)
        .def_rw("commit_min_snapshot", &kv::KVCacheManagerConfig::commitMinSnapshot)
        .def_rw("enable_stats", &kv::KVCacheManagerConfig::enableStats)
        .def_prop_ro("enable_swa_scratch_reuse", &kv::KVCacheManagerConfig::enableSwaScratchReuse)
        .def("validate", &kv::KVCacheManagerConfig::validate) DEF_COPY(kv::KVCacheManagerConfig);

#undef DEF_COPY

    // ---- PlannedDropHandle -------------------------------------------------
    nb::class_<kv::PlannedDropHandle>(m, "PlannedDropHandle")
        .def("drop", &kv::PlannedDropHandle::drop, nb::call_guard<nb::gil_scoped_release>());

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
        .def("commit_pending_stats", [](kv::KvCache& self) { return castStatsDelta(self.commitPendingStats()); })
        .def("discard_pending_stats", &kv::KvCache::discardPendingStats)
        .def(
            "resize",
            [](kv::KvCache& self, std::optional<int> capacity, std::optional<int> historyLength) -> bool
            { return self.resize(capacity, historyLength); },
            nb::arg("capacity") = std::nullopt, nb::arg("history_length") = std::nullopt,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "commit",
            [](kv::KvCache& self, nb::object acceptedInputTokens, nb::object beamSearchIndices, bool isEnd)
            {
                auto vec = castTokenIterable(acceptedInputTokens);
                if (!beamSearchIndices.is_none())
                {
                    PyErr_SetString(PyExc_AssertionError, "beam_search_indices must be None");
                    throw nb::python_error();
                }
                // Note: an empty token list with is_end=True must still stop committing,
                // so we do not early-return on empty; commit() handles it.
                nb::gil_scoped_release release;
                self.commit(vec, isEnd);
            },
            nb::arg("accepted_input_tokens"), nb::arg("beam_search_indices").none() = nb::none(),
            nb::arg("is_end") = false)
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
            "plan_committed_block_drop", [](kv::KvCache& self) { return self.planCommittedBlockDrop(); },
            nb::call_guard<nb::gil_scoped_release>())
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
            nb::arg("beam_idx"), nb::arg("layer_group_id"), nb::arg("buf").none());

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
        "life_cycle_pool_group_indices",
        [](kv::KvCacheManager& manager)
        {
            std::vector<int> result;
            result.reserve(manager.lifeCycles().size().value());
            for (kv::LifeCycleId lifeCycle{0}; lifeCycle < manager.lifeCycles().size(); ++lifeCycle)
            {
                result.push_back(manager.storage().getPoolGroupIndex(lifeCycle).value());
            }
            return result;
        },
        nb::arg("manager"), nb::call_guard<nb::gil_scoped_release>());
    // White-box reuse-tree introspection: mirror the Python manager's _life_cycles
    // and _radix_tree attributes so shared tests can inspect reuse state.
    mIntrospection.def(
        "attention_life_cycle_ids",
        [](kv::KvCacheManager& manager)
        {
            std::vector<int> result;
            for (auto const& [lcId, attn] : manager.lifeCycles().attentionLifeCycles())
                result.push_back(lcId.value());
            return result;
        },
        nb::arg("manager"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "swa_life_cycle_ids",
        [](kv::KvCacheManager& manager)
        {
            std::vector<int> result;
            for (auto const& [lcId, attn] : manager.lifeCycles().attentionLifeCycles())
                if (attn->windowSize.has_value())
                    result.push_back(lcId.value());
            return result;
        },
        nb::arg("manager"), nb::call_guard<nb::gil_scoped_release>());
    mIntrospection.def(
        "ssm_life_cycle_id",
        [](kv::KvCacheManager& manager) -> std::optional<int>
        {
            auto id = manager.lifeCycles().ssmLifeCycleId();
            if (id.has_value())
                return id->value();
            return std::nullopt;
        },
        nb::arg("manager"), nb::call_guard<nb::gil_scoped_release>());
    // Returns (num_tokens, pages) where pages[i] is None for a block with no page in
    // this lifecycle, else (slot_id, num_tokens_in_block) with num_tokens_in_block = -1
    // for a non-SSM (attention) page.
    mIntrospection.def(
        "reuse_match_pages",
        [](kv::KvCacheManager& manager, nb::object reuseScope, nb::object tokens, int lcId, bool enablePartial)
        {
            auto rs = castReuseScope(reuseScope);
            auto vec = castTokenIterable(tokens);
            int numTokens = 0;
            std::vector<std::optional<std::pair<int, int>>> pages;
            {
                nb::gil_scoped_release release;
                auto matchResult = manager.radixTree().match(rs, vec, enablePartial);
                numTokens = matchResult.numTokens;
                kv::LifeCycleId lc{lcId};
                pages.reserve(matchResult.blocks.size());
                for (auto* block : matchResult.blocks)
                {
                    auto* page = block->storage.at(lc);
                    if (page == nullptr)
                    {
                        pages.emplace_back(std::nullopt);
                        continue;
                    }
                    int const slotId = page->slotId().value();
                    int numTokensInBlock = -1;
                    if (auto* ssm = dynamic_cast<kv::SsmCommittedPage*>(page))
                        numTokensInBlock = ssm->numTokensInBlock;
                    pages.emplace_back(std::make_pair(slotId, numTokensInBlock));
                }
            }
            return std::make_tuple(numTokens, std::move(pages));
        },
        nb::arg("manager"), nb::arg("reuse_scope"), nb::arg("tokens"), nb::arg("lc_id"),
        nb::arg("enable_partial") = false);
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
        .def(
            "__init__",
            [](kv::KvCacheManager* self, kv::KVCacheManagerConfig const& config, nb::object eventManager)
            {
                std::shared_ptr<kv::EventSink> eventSink;
                if (!eventManager.is_none())
                {
                    eventSink = nb::cast<std::shared_ptr<kv::EventManager>>(eventManager);
                }
                nb::gil_scoped_release release;
                new (self) kv::KvCacheManager(config, std::move(eventSink));
            },
            nb::arg("config"), nb::arg("event_manager").none() = nb::none())
        .def("shutdown", &kv::KvCacheManager::shutdown, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "clear_reusable_blocks", &kv::KvCacheManager::clearReusableBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_kv_cache",
            [](std::shared_ptr<kv::KvCacheManager> self, nb::object reuseScopeObj, nb::object inputTokens,
                std::optional<kv::RequestIdType> id, nb::object customPriorityCallback,
                std::optional<int> expectedPromptLength)
            {
                kv::ReuseScope reuseScope = castReuseScope(std::move(reuseScopeObj));
                std::vector<kv::TokenIdExt> tokens;
                bool const hasInputTokens = !inputTokens.is_none();
                if (!inputTokens.is_none())
                {
                    tokens = castTokenIterable(inputTokens);
                }
                if (!expectedPromptLength.has_value() && hasInputTokens)
                {
                    expectedPromptLength = static_cast<int>(tokens.size());
                }
                kv::KvCache::PriorityCb priorityCb = castPriorityCallback(*self, std::move(customPriorityCallback));
                nb::gil_scoped_release release;
                return self->createKvCache(
                    std::move(reuseScope), tokens, id, std::move(priorityCb), expectedPromptLength);
            },
            nb::arg("reuse_scope") = nb::none(), nb::arg("input_tokens") = nb::none(), nb::arg("id") = std::nullopt,
            nb::arg("custom_priority_callback") = nb::none(), nb::arg("expected_prompt_length") = std::nullopt)
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
        .def("get_committed_stats",
            [](kv::KvCacheManager const& self) { return castStatsDelta(self.getCommittedStats()); })
        .def("get_and_reset_iteration_stats",
            [](kv::KvCacheManager& self) { return castIterationStatsByLifeCycle(self.getAndResetIterationStats()); })
        .def(
            "get_and_reset_iteration_peak_block_stats",
            [](kv::KvCacheManager& self, int cacheLevel)
            { return castPeakBlockStats(self.getAndResetIterationPeakBlockStats(kv::CacheLevel{cacheLevel})); },
            nb::arg("cache_level"))
        .def("mark_stats_dirty", &kv::KvCacheManager::markStatsDirty, nb::arg("kv_cache_id").none())
        .def("clear_stats_dirty", &kv::KvCacheManager::clearStatsDirty, nb::arg("kv_cache_id").none())
        .def("get_dirty_stats_kv_cache_ids",
            [](kv::KvCacheManager const& self) { return castRequestIds(self.getDirtyStatsKvCacheIds()); })
        .def("mark_stats_excluded", &kv::KvCacheManager::markStatsExcluded, nb::arg("kv_cache_id").none())
        .def("clear_stats_excluded", &kv::KvCacheManager::clearStatsExcluded, nb::arg("kv_cache_id").none())
        .def("is_stats_excluded", &kv::KvCacheManager::isStatsExcluded, nb::arg("kv_cache_id").none())
        .def_prop_ro("tokens_per_block", &kv::KvCacheManager::tokensPerBlock)
        .def_prop_ro("event_manager",
            [](kv::KvCacheManager const& self)
            { return std::dynamic_pointer_cast<kv::EventManager>(self.eventSink()); })
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
        .def_prop_ro("commit_min_snapshot", &kv::KvCacheManager::commitMinSnapshot)
        .def_prop_ro("num_layers", &kv::KvCacheManager::numLayers)
        .def_prop_ro("layer_ids", &kv::KvCacheManager::layerIds)
        .def_prop_ro(
            "layer_grouping", [](kv::KvCacheManager const& self) { return self.layerGrouping().raw(); },
            "Layers grouped by shared lifecycle/pool allocation. The iteration order of the "
            "layer lists (and of the groups) is NOT an API contract and may differ across "
            "backends/runs; do not rely on it for buffer/pool memory order -- use "
            "pool_group_descs (PoolGroupDesc.pools[i].base_address + coalesced_buffers) instead.")
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
