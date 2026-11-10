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

// Torch ops of the adaptive expert load balancer's per-forward device work: the route of expert ids to the slots of a
// layout with several copies of an expert, and the expert statistics. The geometry comes from the tensors' shapes; the
// ops read their tables and write caller-owned CUDA tensors on the current stream, so CUDA graphs can capture them.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/loadBalance/adaptiveEplbKernels.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkIds(at::Tensor const& ids)
{
    TORCH_CHECK(ids.is_cuda(), "expert ids must be a CUDA tensor");
    TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "expert ids must be int32 or int64");
}

void checkVector(at::Tensor const& tensor, at::ScalarType dtype, at::Tensor const& ids, char const* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 1 && tensor.is_contiguous() && tensor.device() == ids.device(),
        name, " must be a contiguous 1-D ", dtype, " tensor on the device of the expert ids");
}

template <typename IdT, typename OutT>
void route(at::Tensor const& ids, int64_t epRank, kernels::AdaptiveEplbRouteTables const& tables, at::Tensor& out,
    kernels::AdaptiveEplbStaging const& staging, kernels::AdaptiveEplbStats const& stats)
{
    kernels::adaptiveEplbRoute(ids.data_ptr<IdT>(), ids.numel(), static_cast<int>(epRank), tables, out.data_ptr<OutT>(),
        staging, stats, at::cuda::getCurrentCUDAStream(ids.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

kernels::AdaptiveEplbRouteTables checkTables(at::Tensor const& ids, at::Tensor const& counts, at::Tensor const& starts,
    at::Tensor const& slots, at::Tensor const& cumulativeShares, int64_t epRank)
{
    checkIds(ids);
    TORCH_CHECK(epRank >= 0, "ep_rank must be non-negative");
    checkVector(counts, at::kInt, ids, "counts");
    checkVector(starts, at::kInt, ids, "starts");
    checkVector(slots, at::kInt, ids, "slots");
    checkVector(cumulativeShares, at::kInt, ids, "cumulative_shares");
    TORCH_CHECK(starts.numel() == counts.numel() && cumulativeShares.numel() == slots.numel(),
        "starts must have one entry per expert like counts, and cumulative_shares one per slot like slots");
    return {counts.data_ptr<int>(), starts.data_ptr<int>(), slots.data_ptr<int>(),
        reinterpret_cast<uint32_t const*>(cumulativeShares.data_ptr<int>()), static_cast<int>(counts.numel()),
        static_cast<int>(slots.numel())};
}

void routeInto(at::Tensor const& ids, kernels::AdaptiveEplbRouteTables const& tables, int64_t epRank, at::Tensor& out,
    kernels::AdaptiveEplbStaging const& staging, kernels::AdaptiveEplbStats const& stats)
{
    TORCH_CHECK(out.scalar_type() == at::kInt || out.scalar_type() == at::kLong, "out must be int32 or int64");
    auto const contiguousIds = ids.contiguous();
    bool const wideIds = ids.scalar_type() == at::kLong;
    bool const wideOut = out.scalar_type() == at::kLong;
    if (wideIds)
    {
        wideOut ? route<int64_t, int64_t>(contiguousIds, epRank, tables, out, staging, stats)
                : route<int64_t, int32_t>(contiguousIds, epRank, tables, out, staging, stats);
    }
    else
    {
        wideOut ? route<int32_t, int64_t>(contiguousIds, epRank, tables, out, staging, stats)
                : route<int32_t, int32_t>(contiguousIds, epRank, tables, out, staging, stats);
    }
}

kernels::AdaptiveEplbStats checkStats(at::Tensor const& ids, at::Tensor& rows, at::Tensor& forwards,
    at::Tensor const& enabled, bool first, std::optional<at::Tensor> const& liveRowWord)
{
    checkVector(rows, at::kFloat, ids, "rows");
    checkVector(forwards, at::kLong, ids, "forwards");
    checkVector(enabled, at::kInt, ids, "enabled");
    TORCH_CHECK(forwards.numel() == 1 && enabled.numel() == 1, "forwards and enabled must be one word each");
    int const* liveRowWordPtr = nullptr;
    if (liveRowWord.has_value())
    {
        checkVector(*liveRowWord, at::kInt, ids, "live_row_word");
        TORCH_CHECK(liveRowWord->numel() == 1, "live_row_word must be one word");
        liveRowWordPtr = liveRowWord->data_ptr<int>();
    }
    auto const topK = static_cast<int>(ids.dim() > 0 ? ids.size(-1) : 1);
    return {rows.data_ptr<float>(), forwards.data_ptr<int64_t>(), enabled.data_ptr<int>(), first, liveRowWordPtr, topK};
}

template <typename IdT>
void record(at::Tensor const& ids, kernels::AdaptiveEplbStats const& stats, int numExperts)
{
    kernels::adaptiveEplbRecord(ids.data_ptr<IdT>(), ids.numel(), numExperts, stats.rows, stats.forwards, stats.enabled,
        stats.first, stats.liveRowWord, stats.topK, at::cuda::getCurrentCUDAStream(ids.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

//! Route expert ids into a caller's buffer; the same launch can also stage the routes and update the layer's
//! statistics. out receives the [tokens, top_k] slots of the ids in its first ids.numel() elements. With weights
//! ([tokens, top_k] float32 gate weights) and weights_out ([rows, >= top_k] float32, unit column stride), out is a
//! bucket of rows >= tokens rows whose elements after the ids get slot -1 and weight 0; without them, out has exactly
//! the ids' elements. With rows, forwards and enabled, the launch adds the ids to the statistics as
//! adaptive_eplb_record does, with live_row_word only the ids of the forward's live rows.
void adaptiveEplbRouteInto(at::Tensor const& ids, at::Tensor const& counts, at::Tensor const& starts,
    at::Tensor const& slots, at::Tensor const& cumulativeShares, int64_t epRank, at::Tensor& out,
    std::optional<at::Tensor> const& weights, std::optional<at::Tensor> weightsOut, std::optional<at::Tensor> rows,
    std::optional<at::Tensor> forwards, std::optional<at::Tensor> const& enabled, bool first,
    std::optional<at::Tensor> const& liveRowWord)
{
    auto const tables = checkTables(ids, counts, starts, slots, cumulativeShares, epRank);
    TORCH_CHECK(out.is_contiguous() && out.device() == ids.device() && out.numel() >= ids.numel(),
        "out must be a contiguous buffer of at least the ids' elements on their device");
    auto const topK = static_cast<int>(ids.dim() > 0 ? ids.size(-1) : 1);
    kernels::AdaptiveEplbStaging staging{nullptr, nullptr, 0, topK, ids.numel()};
    TORCH_CHECK(weights.has_value() == weightsOut.has_value(), "weights and weights_out go together");
    if (weights.has_value())
    {
        TORCH_CHECK(ids.dim() == 2 && weights->scalar_type() == at::kFloat && weights->is_contiguous()
                && weights->sizes() == ids.sizes() && weights->device() == ids.device(),
            "ids must be [tokens, top_k] and weights contiguous float32 of the same shape on their device");
        TORCH_CHECK(weightsOut->scalar_type() == at::kFloat && weightsOut->dim() == 2 && weightsOut->stride(1) == 1
                && weightsOut->size(1) >= topK && weightsOut->stride(0) >= weightsOut->size(1)
                && weightsOut->device() == ids.device() && out.numel() % topK == 0
                && weightsOut->size(0) * topK >= out.numel(),
            "weights_out must be float32 [rows, >= top_k] with unit column stride, rows covering out");
        staging = {weights->data_ptr<float>(), weightsOut->data_ptr<float>(), weightsOut->stride(0), topK, out.numel()};
    }
    else
    {
        TORCH_CHECK(out.numel() == ids.numel(), "without weights, out has exactly the ids' elements");
    }
    kernels::AdaptiveEplbStats stats{};
    TORCH_CHECK(rows.has_value() == forwards.has_value() && rows.has_value() == enabled.has_value(),
        "rows, forwards and enabled go together");
    if (rows.has_value())
    {
        stats = checkStats(ids, *rows, *forwards, *enabled, first, liveRowWord);
        // The kernel adds one count per routed id at rows[expert id], for every expert of the tables.
        TORCH_CHECK(rows->numel() == tables.numExperts, "rows must have one entry per expert of the tables (",
            tables.numExperts, "), got ", rows->numel());
    }
    routeInto(ids, tables, epRank, out, staging, stats);
}

//! Add expert ids to a layer's statistics while enabled[0] is set: every call adds its per-expert counts to rows,
//! and the first call of a forward (not a further chunk or repeat) also increments forwards. With live_row_word, only
//! the ids of the forward's live rows are counted.
void adaptiveEplbRecord(at::Tensor const& ids, at::Tensor& rows, at::Tensor& forwards, at::Tensor const& enabled,
    bool first, std::optional<at::Tensor> const& liveRowWord)
{
    checkIds(ids);
    auto const stats = checkStats(ids, rows, forwards, enabled, first, liveRowWord);
    auto const contiguousIds = ids.contiguous();
    auto const numExperts = static_cast<int>(rows.numel());
    if (ids.scalar_type() == at::kLong)
    {
        record<int64_t>(contiguousIds, stats, numExperts);
    }
    else
    {
        record<int32_t>(contiguousIds, stats, numExperts);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "adaptive_eplb_route_into(Tensor ids, Tensor counts, Tensor starts, Tensor slots, Tensor cumulative_shares, "
        "int ep_rank, Tensor(a!) out, Tensor? weights=None, Tensor(b!)? weights_out=None, Tensor(c!)? rows=None, "
        "Tensor(d!)? forwards=None, Tensor? enabled=None, bool first=True, Tensor? live_row_word=None) -> ()");
    m.def(
        "adaptive_eplb_record(Tensor ids, Tensor(a!) rows, Tensor(b!) forwards, Tensor enabled, "
        "bool first=True, Tensor? live_row_word=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("adaptive_eplb_route_into", &tensorrt_llm::torch_ext::adaptiveEplbRouteInto);
    m.impl("adaptive_eplb_record", &tensorrt_llm::torch_ext::adaptiveEplbRecord);
}
