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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/sparseDecodeOwnT.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <atomic>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHeadsPerKv = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPage = 128;
constexpr int64_t kTopK = 16;

void checkBytes(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.element_size() == 1, "minimax_m3_sm107_sparse_decode_own_t: ", name,
        " must be a one-byte CUDA view (uint8 / e4m3 / packed e2m1)");
}

//! Whether the view's base and its two outer strides (in bytes) are multiples of `align`.
bool aligned(torch::Tensor const& t, int64_t align)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % align == 0 && (t.stride(0) * t.element_size()) % align == 0
        && (t.stride(1) * t.element_size()) % align == 0;
}

struct DeviceState
{
    bool configured = false;
    std::vector<torch::Tensor> held; // every workspace stays alive: captured graphs keep their pointers
};

std::mutex& stateMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<int, DeviceState>& states()
{
    static std::unordered_map<int, DeviceState> map;
    return map;
}

bool capturing(cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone;
}

//! The device's workspace of at least `need` bytes, configuring the device on first use. Both happen only outside
//! graph capture; inside it the call needs the reserve made by the prepare op.
void* ownTWorkspace(int device, size_t need, size_t* bytes, cudaStream_t stream)
{
    std::lock_guard<std::mutex> const lock(stateMutex());
    DeviceState& state = states()[device];
    bool const outside = !capturing(stream);
    if (!state.configured)
    {
        TORCH_CHECK(outside, "minimax_m3_sm107_sparse_decode_own_t: run minimax_m3_sm107_sparse_decode_own_t_prepare ",
            "on device ", device, " before graph capture");
        cudaError_t const err = TRTLLM_MV2_NS::kernels::minimaxM3Sm107SparseDecodeOwnTConfigure();
        TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_sparse_decode_own_t: this build has no OWN-T kernel for ",
            "device ", device, " (", cudaGetErrorString(err), ")");
        state.configured = true;
    }
    if (state.held.empty() || static_cast<size_t>(state.held.back().numel()) < need)
    {
        TORCH_CHECK(outside, "minimax_m3_sm107_sparse_decode_own_t: a captured call needs ", need,
            " workspace bytes, more than the prepare op reserved");
        state.held.push_back(
            torch::empty({static_cast<int64_t>(need)}, torch::dtype(torch::kUInt8).device(torch::kCUDA, device)));
    }
    *bytes = static_cast<size_t>(state.held.back().numel());
    return state.held.back().data_ptr();
}

int32_t mainGrid()
{
    return static_cast<int32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}

} // namespace

//! OWN-T: the MiniMax-M3 block-sparse decode of LC2's operands (8 query rows per request, 16 query heads per KV head,
//! head_dim 128, page 128, top-k 16) over the packed-NVFP4 paged cache, SM107. Within the float64-reference band of
//! LC2, not bitwise it.
//!
//! @param q E4M3 bytes [T, 16 * H, 128], unit stride in the last dim, rows 16-byte aligned.
//! @param k, v Packed E2M1 bytes [pages, H, 128, 64] (the per-layer HND views), 16-byte aligned.
//! @param kBlockScale, vBlockScale E4M3 bytes [pages, H, 128, 8]: K token-major, V in 4x4 token-quad order; 16-byte
//! aligned.
//! @param kGlobalScale, vGlobalScale One-element float32 per-tensor dequant scales.
//! @param topk Int32 [H, T, 16] selected block ids with -1 tail (the head-major selector table).
//! @param blockTable, seqLens Int32 [requests, max blocks] and [requests], T = 8 * requests.
//! @param out BF16 [T, H * 2048] with unit last-dim stride and 16-byte aligned rows, written.
//! @param q8, sf When given, receive out's MXFP8 copy: E4M3 rows [T, numSfKTiles * 128] and their UE8M0 scales in the
//! 128x4 swizzled layout of trtllm::mxfp8_quantize(out, true), bitwise that quantize (LC2's hand-off).
//! @param fastPlan DEP_OWNT_FASTPLAN: derive the union's fixed block weights in the plan (false: load them; same
//! split).
void minimaxM3Sm107SparseDecodeOwnT(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& kBlockScale, torch::Tensor const& vBlockScale, torch::Tensor const& kGlobalScale,
    torch::Tensor const& vGlobalScale, torch::Tensor const& topk, torch::Tensor const& blockTable,
    torch::Tensor const& seqLens, torch::Tensor& out, std::optional<torch::Tensor> q8, std::optional<torch::Tensor> sf,
    int64_t numSfKTiles, int64_t decodeQueryLen, double smScale, bool usePdl, bool fastPlan)
{
    TORCH_CHECK(q.is_cuda(), "minimax_m3_sm107_sparse_decode_own_t: q must be a CUDA tensor");
    c10::cuda::CUDAGuard const deviceGuard{q.device()};
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 10,
        "minimax_m3_sm107_sparse_decode_own_t requires an SM10x GPU");
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{
             &k, &v, &kBlockScale, &vBlockScale, &kGlobalScale, &vGlobalScale, &topk, &blockTable, &seqLens, &out})
    {
        TORCH_CHECK(t->device() == q.device(),
            "minimax_m3_sm107_sparse_decode_own_t: every operand must be on q's device (", q.device(), ")");
    }
    checkBytes(q, "q");
    checkBytes(k, "k");
    checkBytes(v, "v");
    checkBytes(kBlockScale, "k_block_scale");
    checkBytes(vBlockScale, "v_block_scale");
    TORCH_CHECK(decodeQueryLen == TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTRows,
        "minimax_m3_sm107_sparse_decode_own_t: decode_query_len must be ",
        TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTRows);
    TORCH_CHECK(k.dim() == 4 && k.size(2) == kPage && k.size(3) == kHeadDim / 2 && k.stride(3) == 1
            && k.stride(2) == kHeadDim / 2 && v.sizes() == k.sizes() && v.strides() == k.strides() && aligned(k, 16)
            && aligned(v, 16),
        "minimax_m3_sm107_sparse_decode_own_t: k and v must be 16-byte aligned [pages, H, 128, 64] with dense rows "
        "and matching strides");
    int64_t const kvHeads = k.size(1);
    TORCH_CHECK(kBlockScale.dim() == 4 && kBlockScale.size(1) == kvHeads && kBlockScale.size(2) == kPage
            && kBlockScale.size(3) == 8 && kBlockScale.stride(3) == 1 && kBlockScale.stride(2) == 8
            && kBlockScale.size(0) >= k.size(0) && vBlockScale.sizes() == kBlockScale.sizes()
            && vBlockScale.strides() == kBlockScale.strides() && aligned(kBlockScale, 16) && aligned(vBlockScale, 16),
        "minimax_m3_sm107_sparse_decode_own_t: block scales must be 16-byte aligned [pages, H, 128, 8] with dense "
        "rows and matching strides, covering every page of k / v");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeadsPerKv * kvHeads && q.size(2) == kHeadDim && q.stride(2) == 1
            && aligned(q, 16),
        "minimax_m3_sm107_sparse_decode_own_t: q must be [T, 16 * H, 128] with unit last-dim stride, 16-byte aligned");
    int64_t const rows = q.size(0);
    TORCH_CHECK(seqLens.scalar_type() == torch::kInt32 && seqLens.dim() == 1 && seqLens.stride(0) == 1
            && seqLens.size(0) * decodeQueryLen == rows,
        "minimax_m3_sm107_sparse_decode_own_t: seq_lens must be int32 [T / 8] with unit stride");
    int64_t const items = seqLens.size(0) * kvHeads;
    TORCH_CHECK(items >= 1 && items <= TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTMaxItems,
        "minimax_m3_sm107_sparse_decode_own_t: ", items, " (request, KV head) items, outside 1..",
        TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTMaxItems);
    TORCH_CHECK(kGlobalScale.scalar_type() == torch::kFloat32 && vGlobalScale.scalar_type() == torch::kFloat32
            && kGlobalScale.numel() == 1 && vGlobalScale.numel() == 1,
        "minimax_m3_sm107_sparse_decode_own_t: global scales must be one-element float32 tensors");
    TORCH_CHECK(topk.scalar_type() == torch::kInt32 && topk.dim() == 3 && topk.size(0) == kvHeads
            && topk.size(1) == rows && topk.size(2) == kTopK,
        "minimax_m3_sm107_sparse_decode_own_t: topk must be int32 [H, T, 16]");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32 && blockTable.dim() == 2 && blockTable.stride(1) == 1
            && blockTable.size(0) >= seqLens.size(0),
        "minimax_m3_sm107_sparse_decode_own_t: block_table must be int32 [>= T / 8, max blocks] with unit last-dim "
        "stride");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == rows
            && out.size(1) == kvHeads * kHeadsPerKv * kHeadDim && out.stride(1) == 1
            && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && out.stride(0) % 8 == 0,
        "minimax_m3_sm107_sparse_decode_own_t: out must be bfloat16 [T, H * 2048] with unit last-dim stride and "
        "16-byte aligned rows");
    TORCH_CHECK(
        q8.has_value() == sf.has_value(), "minimax_m3_sm107_sparse_decode_own_t: pass both q8 and sf, or neither");
    bool const mxfp8 = q8.has_value();
    if (mxfp8)
    {
        checkBytes(*q8, "q8");
        checkBytes(*sf, "sf");
        TORCH_CHECK(q8->device() == q.device() && sf->device() == q.device(),
            "minimax_m3_sm107_sparse_decode_own_t: q8 and sf must be on q's device (", q.device(), ")");
        TORCH_CHECK(numSfKTiles * 128 >= out.size(1) && q8->is_contiguous() && q8->numel() >= rows * numSfKTiles * 128
                && reinterpret_cast<uintptr_t>(q8->data_ptr()) % 4 == 0,
            "minimax_m3_sm107_sparse_decode_own_t: q8 must be contiguous 4-byte aligned bytes "
            "[T, num_sf_k_tiles * 128] covering out's row");
        TORCH_CHECK(sf->is_contiguous() && sf->numel() >= (rows + 127) / 128 * 128 * numSfKTiles * 4,
            "minimax_m3_sm107_sparse_decode_own_t: sf must be contiguous and hold the swizzled scales of whole "
            "128-row tiles");
    }

    TRTLLM_MV2_NS::kernels::MinimaxM3Sm107SparseDecodeOwnTParams prm{};
    prm.q = reinterpret_cast<uint8_t const*>(q.data_ptr());
    prm.q_tok = q.stride(0);
    prm.q_head = q.stride(1);
    prm.k = reinterpret_cast<uint8_t const*>(k.data_ptr());
    prm.v = reinterpret_cast<uint8_t const*>(v.data_ptr());
    prm.k_blk = k.stride(0);
    prm.v_blk = v.stride(0);
    prm.k_h = k.stride(1);
    prm.v_h = v.stride(1);
    prm.ksc = reinterpret_cast<uint8_t const*>(kBlockScale.data_ptr());
    prm.vsc = reinterpret_cast<uint8_t const*>(vBlockScale.data_ptr());
    prm.ksc_blk = kBlockScale.stride(0);
    prm.vsc_blk = vBlockScale.stride(0);
    prm.ksc_h = kBlockScale.stride(1);
    prm.vsc_h = vBlockScale.stride(1);
    prm.kgs = kGlobalScale.data_ptr<float>();
    prm.vgs = vGlobalScale.data_ptr<float>();
    prm.topk = topk.data_ptr<int>();
    prm.t_h = topk.stride(0);
    prm.t_tok = topk.stride(1);
    prm.t_k = topk.stride(2);
    prm.block_table = blockTable.data_ptr<int>();
    prm.bt_stride = blockTable.stride(0);
    prm.seq_lens = seqLens.data_ptr<int>();
    prm.dql = static_cast<int>(decodeQueryLen);
    prm.sm_scale = static_cast<float>(smScale);
    prm.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    prm.out_stride = out.stride(0);
    prm.num_kv_heads = static_cast<int>(kvHeads);
    prm.num_requests = static_cast<int>(seqLens.size(0));
    prm.q8 = mxfp8 ? reinterpret_cast<uint8_t*>(q8->data_ptr()) : nullptr;
    prm.sf = mxfp8 ? reinterpret_cast<uint8_t*>(sf->data_ptr()) : nullptr;
    prm.num_sf_k_tiles = static_cast<int>(numSfKTiles);

    auto const stream = at::cuda::getCurrentCUDAStream(q.get_device());
    int32_t const grid = mainGrid();
    size_t bytes = 0;
    void* const ws = ownTWorkspace(q.get_device(),
        TRTLLM_MV2_NS::kernels::minimaxM3Sm107SparseDecodeOwnTWorkspaceBytes(static_cast<int32_t>(items), grid), &bytes,
        stream);
    cudaError_t const err
        = TRTLLM_MV2_NS::kernels::invokeMinimaxM3Sm107SparseDecodeOwnT(prm, ws, bytes, grid, usePdl, stream, fastPlan);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_sparse_decode_own_t launch failed: ", cudaGetErrorString(err));
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true))
    {
        std::fprintf(stderr, "[M31 OWN_T] taken device=%d rows=%lld grid=%d workspace_bytes=%zu\n", q.get_device(),
            static_cast<long long>(rows), grid, bytes);
    }
}

//! Configure the device for OWN-T and reserve its workspace for `maxItems` (request, KV head) items, outside graph
//! capture and before the executor profiles memory and sizes its KV pool; returns the workspace bytes. Fails on a
//! build without an OWN-T kernel for the device's architecture.
int64_t prepareSparseDecodeOwnT(torch::Tensor const& anchor, int64_t maxItems)
{
    TORCH_CHECK(anchor.is_cuda(), "OWN-T workspace anchor must be CUDA");
    TORCH_CHECK(maxItems >= 1 && maxItems <= TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTMaxItems,
        "OWN-T reserves 1..", TRTLLM_MV2_NS::kernels::kMinimaxM3Sm107SparseDecodeOwnTMaxItems, " items");
    c10::cuda::CUDAGuard const guard(anchor.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 10, "OWN-T requires an SM10x GPU");
    auto const stream = at::cuda::getCurrentCUDAStream(anchor.get_device());
    TORCH_CHECK(!capturing(stream), "OWN-T workspace must be prepared outside graph capture");
    size_t bytes = 0;
    ownTWorkspace(anchor.get_device(),
        TRTLLM_MV2_NS::kernels::minimaxM3Sm107SparseDecodeOwnTWorkspaceBytes(
            static_cast<int32_t>(maxItems), mainGrid()),
        &bytes, stream);
    std::fprintf(stderr, "[M31 OWN_T] reserved device=%d workspace_bytes=%zu\n", anchor.get_device(), bytes);
    return static_cast<int64_t>(bytes);
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m3_sm107_sparse_decode_own_t_prepare(Tensor anchor, int max_items) -> int");
    m.def(
        "minimax_m3_sm107_sparse_decode_own_t(Tensor q, Tensor k, Tensor v, Tensor k_block_scale, "
        "Tensor v_block_scale, Tensor k_global_scale, Tensor v_global_scale, Tensor topk, Tensor block_table, "
        "Tensor seq_lens, Tensor(a!) out, Tensor(b!)? q8, Tensor(c!)? sf, int num_sf_k_tiles, int decode_query_len, "
        "float sm_scale, bool use_pdl, bool fast_plan=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107_sparse_decode_own_t_prepare", &TRTLLM_MV2_NS::torch_ext::prepareSparseDecodeOwnT);
    m.impl("minimax_m3_sm107_sparse_decode_own_t", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107SparseDecodeOwnT);
}
