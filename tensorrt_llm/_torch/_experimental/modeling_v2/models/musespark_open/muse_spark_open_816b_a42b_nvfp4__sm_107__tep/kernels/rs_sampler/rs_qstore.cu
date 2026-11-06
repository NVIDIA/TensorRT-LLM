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

// The q store of fused T = 1 rejection sampling (rejection_sampler; kernels/rs_sampler/rs_ops.py): the kernels
// that keep the q records (rs_common.cuh QRecord, one per spec-decode slot) coherent with the stock dense draft
// distribution `spec_metadata.draft_probs` [R, 7, Vd] when a step runs the stock path instead of the fused one.
//
// - rsInitKernel: every record to the empty state (ids -1, qf 0, draft -1, valid 0, kind kNone, flags 0).
// - rsWriteOnehotKernel: after the stock context one-hot (write_context_onehot_draft_probs), the record of each
//   context slot is kOnehot0: lane 0 holds id 0 with q 1 at every position, the drafts are the placeholder 0s.
// - rsSparsifyKernel: after the stock block draft sampler, the record of each generation slot is kFromDense: the
//   ids of position t are the entries of the selector output row (g, t) that are not -inf, in ascending id order
//   (the stock selector writes -inf everywhere but its 16 candidates, so the ids follow from the exchanged keys and
//   are the same on every rank), q is an exact copy of draft_probs[slot, t, id], the drafts are the stock sampler's
//   tokens. A position whose row does not hold exactly 16 such entries gets ids -1 and kPosQInvalid; one whose 16
//   values do not sum to 1 within 1e-5 (stock q with mass outside the ids) gets kPosMassOff | kPosQInvalid. A call
//   with structuralOk = false (the logits are not the selector output) writes kind kNone with valid 0.
// - rsDensifyKernel: before a stock rejection verify, every position t whose valid bit is set is consumed (bit t
//   cleared); if the record is kSampled (the fused walk, the only writer that leaves the dense row stale) the dense
//   row draft_probs[slot, t, 0 : storedVocab] is first rewritten as the scatter of the record's q (a kPosPoisoned
//   position: one-hot at the draft, the deterministic token the walk emitted). Other kinds were written next to
//   their dense row by the stock code, so their row is left as it is.
//
// All four are plain launches (no PDL), static shapes, no host synchronization: CUDA-graph safe. Records are
// written by plain stores with `valid` and `kind` last; their readers are later kernels on the same stream. Slots
// come from the stock slot table (int64); a slot outside [0, R) counts kSlotOob once per slot entry and is skipped,
// or traps with kCheckSlot under `check`. A slot may repeat within one call (every CUDA-graph padding request maps to
// the stock dummy slot row): densify reads a position's valid bit once per CTA, so a repeated slot ends with the same
// dense row and record as a single entry; sparsify / onehot leave a repeated slot's record as scratch.

#include "rs_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rs
{

constexpr int kInitThreads = kRecordBytes / 4;
constexpr int kOnehotThreads = 128;
constexpr int kRowThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kRowWarps = kRowThreads / kWarpSize;
//! Sparsify scans a selector row in tiles of kTileVectors x kRowThreads elements.
constexpr int kTileVectors = 4;
constexpr int64_t kTileElems = int64_t{kTileVectors} * kRowThreads;
constexpr uint32_t kNegInfBits = 0xFF800000u;
//! |sum(qf) - 1| above this makes a sparsified position kPosMassOff | kPosQInvalid.
constexpr float kMassTolerance = 1e-5f;
//! 32-bit words of a record that hold -1 after rsInitKernel: ids, then draft.
constexpr int kIdWordEnd = static_cast<int>(offsetof(QRecord, qf) / 4);
constexpr int kDraftWordBegin = static_cast<int>(offsetof(QRecord, draft) / 4);
constexpr int kDraftWordEnd = kDraftWordBegin + kPos;

static_assert(kInitThreads * 4 == kRecordBytes, "one word per init thread");
static_assert(kOnehotThreads >= kPos * kLanes, "one onehot thread per (position, lane)");
static_assert(kTileVectors * kRowWarps == kWarpSize * 4, "the tile scan gives 4 warp counts to each lane of warp 0");

struct SparsifyParams
{
    QRecord* records;
    int64_t numSlots;
    int64_t const* slots;
    //! Selector output row (g, t) at logits + g * logitsStrideG + t * logitsStrideT, `vocab` contiguous FP32.
    float const* logits;
    int64_t logitsStrideG;
    int64_t logitsStrideT;
    int64_t vocab;
    //! draft_probs row (slot, t) at probs + slot * probsStrideR + t * probsStrideT.
    float const* probs;
    int64_t probsStrideR;
    int64_t probsStrideT;
    int64_t storedVocab;
    //! The stock sampler's tokens [G, >= 7], int64 or int32.
    void const* tokens;
    int64_t tokensStrideG;
    int64_t tokensStrideT;
    bool tokens64;
    bool structuralOk;
    bool check;
    unsigned long long* diag;
};

struct DensifyParams
{
    QRecord* records;
    int64_t numSlots;
    int64_t const* slots;
    float* probs;
    int64_t probsStrideR;
    int64_t probsStrideT;
    int64_t storedVocab;
    bool check;
    unsigned long long* diag;
};

__device__ __forceinline__ bool slotInRange(int64_t slot, int64_t numSlots)
{
    return slot >= 0 && slot < numSlots;
}

__device__ __forceinline__ void slotOutOfRange(unsigned long long* diag, int64_t slot, bool check)
{
    diagAdd(diag, Diag::kSlotOob);
    if (check)
    {
        trap(diag, TrapCode::kCheckSlot, slot);
    }
}

//! Every record to the empty state. Grid: records (grid-stride), block: one thread per 32-bit word.
__global__ void __launch_bounds__(kInitThreads) rsInitKernel(QRecord* records, int64_t numSlots)
{
    int const w = threadIdx.x;
    uint32_t const value = (w < kIdWordEnd || (w >= kDraftWordBegin && w < kDraftWordEnd)) ? 0xFFFFFFFFu : 0u;
    for (int64_t r = blockIdx.x; r < numSlots; r += gridDim.x)
    {
        reinterpret_cast<uint32_t*>(records + r)[w] = value;
    }
}

//! The kOnehot0 record of slots[blockIdx.x].
__global__ void __launch_bounds__(kOnehotThreads)
    rsWriteOnehotKernel(QRecord* records, int64_t numSlots, int64_t const* slots, bool check, unsigned long long* diag)
{
    int const tid = threadIdx.x;
    int64_t const slot = slots[blockIdx.x];
    if (!slotInRange(slot, numSlots))
    {
        if (tid == 0)
        {
            slotOutOfRange(diag, slot, check);
        }
        return;
    }
    QRecord& rec = records[slot];
    if (tid < kPos * kLanes)
    {
        int const t = tid / kLanes;
        int const k = tid % kLanes;
        rec.ids[t][k] = k == 0 ? 0 : -1;
        rec.qf[t][k] = k == 0 ? 1.0f : 0.0f;
    }
    if (tid < kPos)
    {
        rec.draft[tid] = 0;
        rec.posFlags[tid] = 0;
    }
    if (tid == 0)
    {
        rec.mode = 0;
    }
    __syncthreads();
    if (tid == 0)
    {
        rec.kind = static_cast<uint8_t>(Kind::kOnehot0);
        rec.valid = kValidAll;
        diagAdd(diag, Diag::kOnehotWrites);
    }
}

__device__ __forceinline__ int64_t sparsifyToken(SparsifyParams const& p, int g, int t)
{
    int64_t const at = g * p.tokensStrideG + t * p.tokensStrideT;
    return p.tokens64 ? static_cast<int64_t const*>(p.tokens)[at] : static_cast<int32_t const*>(p.tokens)[at];
}

//! Position blockIdx.y of the kFromDense record of slots[blockIdx.x]; the CTA of position 0 also writes the drafts,
//! mode, kind and valid.
__global__ void __launch_bounds__(kRowThreads) rsSparsifyKernel(SparsifyParams const p)
{
    __shared__ int32_t sIds[kLanes];
    __shared__ float sQ[kLanes];
    __shared__ uint32_t sPrefix[kTileVectors * kRowWarps];
    __shared__ int32_t sTotal;

    int const g = blockIdx.x;
    int const t = blockIdx.y;
    int const tid = threadIdx.x;
    int64_t const slot = p.slots[g];
    if (!slotInRange(slot, p.numSlots))
    {
        if (t == 0 && tid == 0)
        {
            slotOutOfRange(p.diag, slot, p.check);
        }
        return;
    }
    QRecord& rec = p.records[slot];

    // Ascending compaction of the entries that are not -inf: per tile, a ballot per (vector j, warp), an exclusive
    // scan of the 128 warp counts in element order (index j * kRowWarps + warp) by warp 0, then each flagged element
    // lands at cnt + its warp's prefix + its rank in the ballot. Deterministic, no atomics; stops once the count
    // exceeds kLanes (the position is then not a selector row).
    int cnt = 0;
    if (p.structuralOk)
    {
        float const* row = p.logits + g * p.logitsStrideG + t * p.logitsStrideT;
        int const warp = tid / kWarpSize;
        int const lane = tid % kWarpSize;
        uint32_t const lowerLanes = (1u << lane) - 1u;
        for (int64_t base = 0; base < p.vocab && cnt <= kLanes; base += kTileElems)
        {
            uint32_t masks[kTileVectors];
            bool flags[kTileVectors];
#pragma unroll
            for (int j = 0; j < kTileVectors; ++j)
            {
                int64_t const id = base + j * kRowThreads + tid;
                flags[j] = id < p.vocab && __float_as_uint(row[id]) != kNegInfBits;
                masks[j] = __ballot_sync(0xFFFFFFFFu, flags[j]);
            }
            if (lane == 0)
            {
#pragma unroll
                for (int j = 0; j < kTileVectors; ++j)
                {
                    sPrefix[j * kRowWarps + warp] = __popc(masks[j]);
                }
            }
            __syncthreads();
            if (warp == 0)
            {
                uint32_t c[4];
                uint32_t sum = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    c[i] = sPrefix[4 * lane + i];
                    sum += c[i];
                }
                uint32_t inclusive = sum;
#pragma unroll
                for (int d = 1; d < kWarpSize; d <<= 1)
                {
                    uint32_t const other = __shfl_up_sync(0xFFFFFFFFu, inclusive, d);
                    inclusive += lane >= d ? other : 0u;
                }
                uint32_t running = inclusive - sum;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sPrefix[4 * lane + i] = running;
                    running += c[i];
                }
                if (lane == kWarpSize - 1)
                {
                    sTotal = static_cast<int32_t>(inclusive);
                }
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < kTileVectors; ++j)
            {
                if (flags[j])
                {
                    int const pos
                        = cnt + static_cast<int>(sPrefix[j * kRowWarps + warp] + __popc(masks[j] & lowerLanes));
                    if (pos < kLanes)
                    {
                        sIds[pos] = static_cast<int32_t>(base + j * kRowThreads + tid);
                    }
                }
            }
            cnt += sTotal;
            __syncthreads();
        }
    }
    bool const ok = p.structuralOk && cnt == kLanes;

    if (tid < kLanes)
    {
        int32_t const id = ok ? sIds[tid] : -1;
        float const q = (ok && id < p.storedVocab) ? p.probs[slot * p.probsStrideR + t * p.probsStrideT + id] : 0.0f;
        rec.ids[t][tid] = id;
        rec.qf[t][tid] = q;
        sQ[tid] = q;
    }
    __syncthreads();
    if (tid == 0)
    {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kLanes; ++k)
        {
            sum = __fadd_rn(sum, sQ[k]);
        }
        bool const massOff = ok && !(fabsf(__fsub_rn(sum, 1.0f)) <= kMassTolerance);
        rec.posFlags[t] = !ok ? kPosQInvalid : (massOff ? static_cast<uint8_t>(kPosMassOff | kPosQInvalid) : 0);
        if (!ok)
        {
            diagAdd(p.diag, Diag::kSparsifyNone);
        }
        if (massOff)
        {
            diagAdd(p.diag, Diag::kMassOff);
        }
    }
    if (t == 0)
    {
        if (tid < kPos)
        {
            rec.draft[tid] = static_cast<int32_t>(sparsifyToken(p, g, tid));
        }
        __syncthreads();
        if (tid == 0)
        {
            rec.mode = 0;
            rec.kind = static_cast<uint8_t>(p.structuralOk ? Kind::kFromDense : Kind::kNone);
            rec.valid = p.structuralOk ? kValidAll : 0u;
            diagAdd(p.diag, Diag::kSparsifySlots);
        }
    }
}

//! Position blockIdx.y of slots[blockIdx.x] before a stock rejection verify: scatter a kSampled record's q into the
//! dense row, then consume the position.
__global__ void __launch_bounds__(kRowThreads) rsDensifyKernel(DensifyParams const p)
{
    int const g = blockIdx.x;
    int const t = blockIdx.y;
    int const tid = threadIdx.x;
    int64_t const slot = p.slots[g];
    if (!slotInRange(slot, p.numSlots))
    {
        if (t == 0 && tid == 0)
        {
            slotOutOfRange(p.diag, slot, p.check);
        }
        return;
    }
    QRecord& rec = p.records[slot];
    // Another CTA of the same position clears bit t when the slot repeats in this call: thread 0 reads the bit (and
    // the kind) once for the CTA, so every thread takes the same branch to the barriers below.
    __shared__ uint32_t sTake;
    if (tid == 0)
    {
        sTake = ((rec.valid >> t) & 1u) == 0 ? 0u : (rec.kind == static_cast<uint8_t>(Kind::kSampled) ? 2u : 1u);
    }
    __syncthreads();
    uint32_t const take = sTake;
    if (take == 0)
    {
        return;
    }
    if (take == 2)
    {
        float* row = p.probs + slot * p.probsStrideR + t * p.probsStrideT;
        int64_t done = 0;
        if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0)
        {
            int64_t const vecs = p.storedVocab / 4;
            float4 const zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int64_t i = tid; i < vecs; i += kRowThreads)
            {
                reinterpret_cast<float4*>(row)[i] = zero;
            }
            done = vecs * 4;
        }
        for (int64_t i = done + tid; i < p.storedVocab; i += kRowThreads)
        {
            row[i] = 0.0f;
        }
        __syncthreads();
        if ((rec.posFlags[t] & kPosPoisoned) != 0)
        {
            if (tid == 0)
            {
                int32_t const d = rec.draft[t];
                if (d >= 0 && d < p.storedVocab)
                {
                    row[d] = 1.0f;
                }
            }
        }
        else if (tid < kLanes)
        {
            int32_t const id = rec.ids[t][tid];
            if (id >= 0 && id < p.storedVocab)
            {
                row[id] = rec.qf[t][tid];
            }
        }
        if (tid == 0)
        {
            diagAdd(p.diag, Diag::kDensifyRows);
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        atomicAnd(&rec.valid, ~(1u << t));
        if (t == 0)
        {
            diagAdd(p.diag, Diag::kDensifyCalls);
        }
    }
}

} // namespace rs

namespace
{

void checkRecords(at::Tensor const& records)
{
    TORCH_CHECK(records.is_cuda() && records.scalar_type() == at::kByte && records.dim() == 2
            && records.size(1) == rs::kRecordBytes && records.is_contiguous(),
        "rs: records must be a contiguous uint8 [R, ", rs::kRecordBytes, "] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(records.data_ptr()) % alignof(rs::QRecord) == 0,
        "rs: records must be 16-byte aligned");
}

void checkSlots(at::Tensor const& slots, at::Tensor const& records)
{
    TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kLong && slots.dim() == 1 && slots.is_contiguous()
            && slots.device() == records.device(),
        "rs: slots must be a contiguous int64 1-D tensor on the records' device");
    TORCH_CHECK(slots.size(0) <= 65535, "rs: at most 65535 slots per call");
}

void checkDiag(at::Tensor const& diag, at::Tensor const& records)
{
    TORCH_CHECK(diag.is_cuda() && diag.scalar_type() == at::kLong && diag.is_contiguous()
            && diag.numel() >= rs::kDiagWords && diag.device() == records.device(),
        "rs: diag must be a contiguous int64 [", rs::kDiagWords, "] tensor on the records' device");
}

void checkProbs(at::Tensor const& probs, at::Tensor const& records, int64_t storedVocab)
{
    TORCH_CHECK(probs.is_cuda() && probs.scalar_type() == at::kFloat && probs.dim() == 3 && probs.stride(2) == 1
            && probs.device() == records.device(),
        "rs: probs must be an FP32 [R, >= ", rs::kPos, ", Vd] tensor with a contiguous last dimension");
    TORCH_CHECK(probs.size(0) >= records.size(0) && probs.size(1) >= rs::kPos, "rs: probs must hold a [", rs::kPos,
        ", Vd] row for every record");
    TORCH_CHECK(storedVocab >= 1 && storedVocab <= probs.size(2), "rs: stored_vocab must be in [1, Vd]");
}

unsigned long long* diagPtr(at::Tensor const& diag)
{
    return reinterpret_cast<unsigned long long*>(diag.data_ptr());
}

rs::QRecord* recordsPtr(at::Tensor const& records)
{
    return reinterpret_cast<rs::QRecord*>(records.data_ptr());
}

void init(at::Tensor const& records)
{
    at::cuda::CUDAGuard guard(records.device());
    checkRecords(records);
    int64_t const numSlots = records.size(0);
    if (numSlots == 0)
    {
        return;
    }
    int const grid = static_cast<int>(std::min<int64_t>(numSlots, 65535));
    rs::rsInitKernel<<<grid, rs::kInitThreads, 0, at::cuda::getCurrentCUDAStream()>>>(recordsPtr(records), numSlots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void writeOnehot(at::Tensor const& records, at::Tensor const& slots, at::Tensor const& diag, bool check)
{
    at::cuda::CUDAGuard guard(records.device());
    checkRecords(records);
    checkSlots(slots, records);
    checkDiag(diag, records);
    int64_t const n = slots.size(0);
    if (n == 0)
    {
        return;
    }
    rs::rsWriteOnehotKernel<<<static_cast<int>(n), rs::kOnehotThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        recordsPtr(records), records.size(0), slots.data_ptr<int64_t>(), check, diagPtr(diag));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparsify(at::Tensor const& records, at::Tensor const& slots, at::Tensor const& logits, at::Tensor const& probs,
    int64_t storedVocab, at::Tensor const& tokens, bool structuralOk, at::Tensor const& diag, bool check)
{
    at::cuda::CUDAGuard guard(records.device());
    checkRecords(records);
    checkSlots(slots, records);
    checkDiag(diag, records);
    checkProbs(probs, records, storedVocab);
    int64_t const n = slots.size(0);
    TORCH_CHECK(tokens.is_cuda() && (tokens.scalar_type() == at::kLong || tokens.scalar_type() == at::kInt)
            && tokens.dim() == 2 && tokens.size(0) >= n && tokens.size(1) >= rs::kPos
            && tokens.device() == records.device(),
        "rs sparsify: tokens must be an int64 or int32 [>= G, >= ", rs::kPos, "] tensor on the records' device");
    rs::SparsifyParams p{};
    if (structuralOk)
    {
        TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 3 && logits.size(0) >= n
                && logits.size(1) >= rs::kPos && logits.size(2) >= 1 && logits.stride(2) == 1
                && logits.device() == records.device(),
            "rs sparsify: with structural_ok, logits must be an FP32 [>= G, >= ", rs::kPos,
            ", V] tensor with a contiguous last dimension");
        TORCH_CHECK(logits.size(2) <= INT32_MAX, "rs sparsify: ids are int32");
        p.logits = logits.data_ptr<float>();
        p.logitsStrideG = logits.stride(0);
        p.logitsStrideT = logits.stride(1);
        p.vocab = logits.size(2);
    }
    if (n == 0)
    {
        return;
    }
    p.records = recordsPtr(records);
    p.numSlots = records.size(0);
    p.slots = slots.data_ptr<int64_t>();
    p.probs = probs.data_ptr<float>();
    p.probsStrideR = probs.stride(0);
    p.probsStrideT = probs.stride(1);
    p.storedVocab = storedVocab;
    p.tokens = tokens.data_ptr();
    p.tokensStrideG = tokens.stride(0);
    p.tokensStrideT = tokens.stride(1);
    p.tokens64 = tokens.scalar_type() == at::kLong;
    p.structuralOk = structuralOk;
    p.check = check;
    p.diag = diagPtr(diag);
    rs::rsSparsifyKernel<<<dim3(static_cast<unsigned>(n), rs::kPos), rs::kRowThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void densify(at::Tensor const& records, at::Tensor const& slots, at::Tensor const& probs, int64_t storedVocab,
    at::Tensor const& diag, bool check)
{
    at::cuda::CUDAGuard guard(records.device());
    checkRecords(records);
    checkSlots(slots, records);
    checkDiag(diag, records);
    checkProbs(probs, records, storedVocab);
    int64_t const n = slots.size(0);
    if (n == 0)
    {
        return;
    }
    rs::DensifyParams p{};
    p.records = recordsPtr(records);
    p.numSlots = records.size(0);
    p.slots = slots.data_ptr<int64_t>();
    p.probs = probs.data_ptr<float>();
    p.probsStrideR = probs.stride(0);
    p.probsStrideT = probs.stride(1);
    p.storedVocab = storedVocab;
    p.check = check;
    p.diag = diagPtr(diag);
    rs::rsDensifyKernel<<<dim3(static_cast<unsigned>(n), rs::kPos), rs::kRowThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void appendAttributes(std::vector<int64_t>& out, void const* kernel)
{
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    out.push_back(attr.numRegs);
    out.push_back(static_cast<int64_t>(attr.localSizeBytes));
    out.push_back(static_cast<int64_t>(attr.sharedSizeBytes));
    out.push_back(attr.maxThreadsPerBlock);
}

//! Resolves every q-store kernel on `anchor`'s device (so nothing loads during a CUDA-graph capture) and returns
//! (numRegs, localSizeBytes, sharedSizeBytes, maxThreadsPerBlock) of rsInitKernel, rsWriteOnehotKernel,
//! rsSparsifyKernel, rsDensifyKernel, in that order.
std::vector<int64_t> kernelAttributes(at::Tensor const& anchor)
{
    at::cuda::CUDAGuard guard(anchor.device());
    std::vector<int64_t> out;
    appendAttributes(out, reinterpret_cast<void const*>(&rs::rsInitKernel));
    appendAttributes(out, reinterpret_cast<void const*>(&rs::rsWriteOnehotKernel));
    appendAttributes(out, reinterpret_cast<void const*>(&rs::rsSparsifyKernel));
    appendAttributes(out, reinterpret_cast<void const*>(&rs::rsDensifyKernel));
    return out;
}

} // namespace

TORCH_LIBRARY(musespark_open_rs, m)
{
    m.def("init(Tensor(a!) records) -> ()");
    m.def("write_onehot(Tensor(a!) records, Tensor slots, Tensor(b!) diag, bool check=False) -> ()");
    m.def(
        "sparsify(Tensor(a!) records, Tensor slots, Tensor logits, Tensor probs, int stored_vocab, Tensor tokens, "
        "bool structural_ok, Tensor(b!) diag, bool check=False) -> ()");
    m.def(
        "densify(Tensor(a!) records, Tensor slots, Tensor(b!) probs, int stored_vocab, Tensor(c!) diag, "
        "bool check=False) -> ()");
    m.def("kernel_attributes(Tensor anchor) -> int[]");
}

TORCH_LIBRARY_IMPL(musespark_open_rs, CUDA, m)
{
    m.impl("init", &init);
    m.impl("write_onehot", &writeOnehot);
    m.impl("sparsify", &sparsify);
    m.impl("densify", &densify);
    m.impl("kernel_attributes", &kernelAttributes);
}
