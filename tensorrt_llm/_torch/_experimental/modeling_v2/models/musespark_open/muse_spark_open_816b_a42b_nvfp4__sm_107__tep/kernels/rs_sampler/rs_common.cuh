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

// Fused T = 1 rejection sampling (rejection_sampler): what the draft-side walk (dr_draft_tail.cu), the verify-side
// fold and draw (s2_sampler.cu) and the q-store kernels (rs_qstore.cu) share. rs_common.py mirrors every constant
// and offset here; the rs_sampler unit test checks the mirror against this header.
//
// The q record (QRecord, 1024 bytes, one per spec-decode slot, row `slot` of rs.records [R, 1024] uint8):
//   ids[t][k]   int32  candidate id of lane k at draft position t (-1 = empty lane); distinct within a position
//   qf[t][k]    fp32   q_t(ids[t][k]), the probability the draft at position t was drawn with (0 for empty lanes)
//   draft[t]    int32  the draft written to next_tokens[1 + t] (-1 = none)
//   valid       bit t: position t was written and not yet consumed. A writer sets kValidAll; the consumer of
//               position t (the verify that reads it) clears bit t, so a record serves at most one verify and a
//               producer that does not rewrite it leaves a strict row behind.
//   kind        Kind; mode: bit 0 chain sampled, bits 1..2 Transform (informational); posFlags[t]: kPos* flags.
// Writers store whole positions, then `valid` and `kind` last from one thread; readers are later kernels on the
// stream. Protocol words (rank-uniform by construction): ids, kind, valid. Value words: qf, posFlags, mode, draft.
//
// Randomness (Philox4x32-10, curand's generator): Ph(seed, S, n) is the uint4 curand_init(seed, S, 4n, &st) +
// curand4(&st) returns, i.e. key (lo32 seed, hi32 seed), counter (lo32 n, hi32 n, lo32 S, hi32 S). Subsequences of a
// request's step window (row j at off[8b + j] = B + 32 j): draft position t uses draftSubseq(off[8b], t) word .x;
// the race noise of row j uses raceSubseq(off[8b + j]) with counter 2v (hi words) and 2v + 1 (lo words) for global
// vector v = id >> 2, element id & 3 (logE); the accept test of row j uses acceptSubseq(off[8b + j]) word .x (unif),
// and a filtered row j (s2_sampler.cu rsFilteredRow) takes its residual / plain inverse-CDF uniform u_r from word .y of
// the same draw. Words .z / .w of acceptSubseq are unused.
//
// Floating point: every statement is spelled with __f*_rn / __fmaf_rn so -fmad=true cannot contract it, and logf /
// log1pf are the accurate library functions. The exponential race keys and the replay tests depend on both: never
// build a library that includes this header with --use_fast_math (nvcc defines no macro for it, so the header cannot
// refuse it; the unit test builds its probe with the consumers' flags and checks expo / logE against float64).

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rs
{

// ---- the q record ---------------------------------------------------------------------------------------------------

//! Draft positions per record (K = 7) and candidate lanes per position.
constexpr int kPos = 7;
constexpr int kLanes = 16;
constexpr int kRecordBytes = 1024;
//! `valid` after a writer: one bit per position.
constexpr uint32_t kValidAll = (1u << kPos) - 1u;

//! QRecord::kind. kSampled (the fused walk), kOnehot0 (the context placeholder drafts), kFromDense (the stock draft
//! sampler's dense q, sparsified) are the usable kinds.
enum class Kind : uint8_t
{
    kNone = 0,
    kSampled = 1,
    kOnehot0 = 2,
    kFromDense = 3,
};

__host__ __device__ __forceinline__ constexpr bool isUsableKind(uint8_t kind)
{
    return kind == static_cast<uint8_t>(Kind::kSampled) || kind == static_cast<uint8_t>(Kind::kOnehot0)
        || kind == static_cast<uint8_t>(Kind::kFromDense);
}

//! QRecord::posFlags[t] bits. kPosPoisoned: the walk's scores were not finite, q is one-hot at the greedy token (still
//! usable). kPosQInvalid: q of this position must not be used (strict row). kPosMassOff: the sparsified stock q has
//! |sum(qf) - 1| > 1e-5 (always set together with kPosQInvalid). kPosFiltered: q of this position is the request's
//! top-k / top-p / min-p filtered law over the 16 candidates (informational; part of hQ).
constexpr uint8_t kPosPoisoned = 1;
constexpr uint8_t kPosQInvalid = 2;
constexpr uint8_t kPosMassOff = 4;
constexpr uint8_t kPosFiltered = 8;

//! q transform of the walk (rs.py Knobs.q_transform, unary), stored in QRecord::mode bits 1..2. kRaw: q
//! is the softmax of the walk score R = unary + edge; kScore: of f(R); kUnary: the walk itself runs on f(unary) + edge,
//! the draft's own sampling law (f(x) = cap tanh(x m / cap), or x m when cap = 0). With kRaw or kScore the walk's
//! greedy path is the untransformed walk's; with kUnary it is the greedy path of f(unary) + edge, which may differ.
enum class Transform : uint8_t
{
    kRaw = 0,
    kScore = 1,
    kUnary = 2,
};

//! Filtered rows of a request with top-k / top-p / min-p (rs.py Knobs.filter, agreed across the TP group).
//! kExact: exact filtered rejection sampling against the filtered target and draft laws; kStrict: the strict body
//! (acceptance against the target's filtered sample).
enum class FilterMode : int
{
    kExact = 0,
    kStrict = 1,
};

//! QRecord::mode: bit 0 = the chain follows the sampled lane (Knobs.q_chain sampled, as installed), bits
//! 1..2 the Transform.
constexpr uint8_t kModeChainSampled = 1;
constexpr int kModeTransformShift = 1;

__host__ __device__ __forceinline__ constexpr uint8_t modeByte(bool chainSampled, Transform transform)
{
    return static_cast<uint8_t>(
        (chainSampled ? kModeChainSampled : 0u) | (static_cast<uint32_t>(transform) << kModeTransformShift));
}

struct alignas(16) QRecord
{
    int32_t ids[kPos][kLanes]; //    0
    float qf[kPos][kLanes];    //  448
    int32_t draft[kPos];       //  896
    uint32_t valid;            //  924
    uint8_t kind;              //  928
    uint8_t mode;              //  929
    uint8_t posFlags[kPos];    //  930
    uint8_t pad0;              //  937 (938 .. 939: alignment of reserved)
    uint32_t reserved[21];     //  940 .. 1023
};

static_assert(std::is_standard_layout<QRecord>::value, "QRecord is addressed by byte offsets");
static_assert(sizeof(QRecord) == kRecordBytes, "QRecord size");
static_assert(alignof(QRecord) == 16, "QRecord alignment");
static_assert(offsetof(QRecord, ids) == 0, "QRecord::ids");
static_assert(offsetof(QRecord, qf) == 448, "QRecord::qf");
static_assert(offsetof(QRecord, draft) == 896, "QRecord::draft");
static_assert(offsetof(QRecord, valid) == 924, "QRecord::valid");
static_assert(offsetof(QRecord, kind) == 928, "QRecord::kind");
static_assert(offsetof(QRecord, mode) == 929, "QRecord::mode");
static_assert(offsetof(QRecord, posFlags) == 930, "QRecord::posFlags");
static_assert(offsetof(QRecord, pad0) == 937, "QRecord::pad0");
static_assert(offsetof(QRecord, reserved) == 940, "QRecord::reserved");

// ---- diagnostics (rs.diag, int64 [kDiagWords], rank-local, atomicAdd) -----------------------------------------------

enum class Diag : int
{
    kRsRows = 0, // neutral rows t < 7 decided by the RS draw
    kAccepts = 1,
    kRejects = 2,
    kStrictKind = 3,      // not an RS row: kind / valid bit
    kStrictDrafts = 4,    // draftIn != record drafts
    kStrictKd = 5,        // the draft is not among the ids
    kStrictDup = 6,       // duplicate ids or an id out of range
    kStrictQzero = 7,     // qf[kd] == 0
    kStrictPoison = 8,    // M / Z / w not finite
    kStrictQdiv = 9,      // hQ differs across ranks
    kStrictQinvalid = 10, // posFlags
    kFilteredRows = 11,
    kDraftPoisoned = 12,
    kDensifyRows = 13, // densify scatters
    kSparsifySlots = 14,
    kMassOff = 15,
    kOnehotWrites = 16,
    kSlotOob = 17,
    kDummyRows = 18,
    kWalkRsCalls = 19,
    kRsDrawCalls = 20,    // CTA 7 of each RS draw
    kDensifyCalls = 21,
    kSparsifyNone = 22,   // selector structure is not 16 ids
    kConsumedUnused = 23, // valid bit cleared on a strict / non-RS row
    kWalkDupIds = 24,
    kTrapRankHint = 29,
    kTrapCode = 30, // written before every rs::trap
    kTrapDetail = 31,
    // Filtered rows (top-k / top-p / min-p). kFilteredRows (11) counts every filtered row t < 7 of an RS draw. The
    // words marked "owner" count on the rank that owns the row's list; every other word is equal on all ranks.
    kFiltRsRows = 32,         // filtered rows t < 7 decided by exact filtered RS
    kFiltAccepts = 33,        //
    kFiltRejects = 34,        //
    kFiltDUnkept = 35,        // the draft is not in the filtered target support (a rejection)
    kFiltPlainRows = 36,      // filtered rows drawn from the filtered target alone (row 7, unusable q)
    kFiltOwnerRows = 37,      // owner
    kFiltListEntries = 38,    // owner: entries of the gathered list
    kFiltListMax = 39,        // owner: longest list (diagMax)
    kFiltScreenFallback = 40, // owner: the screened gather kept fewer than top-k entries and ran again unscreened
    kFiltResidFallback = 41,  // a rounding fallback of the residual draw
    kWalkFilteredPositions = 42,
    kWalkFilterRemovedLanes = 43,
    kFiltStrictLastResort = 44, // a filtered row the exact path could not decide, taken by the strict body
    kFiltStrictRows = 45,       // filtered rows taken by the strict body (Knobs.filter strict)
};

constexpr int kDiagWords = 48;

//! Codes written to diag[kTrapCode] before a trap.
enum class TrapCode : int
{
    kClaims = 1,       // candidate claims inconsistent (ids differ across ranks, owner count != 1, claim != own id)
    kDigestD = 2,      // hD differs across ranks
    kCheckStrict = 3,  // Knobs.check: a strict counter on a non-dummy slot
    kCheckReread = 4,  // Knobs.check: the walk's post-wait re-read differs
    kCheckSlot = 5,    // Knobs.check: slot out of range
    kCheckDupIds = 6,  // Knobs.check: the walk saw duplicate ids
    kFilterParams = 7, // a filtered row's sampling parameters differ from its owner's (detail = row)
};

__device__ __forceinline__ void diagAdd(unsigned long long* diag, Diag index, unsigned long long count = 1)
{
    atomicAdd(diag + static_cast<int>(index), count);
}

__device__ __forceinline__ void diagMax(unsigned long long* diag, Diag index, unsigned long long value)
{
    atomicMax(diag + static_cast<int>(index), value);
}

//! Record the code and a detail word (e.g. the row) in diag, make them visible, and trap.
__device__ __forceinline__ void trap(unsigned long long* diag, TrapCode code, long long detail)
{
    diag[static_cast<int>(Diag::kTrapCode)] = static_cast<unsigned long long>(static_cast<int>(code));
    diag[static_cast<int>(Diag::kTrapDetail)] = static_cast<unsigned long long>(detail);
    __threadfence_system();
    __trap();
}

// ---- fabric words ---------------------------------------------------------------------------------------------------

//! Candidate-slot claim word: kClaimOwner | id from the owner of id, the id from a non-owner, kClaimEmpty for id < 0.
constexpr uint32_t kClaimOwner = 0x80000000u;
constexpr uint32_t kClaimEmpty = 0x7FFFFFFFu;

__host__ __device__ __forceinline__ constexpr uint32_t claimWord(int32_t id, bool owner)
{
    return id < 0 ? kClaimEmpty : (owner ? kClaimOwner | static_cast<uint32_t>(id) : static_cast<uint32_t>(id));
}

// ---- hash (record digests) ------------------------------------------------------------------------------------------

constexpr uint32_t kHashInit = 0x9E3779B9u;

//! murmur3's 32-bit finalizer.
__host__ __device__ __forceinline__ constexpr uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

//! One step of H: H(w_0 .. w_{n-1}) = hashStep(... hashStep(kHashInit, w_0) ..., w_{n-1}).
__host__ __device__ __forceinline__ constexpr uint32_t hashStep(uint32_t h, uint32_t word)
{
    return fmix32(h ^ word);
}

__host__ __device__ __forceinline__ uint32_t hashWords(uint32_t const* words, int n, uint32_t h = kHashInit)
{
    for (int i = 0; i < n; ++i)
    {
        h = hashStep(h, words[i]);
    }
    return h;
}

//! First word of hD(t): kind | mode << 8 | valid bit t << 16 | t << 24 (followed by draft[0..6], draftIn[0..6]);
//! hQ(t) = H(bits(qf[t][0..15]), posFlags[t]).
__host__ __device__ __forceinline__ constexpr uint32_t digestHead(uint8_t kind, uint8_t mode, uint32_t validBit, int t)
{
    return static_cast<uint32_t>(kind) | static_cast<uint32_t>(mode) << 8 | (validBit & 1u) << 16
        | static_cast<uint32_t>(t) << 24;
}

// ---- RNG ------------------------------------------------------------------------------------------------------------

//! Bit 63 marks the RS subsequences (disjoint from the fused sampler's subsequence 0 and flashinfer's).
constexpr uint64_t kSubseqFlag = 1ull << 63;
//! Subsequences per verify row in a request's step window, and the first draft slot (stock rng_draft_slot(t) = 8 + t).
constexpr uint64_t kRowStride = 32;
constexpr uint64_t kDraftSlot0 = 8;
constexpr uint64_t kRaceSub = 1;
constexpr uint64_t kAcceptSub = 2;
//! Cap of an exponential variate (E = 23 only for u = 1).
constexpr float kExpoMax = 23.0f;

__host__ __device__ __forceinline__ constexpr uint64_t draftSubseq(int64_t requestOffset, int t)
{
    return kSubseqFlag | (static_cast<uint64_t>(requestOffset) + kRowStride * (kDraftSlot0 + static_cast<uint64_t>(t)));
}

__host__ __device__ __forceinline__ constexpr uint64_t raceSubseq(int64_t rowOffset)
{
    return kSubseqFlag | (static_cast<uint64_t>(rowOffset) + kRaceSub);
}

__host__ __device__ __forceinline__ constexpr uint64_t acceptSubseq(int64_t rowOffset)
{
    return kSubseqFlag | (static_cast<uint64_t>(rowOffset) + kAcceptSub);
}

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

__device__ __forceinline__ uint4 philoxRound(uint4 c, uint32_t k0, uint32_t k1)
{
    uint32_t const hi0 = __umulhi(kPhiloxM0, c.x);
    uint32_t const lo0 = kPhiloxM0 * c.x;
    uint32_t const hi1 = __umulhi(kPhiloxM1, c.z);
    uint32_t const lo1 = kPhiloxM1 * c.z;
    return make_uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
}

//! Philox4x32-10 at key (lo32 seed, hi32 seed), counter (lo32 n, hi32 n, lo32 S, hi32 S): what
//! curand_init(seed, S, 4 n, &st) followed by curand4(&st) returns.
__device__ __forceinline__ uint4 Ph(uint64_t seed, uint64_t subseq, uint64_t n)
{
    uint4 c = make_uint4(static_cast<uint32_t>(n), static_cast<uint32_t>(n >> 32), static_cast<uint32_t>(subseq),
        static_cast<uint32_t>(subseq >> 32));
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
#pragma unroll
    for (int round = 0; round < 9; ++round)
    {
        c = philoxRound(c, k0, k1);
        k0 += kPhiloxW0;
        k1 += kPhiloxW1;
    }
    return philoxRound(c, k0, k1);
}

//! Word e (0 .. 3) of a Philox output.
__device__ __forceinline__ uint32_t philoxWord(uint4 v, int e)
{
    return e == 0 ? v.x : (e == 1 ? v.y : (e == 2 ? v.z : v.w));
}

//! Uniform in (0, 1] from one word (curand_uniform's formula with the contraction explicit): 2^-33 .. 1.
__device__ __forceinline__ float unif(uint32_t x)
{
    return __fmaf_rn(__uint2float_rn(x), 0x1p-32f, 0x1p-33f);
}

//! Exp(1) variate from a 64-bit uniform u = hi 2^-32 + lo 2^-64 + 2^-65 in (0, 1]: E = -log1p(-u), capped at kExpoMax.
//! E is in [2^-65, kExpoMax] and finite.
__device__ __forceinline__ float expo(uint32_t hi, uint32_t lo)
{
    float const u = __fmaf_rn(__uint2float_rn(hi), 0x1p-32f, __fmaf_rn(__uint2float_rn(lo), 0x1p-64f, 0x1p-65f));
    return fminf(-log1pf(-u), kExpoMax);
}

//! log E of element id of the race of subsequence S (the fold's per-element E and the draw's candidate E are this
//! function): hi word from counter 2 (id >> 2), lo word from counter 2 (id >> 2) + 1, both word id & 3.
__device__ __forceinline__ float logE(uint64_t seed, uint64_t subseq, int id)
{
    uint64_t const v = static_cast<uint64_t>(static_cast<uint32_t>(id) >> 2);
    int const e = id & 3;
    uint32_t const hi = philoxWord(Ph(seed, subseq, 2 * v), e);
    uint32_t const lo = philoxWord(Ph(seed, subseq, 2 * v + 1), e);
    return logf(expo(hi, lo));
}

//! Race order: (k1, i1) beats (k2, i2) if its key is larger, ties to the lower id; a NaN key counts as -inf.
__device__ __forceinline__ bool better(float k1, int i1, float k2, int i2)
{
    float const a = (k1 == k1) ? k1 : -__int_as_float(0x7F800000);
    float const b = (k2 == k2) ? k2 : -__int_as_float(0x7F800000);
    return a > b || (a == b && i1 < i2);
}

} // namespace rs
