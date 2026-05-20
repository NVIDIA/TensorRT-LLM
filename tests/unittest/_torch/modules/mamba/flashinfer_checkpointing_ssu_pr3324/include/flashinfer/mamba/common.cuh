/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MAMBA_COMMON_CUH_
#define FLASHINFER_MAMBA_COMMON_CUH_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <sstream>
#include <utility>

#include <flashinfer/utils.cuh>

namespace flashinfer::mamba
{

constexpr unsigned warpSize = 32;

// =============================================================================
// Common types and utilities
// =============================================================================

// Largest power of 2 that divides v (i.e. v & -v). Returns 1 when v == 0.
inline constexpr unsigned largestPow2Divisor(unsigned v)
{
    return v ? (v & (~v + 1)) : 1;
}

// Simple packed vector type for loading N elements of type T.
// Alignment is the largest power-of-2 factor of the total byte size,
// so it is always valid even when N * sizeof(T) is not a power of 2 (e.g. 3 × 2 = 6).
template <typename T, int N = sizeof(float4) / sizeof(T)>
struct alignas(largestPow2Divisor(N * sizeof(T))) PackedAligned
{
    T val[N];
    static constexpr int count = N;
    using dtype = T;
};

template <class load_t>
__device__ __forceinline__ auto make_zeros() -> load_t
{
    load_t ret{};
#pragma unroll
    for (int i = 0; i < ret.count; i++)
        ret.val[i] = typename load_t::dtype{}; // default initialization
    return ret;
};

// Computes the vector load size that ensures full warp utilization.
// Avoids cases like: dstate=64, load_t = sizeof(float4)/sizeof(f16), warpsize=32 (32 * 8 > 64)
// in which case a part of the warp would be idle.
template <typename T, int DSTATE>
inline constexpr auto getVectorLoadSizeForFullUtilization() -> unsigned
{
    static_assert(sizeof(float4) >= sizeof(T));
    constexpr unsigned maxHardwareLoadSize = sizeof(float4) / sizeof(T);
    constexpr unsigned maxLogicalLoadSize = (unsigned) DSTATE / warpSize;
    return maxHardwareLoadSize < maxLogicalLoadSize ? maxHardwareLoadSize : maxLogicalLoadSize;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int s = warpSize / 2; s > 0; s /= 2)
    {
        val += __shfl_down_sync(UINT32_MAX, val, s);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
    for (int s = warpSize / 2; s > 0; s /= 2)
    {
        val = max(val, __shfl_down_sync(UINT32_MAX, val, s));
    }
    return val;
}

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
}

__device__ __forceinline__ float thresholded_softplus(float dt_value)
{
    constexpr float threshold = 20.f;
    return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

// =============================================================================
// Dispatch helpers
// =============================================================================

// Format an integer_sequence as a comma-separated string for error messages
template <int... Values>
std::string format_sequence(std::integer_sequence<int, Values...>)
{
    std::ostringstream oss;
    bool first = true;
    ((oss << (first ? (first = false, "") : ", ") << Values), ...);
    return oss.str();
}

// Helper function to dispatch dim and dstate with a kernel launcher
template <typename ParamsType, typename KernelLauncher, int... AllowedDims, int... AllowedDstates>
void dispatchDimDstate(ParamsType& params, std::integer_sequence<int, AllowedDims...> dims_seq,
    std::integer_sequence<int, AllowedDstates...> dstates_seq, KernelLauncher&& launcher)
{
    auto dispatch_dstate = [&]<int DIM>()
    {
        auto try_dstate = [&]<int DSTATE>()
        {
            if (params.dstate == DSTATE)
            {
                launcher.template operator()<DIM, DSTATE>();
                return true;
            }
            return false;
        };
        bool dispatched = (try_dstate.template operator()<AllowedDstates>() || ...);
        FLASHINFER_CHECK(dispatched, "Unsupported dstate value: ", params.dstate,
            ".\nSupported values: ", format_sequence(dstates_seq));
    };

    auto try_dim = [&]<int DIM>()
    {
        if (params.dim == DIM)
        {
            dispatch_dstate.template operator()<DIM>();
            return true;
        }
        return false;
    };

    bool dim_dispatched = (try_dim.template operator()<AllowedDims>() || ...);
    FLASHINFER_CHECK(
        dim_dispatched, "Unsupported dim value: ", params.dim, ".\nSupported values: ", format_sequence(dims_seq));
}

// Helper function to dispatch ratio with a kernel launcher
template <typename ParamsType, typename KernelLauncher, int... AllowedRatios>
void dispatchRatio(
    ParamsType& params, std::integer_sequence<int, AllowedRatios...> ratios_seq, KernelLauncher&& launcher)
{
    auto try_ratio = [&]<int RATIO>()
    {
        if (params.nheads / params.ngroups == RATIO)
        {
            launcher.template operator()<RATIO>();
            return true;
        }
        return false;
    };

    bool ratio_dispatched = (try_ratio.template operator()<AllowedRatios>() || ...);
    FLASHINFER_CHECK(ratio_dispatched, "Unsupported nheads/ngroups ratio: ", params.nheads / params.ngroups,
        ".\nSupported values: ", format_sequence(ratios_seq));
}

// Helper function to dispatch dim, dstate, and ntokens_mtp with a kernel launcher
// Reuses dispatchDimDstate by wrapping the launcher to add token dispatch
template <typename ParamsType, typename KernelLauncher, int... AllowedDims, int... AllowedDstates, int... AllowedTokens>
void dispatchDimDstateTokens(ParamsType& params, std::integer_sequence<int, AllowedDims...> dims_seq,
    std::integer_sequence<int, AllowedDstates...> dstates_seq, std::integer_sequence<int, AllowedTokens...> tokens_seq,
    KernelLauncher&& launcher)
{
    // Wrap the launcher to add token dispatch as the innermost level
    auto dim_dstate_launcher = [&]<int DIM, int DSTATE>()
    {
        auto try_tokens = [&]<int TOKENS_MTP>()
        {
            if (params.ntokens_mtp == TOKENS_MTP)
            {
                launcher.template operator()<DIM, DSTATE, TOKENS_MTP>();
                return true;
            }
            return false;
        };
        bool dispatched = (try_tokens.template operator()<AllowedTokens>() || ...);
        FLASHINFER_CHECK(dispatched, "Unsupported ntokens_mtp value: ", params.ntokens_mtp,
            ".\nSupported values: ", format_sequence(tokens_seq));
    };

    dispatchDimDstate(params, dims_seq, dstates_seq, dim_dstate_launcher);
}

// =============================================================================
// Alignment checks
// =============================================================================

// Check alignment for common input variables (x, z, B, C)
// Works for both STP (SelectiveStateUpdateParams) and MTP (SelectiveStateMTPParams)
template <typename input_t, typename ParamsType>
void check_ptr_alignment_input_vars(ParamsType const& params)
{
    using load_input_t = PackedAligned<input_t>;
    FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.x) % sizeof(load_input_t) == 0, "x pointer must be aligned to ",
        sizeof(load_input_t), " bytes");
    FLASHINFER_CHECK((params.x_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "x batch stride must be aligned to ", sizeof(load_input_t), " bytes");
    if (params.z)
    {
        FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.z) % sizeof(load_input_t) == 0,
            "z pointer must be aligned to ", sizeof(load_input_t), " bytes");
        FLASHINFER_CHECK((params.z_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
            "z batch stride must be aligned to ", sizeof(load_input_t), " bytes");
    }
    FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.B) % sizeof(load_input_t) == 0, "B pointer must be aligned to ",
        sizeof(load_input_t), " bytes");
    FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.C) % sizeof(load_input_t) == 0, "C pointer must be aligned to ",
        sizeof(load_input_t), " bytes");
    FLASHINFER_CHECK((params.B_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "B batch stride must be aligned to ", sizeof(load_input_t), " bytes");
    FLASHINFER_CHECK((params.C_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "C batch stride must be aligned to ", sizeof(load_input_t), " bytes");
}

} // namespace flashinfer::mamba

#endif // FLASHINFER_MAMBA_COMMON_CUH_
