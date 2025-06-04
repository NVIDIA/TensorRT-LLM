/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Fp8GemmSwiGLUPerBlockTemplate.cuh"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Utils.cuh"

#include <map>
#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_fp8_gemm_swiglu
{

DEFINE_GET_FUNC_PTR(5120, true);

dim3 get_grid_size(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out)
{
    if (num_tokens % tile_token != 0)
    {
        throw std::runtime_error("num_tokens must be divisible by tile_token");
    }
    if (hidden_out % tile_out != 0)
    {
        throw std::runtime_error("hidden_out must be divisible by tile_out");
    }
    if (hidden_in != 5120)
    {
        throw std::runtime_error("hidden_in must be 5120");
    }
    return dim3(div_up(hidden_out, tile_out), div_up(num_tokens, tile_token), 1);
}

bool is_supported(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out)
{
    if (hidden_in != 5120)
    {
        return false;
    }
    if (num_tokens % tile_token != 0)
    {
        return false;
    }
    if (hidden_out % tile_out != 0)
    {
        return false;
    }
    return true;
}

template <int TILE_TOKEN, int TILE_OUT>
void dispatch_llama4_fp8_fp8_gemm_swiglu_hidden_in(void const* __restrict__ A, void const* __restrict__ B,
    void* __restrict__ C, void const* __restrict__ in_scale, void const* __restrict__ out_scale_inv, int num_tokens,
    int hidden_in, int hidden_out, cudaStream_t stream)
{
    void* func_ptr;
    constexpr int step = BLOCK_SIZE * VEC_SIZE;

    if (hidden_in == 5 * step)
    {
        func_ptr = get_func_ptr_aligned_true_5120_(TILE_TOKEN, TILE_OUT);
    }
    else
    {
        throw std::runtime_error("Unsupported hidden_in size " + std::to_string(hidden_in));
    }

    dim3 const grid_size = get_grid_size(num_tokens, hidden_in, hidden_out, TILE_TOKEN, TILE_OUT);

    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &in_scale, (void*) &out_scale_inv, (void*) &num_tokens,
        (void*) &hidden_in, (void*) &hidden_out};
    launch_kernel_pdl(grid_size, dim3(BLOCK_SIZE), stream, func_ptr, args, 8);
}

template <int TILE_TOKEN>
void dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out(void const* __restrict__ A, void const* __restrict__ B,
    void* __restrict__ C, void const* __restrict__ in_scale, void const* __restrict__ out_scale_inv, int num_tokens,
    int hidden_in, int hidden_out, int tile_out, cudaStream_t stream)
{
    if (tile_out == 1)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_hidden_in<TILE_TOKEN, 1>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    }
    else if (tile_out == 2)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_hidden_in<TILE_TOKEN, 2>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    }
    else if (tile_out == 4)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_hidden_in<TILE_TOKEN, 4>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported tile_out size " + std::to_string(tile_out));
    }
}

void dispatch_llama4_fp8_fp8_gemm_swiglu_tile_token(void const* __restrict__ A, void const* __restrict__ B,
    void* __restrict__ C, void const* __restrict__ in_scale, void const* __restrict__ out_scale_inv, int num_tokens,
    int hidden_in, int hidden_out, int tile_token, int tile_out, cudaStream_t stream)
{
    if (tile_token == 1)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out<1>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    }
    else if (tile_token == 2)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out<2>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    }
    else if (tile_token == 3)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out<3>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    }
    else if (tile_token == 4)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out<4>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    }
    else if (tile_token == 8)
    {
        dispatch_llama4_fp8_fp8_gemm_swiglu_tile_out<8>(
            A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported tile_token size " + std::to_string(tile_token));
    }
}

using Tactic = std::pair<int, int>;

int get_grid_size_total(int num_tokens, int hidden_in, int hidden_out, Tactic tactic)
{
    dim3 grid_size = get_grid_size(num_tokens, hidden_in, hidden_out, tactic.first, tactic.second);
    return grid_size.x * grid_size.y * grid_size.z;
}

std::optional<Tactic> get_hard_coded_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out)
{

    std::map<std::tuple<int, int, int>, Tactic> hard_coded_tactics = {
        {{1, 5120, 2048}, {1, 2}},
        {{1, 5120, 1024}, {1, 1}},
        {{2, 5120, 2048}, {2, 2}},
        {{2, 5120, 1024}, {1, 2}},
        {{4, 5120, 2048}, {2, 2}},
        {{4, 5120, 1024}, {2, 2}},
        {{8, 5120, 2048}, {4, 1}},
        {{8, 5120, 1024}, {4, 1}},
    };

    auto it = hard_coded_tactics.find({num_tokens, hidden_in, hidden_out});
    if (it != hard_coded_tactics.end())
    {
        return it->second;
    }

    return std::nullopt;
}

Tactic get_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out)
{
    // Try hard-coded heuristics first.
    auto hard_coded_tactic = get_hard_coded_heuristic_tactic(num_tokens, hidden_in, hidden_out);
    if (hard_coded_tactic)
    {
        return hard_coded_tactic.value();
    }

    // Fall back to analysis-based heuristics.
    int tile_token = 1;
    int tile_out = 1;

    std::vector<int> const tile_tokens = {1, 2, 3, 4, 8};
    for (auto t = tile_tokens.rbegin(); t != tile_tokens.rend(); ++t)
    {
        int const cta_threshold = 1024;
        if (num_tokens % *t == 0
            && get_grid_size_total(num_tokens, hidden_in, hidden_out, Tactic{*t, tile_out}) >= cta_threshold)
        {
            tile_token = *t;
            break;
        }
    }

    std::vector<int> const tile_outs = {1, 2, 4};
    for (auto t = tile_outs.rbegin(); t != tile_outs.rend(); ++t)
    {
        int const cta_threshold = 1024;
        if (hidden_in % *t == 0
            && get_grid_size_total(num_tokens, hidden_in, hidden_out, Tactic{tile_token, *t}) >= cta_threshold)
        {
            tile_out = *t;
            break;
        }
    }

    Tactic tactic = {tile_token, tile_out};
    if (!is_supported(num_tokens, hidden_in, hidden_out, tactic.first, tactic.second))
    {
        throw std::runtime_error("llama4Fp8Fp8GemmSwiGLU: unsupported tactic: " + std::to_string(tactic.first) + "-"
            + std::to_string(tactic.second));
    }

    return tactic;
}

void llama4_fp8_fp8_gemm_swiglu_op(int num_tokens, int hidden_in, int hidden_out, void const* A, void const* B, void* C,
    void const* in_scale, void const* out_scale_inv, cudaStream_t stream)
{
    Tactic tactic = get_heuristic_tactic(num_tokens, hidden_in, hidden_out);
    if (!is_supported(num_tokens, hidden_in, hidden_out, tactic.first, tactic.second))
    {
        throw std::runtime_error("llama4Fp8Fp8GemmSwiGLU: unsupported tactic: " + std::to_string(tactic.first) + "-"
            + std::to_string(tactic.second));
    }

    dispatch_llama4_fp8_fp8_gemm_swiglu_tile_token(
        A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tactic.first, tactic.second, stream);
}

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_fp8_gemm_swiglu
