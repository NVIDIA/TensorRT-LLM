/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fmha/numeric_types.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reference host-side quantizer for SageAttention, used by bin/fmha.exe to feed the SM90 kernels.
// It is a correctness reference, not a fast path: production callers quantize on the device.
//
// SageAttention is INT8 QK with an e4m3 PV. Q and K are quantized along the sequence axis and V
// along the channel axis, with the groups chosen so a thread never needs more than one scale per
// axis. See fmha/warpspec/compute.h for the matching kernel-side index arithmetic.

namespace sage_quant
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Geometry of the SM90 warp-specialized sage kernels, mirrored on the host so the scale layout
// built here is the one the kernel indexes into.
struct Geometry
{
    // The Q and KV tile sizes of the kernel.
    static constexpr int step_q = 64;
    static constexpr int step_kv = 256;
    // Q is quantized per thread: a thread owns the 2 strided rows quad_row and quad_row + 8.
    static constexpr int q_tokens_per_scale = 2;

    // Scales per q tile, laid out as [q_tile][warp * 8 + lane / 4].
    static constexpr int q_scales_per_tile = step_q / q_tokens_per_scale;
    // K sub-fragments per thread. A thread owns 64 key columns; splitting them by ni-range gives
    // this many groups, and a thread loads all of its scales with one 128-bit load.
    int k_chunks;
    // Scales per kv tile, laid out as [kv_tile][quad][chunk].
    int k_scales_per_tile;
    // Keys covered by one K scale, i.e. the K block size the kernel was compiled for.
    int k_tokens_per_scale;

    // The scale buffers are laid out as (H, scales for the whole batch), so a sequence's scales
    // start at a base derived from its cu_seqlens entry. Each sequence reserves one spare tile
    // because the kernel indexes a partial final tile in full; K reserves 4 more so its base can be
    // rounded up to a multiple of 4 and keep the kernel's float4 scale load aligned.
    static constexpr int q_slots_per_seq = q_scales_per_tile;
    int k_slots_per_seq;

    explicit Geometry(int block_size_k)
        : k_chunks(step_kv / (4 * std::max(1, block_size_k)))
        , k_scales_per_tile(4 * k_chunks)
        , k_tokens_per_scale(step_kv / k_scales_per_tile)
        , k_slots_per_seq(k_scales_per_tile + 4)
    {
    }

    // Where sequence bi's scales start, given its first token. Must match warpspec/compute.h.
    int q_base(int cu_seqlen, size_t bi) const
    {
        return cu_seqlen / q_tokens_per_scale + (int) bi * q_slots_per_seq;
    }

    int k_base(int cu_seqlen, size_t bi) const
    {
        return ((cu_seqlen / k_tokens_per_scale + 3) & ~3) + (int) bi * k_slots_per_seq;
    }

    // Scales per head, i.e. what the kernel reads as max_nblock. Depends only on the batch size and
    // the total token count, both of which the caller knows before it sees any sequence length.
    int q_stride(int total_tokens, size_t b) const
    {
        return q_base(total_tokens, b);
    }

    int k_stride(int total_tokens, size_t b) const
    {
        return k_base(total_tokens, b);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// One of Q, K or V: where to read fp32 from, where to write the quantized byte to, and how the
// tensor is addressed. Element (token ti, head hi, channel di) sits at
// ti * token_stride + hi * head_stride + di in both buffers, so one stride pair serves both.
//
// Strides rather than a layout enum because that covers every layout the harness feeds in with no
// branching in the inner loops:
//   MHA packed   [token][h][3][d]          token_stride = h * 3 * d,          head_stride = 3 * d
//   MQA packed   [token][h + 2 * h_kv][d]  token_stride = (h + 2 * h_kv) * d, head_stride = d
//   separate     [token][h][d]             token_stride = h * d,              head_stride = d
// For a packed layout the three Tensors alias one buffer at different offsets; for separate Q/K/V
// they are three distinct buffers, and the q and kv sequence lengths may differ.
struct Tensor
{
    float const* src;
    uint8_t* dst;
    size_t token_stride, head_stride;
};

// What to quantize.
struct Input
{
    Tensor q, k, v;

    // The dimensions.
    size_t b, h, h_kv, d, dv;
    // Prefix sums of the actual sequence lengths, b + 1 entries each. They are the same array for
    // a packed layout; separate Q/K/V allows them to differ.
    int const *cu_q_seqlens, *cu_kv_seqlens;

    // The block sizes requested on the command line, counted along the axis each tensor is
    // quantized over: 2 rows for Q, 16 keys for K, 1 channel for V (0 leaves V unquantized).
    int block_size_q, block_size_k, block_size_v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The scales. The quantized values themselves go straight into the caller's Tensor::dst buffers;
// Q and K are INT8 and V is e4m3, both one byte per element, so every stride is the input's.
struct Output
{
    // Layout (h, q_stride), a sequence's tiles swizzled as [q_tile][warp * 8 + lane / 4].
    std::vector<float> scales_q;
    // Layout (h_kv, k_stride), a sequence's tiles swizzled as [kv_tile][quad][chunk].
    std::vector<float> scales_k;
    // Layout (h_kv, dv), one scale per channel. Empty when V is not quantized. The amax reduces
    // over every token of every sequence, so there is no batch dimension.
    std::vector<float> scales_v;
    // Scales per head, what the kernel reads as max_nblock.
    int q_stride, k_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{

// (token, head, channel) -> flat index into a Tensor's src and dst buffers.
inline size_t at(Tensor const& t, size_t ti, size_t hi, size_t di)
{
    return ti * t.token_stride + hi * t.head_stride + di;
}

// A group with no valid tokens has amax 0. Give it a representative scale rather than an epsilon:
// the kernel masks with INT_MIN and relies on that surviving the multiply by this scale (see
// MASKED_ACC in fmha/warpspec/epilogue.h).
inline float make_scale(float amax, float div)
{
    return amax > 0.f ? amax / div : 1.f;
}

// With random inputs the amax of every group comes out nearly identical, which would make the
// scales effectively constant and hide any mis-indexed scale. Spread them instead.
//
// All three axes are mixed so the factor depends on every one of them. A plain `idx % 4` over
// `quad * k_chunks + g` would give every quad the same pattern (4 * quad is 0 mod 4) and a
// wrong-quad read would be invisible; likewise `pair % 4` over `warp * 8 + lane / 4` hides a
// wrong-warp read. The tile index has to be in the mix as well: without it every tile gets an
// identical pattern, so reading a neighbouring tile's scales -- which is exactly what a wrong
// per-sequence base does -- changes nothing, and with iid inputs the amax of every tile is nearly
// the same, so the error stays at noise.
//
// Q and K are fixed-point INT8, so inflating a scale genuinely throws away bits (8x would cost
// three of them); keep the four factors distinct but gentle.
inline float qk_spread(size_t minor, size_t major, size_t tile)
{
    return 1.f + 0.25f * ((minor + 3 * major + tile) % 4);
}

// V is e4m3, where a power-of-two factor is free: fp8 rounding is scale-invariant, so it leaves the
// error of a correct implementation unchanged and any extra error is a genuine indexing bug.
inline float v_spread(size_t idx)
{
    return float(1u << (idx % 4));
}

// Symmetric INT8, saturating.
inline void store_int8(Tensor const& t, size_t idx, float value, float scale)
{
    int const q = (int) lrintf(value / scale);
    reinterpret_cast<int8_t*>(t.dst)[idx] = (int8_t) std::min(127, std::max(-127, q));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Per-thread Q scales. The BMM1 fragment gives the thread at (warp w, lane l) the two query rows
// quad_row = w * 16 + l / 4 and quad_row + 8, so a group is 2 strided rows and there are
// step_q / 2 of them per q tile.
inline void quant_q_per_thread(Input const& in, Geometry const& geo, Output& out)
{
    out.q_stride = geo.q_stride(in.cu_q_seqlens[in.b], in.b);
    out.scales_q.assign((size_t) out.q_stride * in.h, 1.f);

    for (size_t bi = 0; bi < in.b; bi++)
    {
        int const seq_beg = in.cu_q_seqlens[bi], seq_end = in.cu_q_seqlens[bi + 1];
        int const num_tiles = (seq_end - seq_beg + geo.step_q - 1) / geo.step_q;
        int const base = geo.q_base(seq_beg, bi);
        for (size_t hi = 0; hi < in.h; hi++)
        {
            for (int tile = 0; tile < num_tiles; tile++)
            {
                for (int pair = 0; pair < geo.q_scales_per_tile; pair++)
                {
                    // pair == w * 8 + l / 4 -> quad_row = (pair / 8) * 16 + pair % 8
                    int const quad_row = (pair / 8) * 16 + (pair % 8);
                    int const rows[2] = {tile * geo.step_q + quad_row, tile * geo.step_q + quad_row + 8};

                    float amax = 0.f;
                    for (int r = 0; r < 2; r++)
                    {
                        if (seq_beg + rows[r] >= seq_end)
                        {
                            continue;
                        }
                        size_t const ti = seq_beg + rows[r];
                        for (size_t di = 0; di < in.d; di++)
                        {
                            amax = std::max(amax, fabsf(in.q.src[at(in.q, ti, hi, di)]));
                        }
                    }

                    float const scale = make_scale(amax, 127.f) * qk_spread(pair % 8, pair / 8, tile);
                    out.scales_q[hi * out.q_stride + base + tile * geo.q_scales_per_tile + pair] = scale;

                    for (int r = 0; r < 2; r++)
                    {
                        if (seq_beg + rows[r] >= seq_end)
                        {
                            continue;
                        }
                        size_t const ti = seq_beg + rows[r];
                        for (size_t di = 0; di < in.d; di++)
                        {
                            size_t const idx = at(in.q, ti, hi, di);
                            store_int8(in.q, idx, in.q.src[idx], scale);
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Per-thread K scales. A thread with lane % 4 == quad owns the key columns {8 * ni + 2 * quad + e},
// ni in [0, 32), e in {0, 1}. Splitting that fragment into k_chunks sub-fragments by ni-range gives
// k_tokens_per_scale keys per scale.
inline void quant_k_per_thread(Input const& in, Geometry const& geo, Output& out)
{
    int const ni_per_chunk = geo.step_kv / 8 / geo.k_chunks;
    out.k_stride = geo.k_stride(in.cu_kv_seqlens[in.b], in.b);
    out.scales_k.assign((size_t) out.k_stride * in.h_kv, 1.f);

    for (size_t bi = 0; bi < in.b; bi++)
    {
        int const seq_beg = in.cu_kv_seqlens[bi], seq_end = in.cu_kv_seqlens[bi + 1];
        int const num_tiles = (seq_end - seq_beg + geo.step_kv - 1) / geo.step_kv;
        int const base = geo.k_base(seq_beg, bi);
        for (size_t hi = 0; hi < in.h_kv; hi++)
        {
            for (int tile = 0; tile < num_tiles; tile++)
            {
                for (int quad = 0; quad < 4; quad++)
                {
                    for (int g = 0; g < geo.k_chunks; g++)
                    {
                        std::vector<int> toks;
                        for (int ni = g * ni_per_chunk; ni < (g + 1) * ni_per_chunk; ni++)
                        {
                            for (int e = 0; e < 2; e++)
                            {
                                int const local = tile * geo.step_kv + 8 * ni + 2 * quad + e;
                                if (seq_beg + local < seq_end)
                                {
                                    toks.push_back(seq_beg + local);
                                }
                            }
                        }

                        float amax = 0.f;
                        for (int ti : toks)
                        {
                            for (size_t di = 0; di < in.d; di++)
                            {
                                amax = std::max(amax, fabsf(in.k.src[at(in.k, ti, hi, di)]));
                            }
                        }

                        float const scale = make_scale(amax, 127.f) * qk_spread(g, quad, tile);
                        out.scales_k[hi * out.k_stride + base + tile * geo.k_scales_per_tile + quad * geo.k_chunks + g]
                            = scale;

                        for (int ti : toks)
                        {
                            for (size_t di = 0; di < in.d; di++)
                            {
                                size_t const idx = at(in.k, ti, hi, di);
                                store_int8(in.k, idx, in.k.src[idx], scale);
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Per-(head, channel) V scales: the amax reduces over the sequence, keeping the head dimension.
// There is a single set for the whole batch, so the reduction spans every token of every sequence.
inline void quant_v_per_channel(Input const& in, Output& out)
{
    int const total_tokens = in.cu_kv_seqlens[in.b];
    out.scales_v.assign(in.h_kv * in.dv, 1.f);

    for (size_t hi = 0; hi < in.h_kv; hi++)
    {
        for (size_t di = 0; di < in.dv; di++)
        {
            float amax = 0.f;
            for (int ti = 0; ti < total_tokens; ti++)
            {
                amax = std::max(amax, fabsf(in.v.src[at(in.v, ti, hi, di)]));
            }

            float const scale = make_scale(amax, 448.f) * v_spread(di);
            out.scales_v[hi * in.dv + di] = scale;

            for (int ti = 0; ti < total_tokens; ti++)
            {
                size_t const idx = at(in.v, ti, hi, di);
                reinterpret_cast<fmha::e4m3_t*>(in.v.dst)[idx] = fmha::e4m3_t(in.v.src[idx] / scale);
            }
        }
    }
}

} // namespace detail

inline Output quantize(Input const& in)
{
    Geometry const geo(in.block_size_k);

    Output out;
    out.q_stride = out.k_stride = 0;

    detail::quant_q_per_thread(in, geo, out);
    detail::quant_k_per_thread(in, geo, out);
    if (in.block_size_v == 1)
    {
        detail::quant_v_per_channel(in, out);
    }

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace sage_quant
