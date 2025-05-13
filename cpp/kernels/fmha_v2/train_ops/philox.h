/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// Philox CUDA.
#include <cstdint>

class Philox
{
public:
    __device__ inline Philox(unsigned long long seed, unsigned long long subsequence, unsigned long long offset)
        : STATE(0)
    {
        // key.x = (unsigned int)seed;
        // key.y = (unsigned int)(seed >> 32);
        // counter = make_uint4(0, 0, 0, 0);
        // counter.z = (unsigned int)(subsequence);
        // counter.w = (unsigned int)(subsequence >> 32);
        // STATE = 0;
        // incr_n(offset / 4);

        key = reinterpret_cast<uint2 const&>(seed);
        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x = offset / 4;
        tmp->y = subsequence;
    }

    __device__ inline uint4 operator()()
    {
        if (STATE == 0)
        {
            uint4 counter_ = counter;
            uint2 key_ = key;
            // 7-round philox
            for (int i = 0; i < 6; i++)
            {
                counter_ = single_round(counter_, key_);
                key_.x += (kPhilox10A);
                key_.y += (kPhilox10B);
            }
            output = single_round(counter_, key_);
            incr();
        }
        // return a float4 directly
        // unsigned long ret;
        // switch(STATE) {
        //  case 0: ret = output.x; break;
        //  case 1: ret = output.y; break;
        //  case 2: ret = output.z; break;
        //  case 3: ret = output.w; break;
        //}
        // STATE = (STATE + 1) % 4;
        return output;
    }

    __device__ inline uint4 operator()(unsigned long long const subsequence)
    {
        uint4 counter_ = counter;
        ull2* tmp = reinterpret_cast<ull2*>(&counter_);
        tmp->y = subsequence;
        // if ((threadIdx.x % 32 == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("tidx = %d, counter_: %u, %u, %u, %u\n", threadIdx.x, counter_.x, counter_.y, counter_.z,
        //     counter_.w);
        // }
        uint2 key_ = key;
// 7-round philox
#pragma unroll
        for (int i = 0; i < 6; i++)
        {
            counter_ = single_round(counter_, key_);
            key_.x += (kPhilox10A);
            key_.y += (kPhilox10B);
        }
        output = single_round(counter_, key_);
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
        //     printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
        // }
        return output;
    }

private:
    struct ull2
    {
        uint64_t x;
        uint64_t y;
    };

    uint4 counter;
    uint4 output;
    uint2 key;
    unsigned int STATE;

    __device__ inline void incr_n(unsigned long long n)
    {
        unsigned int nlo = (unsigned int) (n);
        unsigned int nhi = (unsigned int) (n >> 32);
        counter.x += nlo;
        if (counter.x < nlo)
            nhi++;
        counter.y += nhi;
        if (nhi <= counter.y)
            return;
        if (++counter.z)
            return;
        ++counter.w;
    }

    __device__ uint4 incr128(uint4 ctr)
    {
        uint4 res;
        asm("add.cc.u32      %0, %4, %8;\n\t"
            "addc.cc.u32     %1, %5, %9;\n\t"
            "addc.cc.u32     %2, %6, %10;\n\t"
            "addc.u32        %3, %7, %11;\n\t"
            : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
            : "r"(ctr.x), "r"(ctr.y), "r"(ctr.z), "r"(ctr.w), "n"(1), "n"(0), "n"(0), "n"(0));
        return res;
    }

    __device__ inline void incr()
    {
        counter = incr128(counter);
    }

    __device__ unsigned int mulhilo32(unsigned int a, unsigned int b, unsigned int* result_high)
    {
        *result_high = __umulhi(a, b);
        return a * b;
    }

    __device__ uint2 mulhilo32_v2(unsigned int a, unsigned int b)
    {
        uint2* res;
        unsigned long long tmp;
        asm("mul.wide.u32      %0, %1, %2;\n\t" : "=l"(tmp) : "r"(a), "r"(b));
        res = (uint2*) (&tmp);
        return *res;
    }

    __device__ inline uint4 single_round(uint4 ctr, uint2 key)
    {
        // unsigned int hi0;
        // unsigned int hi1;
        // unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
        // unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
        // uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
        uint2 res0 = mulhilo32_v2(kPhiloxSA, ctr.x);
        uint2 res1 = mulhilo32_v2(kPhiloxSB, ctr.z);
        uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
        return ret;
    }

    static constexpr unsigned long kPhilox10A = 0x9E3779B9;
    static constexpr unsigned long kPhilox10B = 0xBB67AE85;
    static constexpr unsigned long kPhiloxSA = 0xD2511F53;
    static constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
};

// Inverse of 2^32.
constexpr float M_RAN_INVM32 = 2.3283064e-10f;

__device__ __inline__ float4 uniform4(uint4 x)
{
    return make_float4(x.x * M_RAN_INVM32, x.y * M_RAN_INVM32, x.z * M_RAN_INVM32, x.w * M_RAN_INVM32);
}

#define DI __device__ inline

struct PCGenerator
{
    /**
     * @brief ctor. Initializes the state for RNG. This code is derived from PCG basic code
     * @param seed the seed (can be same across all threads). Same as PCG's initstate
     * @param subsequence is same as PCG's initseq
     * @param offset unused
     */
    DI PCGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
    {
        state = uint64_t(0);
        inc = (subsequence << 1u) | 1u;
        uint32_t discard;
        next(discard);
        state += seed;
        next(discard);
        skipahead(offset);
    }

    // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
    // Link https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf
    DI void skipahead(uint64_t offset)
    {
        uint64_t G = 1;
        uint64_t h = 6364136223846793005ULL;
        uint64_t C = 0;
        uint64_t f = inc;
        while (offset)
        {
            if (offset & 1)
            {
                G = G * h;
                C = C * h + f;
            }
            f = f * (h + 1);
            h = h * h;
            offset >>= 1;
        }
        state = state * G + C;
    }

    /**
     * @defgroup NextRand Generate the next random number
     * @brief This code is derived from PCG basic code
     * @{
     */
    DI uint32_t next_u32()
    {
        uint32_t ret;
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        return ret;
    }

    DI uint64_t next_u64()
    {
        uint64_t ret;
        uint32_t a, b;
        a = next_u32();
        b = next_u32();
        ret = uint64_t(a) | (uint64_t(b) << 32);
        return ret;
    }

    DI int32_t next_i32()
    {
        int32_t ret;
        uint32_t val;
        val = next_u32();
        ret = int32_t(val & 0x7fffffff);
        return ret;
    }

    DI int64_t next_i64()
    {
        int64_t ret;
        uint64_t val;
        val = next_u64();
        ret = int64_t(val & 0x7fffffffffffffff);
        return ret;
    }

    DI float next_float()
    {
        float ret;
        uint32_t val = next_u32() >> 8;
        ret = static_cast<float>(val) / (1U << 24);
        return ret;
    }

    DI double next_double()
    {
        double ret;
        uint64_t val = next_u64() >> 11;
        ret = static_cast<double>(val) / (1LU << 53);
        return ret;
    }

    DI void next(uint32_t& ret)
    {
        ret = next_u32();
    }

    DI void next(uint64_t& ret)
    {
        ret = next_u64();
    }

    DI void next(int32_t& ret)
    {
        ret = next_i32();
    }

    DI void next(int64_t& ret)
    {
        ret = next_i64();
    }

    DI void next(float& ret)
    {
        ret = next_float();
    }

    DI void next(double& ret)
    {
        ret = next_double();
    }

    DI uint4 operator()()
    {
        return {next_u32(), next_u32(), next_u32(), next_u32()};
    }

    /** @} */

private:
    uint64_t state;
    uint64_t inc;
};
