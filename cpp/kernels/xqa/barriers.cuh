/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cuda_hint.cuh"
#include "defines.h"
#if !USE_CUSTOM_BARRIER
#include <cuda/std/barrier>
using CtaBarrier = cuda::barrier<cuda::thread_scope_block>;
#else

#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#if __CUDACC_VER_MAJOR__ < 12
#define STR_REL_CTA ""
#define STR_ACQ_CTA ""
#else
#define STR_REL_CTA ".release.cta"
#define STR_ACQ_CTA ".acquire.cta"
#endif

class CtaBarrier
{
public:
    enum class ArrivalToken : uint64_t
    {
    };
    using arrival_token = ArrivalToken;

    __device__ inline CtaBarrier(uint32_t count)
    {
        assert(count > 0);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr()), "r"(count));
    }

    __device__ ~CtaBarrier()
    {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(addr()));
    }

    __device__ inline ArrivalToken arrive(uint32_t update = 1)
    {
        ArrivalToken token;
#if __CUDA_ARCH__ >= 900
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1], %2;\n"
                     : "=l"(token)
                     : "r"(addr()), "r"(update));
#else
        if (update > 1)
        {
            asm volatile("mbarrier.arrive.noComplete" STR_REL_CTA ".shared::cta.b64 %0, [%1], %2;\n"
                         : "=l"(token)
                         : "r"(addr()), "r"(update - 1U));
            ArrivalToken refToken;
            asm volatile("mbarrier.arrive" STR_REL_CTA ".shared::cta.b64 %0, [%1];\n" : "=l"(refToken) : "r"(addr()));
            assert(token == refToken);
            return token;
        }
        else
        {
            asm volatile("mbarrier.arrive" STR_REL_CTA ".shared::cta.b64 %0, [%1];\n" : "=l"(token) : "r"(addr()));
        }
#endif
        return token;
    }
#if __CUDA_ARCH__ >= 900
    __device__ inline ArrivalToken arrive_tx(uint32_t txCount, uint32_t arriveCount = 1)
    {
        if (arriveCount == 1)
        {
            ArrivalToken token;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\n"
                         : "=l"(token)
                         : "r"(addr()), "r"(txCount));
            return token;
        }
        else
        {
            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" ::"r"(addr()), "r"(txCount));
            return arrive(arriveCount);
        }
    }
#endif
    __device__ inline bool test_wait(ArrivalToken&& token)
    {
        uint32_t ready;
        asm volatile(
            "{\n"
            ".reg .pred ready;\n"
            "mbarrier.test_wait" STR_ACQ_CTA
            ".shared::cta.b64 ready, [%1], %2;\n"
            "selp.b32 %0, 1, 0, ready;\n"
            "}\n"
            : "=r"(ready)
            : "r"(addr()), "l"(token));
        return ready != 0;
    }

    __device__ inline bool test_wait_parity(bool parity)
    {
        uint32_t ready;
        asm volatile(
            "{\n"
            ".reg .pred ready;\n"
            "mbarrier.test_wait.parity" STR_ACQ_CTA
            ".shared::cta.b64 ready, [%1], %2;\n"
            "selp.b32 %0, 1, 0, ready;\n"
            "}\n"
            : "=r"(ready)
            : "r"(addr()), "r"(uint32_t{parity}));
        return ready != 0;
    }
#if __CUDA_ARCH__ >= 900
    __device__ inline bool try_wait(ArrivalToken&& token)
    {
        uint32_t ready;
        asm volatile(
            "{\n"
            ".reg .pred ready;\n"
            "mbarrier.try_wait.acquire.cta.shared::cta.b64 ready, [%1], %2, %3;\n"
            "selp.b32 %0, 1, 0, ready;\n"
            "}\n"
            : "=r"(ready)
            : "r"(addr()), "l"(token), "n"(kSUSPEND_TIME_HINT));
        return ready != 0;
    }

    __device__ inline bool try_wait_parity(bool parity)
    {
        uint32_t ready;
        asm volatile(
            "{\n"
            ".reg .pred ready;\n"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%1], %2, %3;\n"
            "selp.b32 %0, 1, 0, ready;\n"
            "}\n"
            : "=r"(ready)
            : "r"(addr()), "r"(uint32_t{parity}), "n"(kSUSPEND_TIME_HINT));
        return ready != 0;
    }
#endif
    __device__ inline void wait(ArrivalToken&& token)
    {
#if __CUDA_ARCH__ >= 900
        poll<true>([&]() { return try_wait(ArrivalToken{token}); });
#else
        poll<false>([&]() { return test_wait(ArrivalToken{token}); });
#endif
    }

    // starting from `parity = false`.
    __device__ inline void wait_parity(bool parity)
    {
#if __CUDA_ARCH__ >= 900
        poll<true>([&]() { return try_wait_parity(parity); });
#else
        poll<false>([&]() { return test_wait_parity(parity); });
#endif
    }

    __device__ inline void arrive_and_wait(uint32_t update = 1)
    {
        wait(arrive(update));
    }

private:
    __device__ inline uint32_t addr() const
    {
        return __cvta_generic_to_shared(&mBar);
    }

    template <bool funcSupportsBlocking, typename F>
    __device__ inline static void poll(F&& func)
    {
        if constexpr (funcSupportsBlocking)
        {
            while (!func())
            {
            }
        }
        else
        {
            float sleepDuration = 0.125F;
            while (!func())
            {
                // if (sleepDuration > 1) {
                __nanosleep(uint32_t(sleepDuration));
                // }
                sleepDuration = sleepDuration * 1.25F + 0.F;
            }
        }
    }

public:
    static constexpr uint32_t kSUSPEND_TIME_HINT = 0xFFFFFFFFU;

private:
    uint64_t mBar;
};

__device__ inline void init(CtaBarrier* bar, uint32_t count)
{
    new (bar) CtaBarrier{count};
}

class NamedBarrier
{
public:
    __device__ inline NamedBarrier(uint32_t idxBar, uint32_t arriveCount)
        : mName{idxBar}
        , mArriveCount{arriveCount}
    {
        assert(idxBar < 16 && arriveCount % 32 == 0);
    }

    __device__ inline void arrive() const
    {
        asm volatile("barrier.cta.arrive %0, %1;\n" ::"r"(mName), "r"(mArriveCount));
    }

    __device__ inline void wait() const
    {
        asm volatile("barrier.cta.sync %0, %1;\n" ::"r"(mName), "r"(mArriveCount));
    }

private:
    uint32_t const mName;
    uint32_t const mArriveCount;
};

__device__ inline void namedBarSync(uint32_t idxBar, uint32_t arriveCount)
{
    NamedBarrier bar{idxBar, arriveCount};
    bar.arrive();
    bar.wait();
}
#endif
