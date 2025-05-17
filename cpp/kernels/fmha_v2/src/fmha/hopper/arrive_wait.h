/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// CP ASYNC FEATURES ///////////////////////////////////////////////////////////////////////////////
#if !defined(CUDA_CP_ASYNC_SUPPORTED)                                                                                  \
    && ((__CUDACC_VER_MAJOR__ >= 11) || ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)))
#define CUDA_CP_ASYNC_SUPPORTED 1
#endif

#if !defined(CUDA_CP_ASYNC_ENABLED) && (CUDA_CP_ASYNC_SUPPORTED)
#define CUDA_CP_ASYNC_ENABLED 1
#endif

#if CUDA_CP_ASYNC_ENABLED && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUDA_CP_ASYNC_ACTIVATED 1
#endif

#if !defined(CUDA_CP_ASYNC_GROUP_POLICY_SUPPORTED) && (CUDA_CP_ASYNC_SUPPORTED) && (__CUDACC_VER_MAJOR__ >= 11)
#define CUDA_CP_ASYNC_GROUP_POLICY_SUPPORTED 1
#endif

#if !defined(CUDA_CP_ASYNC_GROUP_POLICY_ENABLED) && (CUDA_CP_ASYNC_GROUP_POLICY_SUPPORTED)
#define CUDA_CP_ASYNC_GROUP_POLICY_ENABLED 1
#endif

#if CUDA_CP_ASYNC_GROUP_POLICY_ENABLED && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUDA_CP_ASYNC_GROUP_POLICY_ACTIVATED 1
#endif

#if !defined(CUDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED) && (CUDA_CP_ASYNC_SUPPORTED) && (__CUDACC_VER_MAJOR__ >= 11)
#define CUDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED 1
#endif

#if !defined(CUDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED) && (CUDA_CP_ASYNC_MBARRIER_ARRIVE_SUPPORTED)
#define CUDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED 1
#endif

#if (CUDA_CP_ASYNC_MBARRIER_ARRIVE_ENABLED) && (__CUDA_ARCH__ >= 800)
#define CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED 1
#endif

#if (CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED) && (CUDACC_VERSION >= 111)
#define CUDA_CP_ASYNC_MBARRIER_WAIT_ACTIVATED 1
#endif

#if !defined(FMHA_PTX_MBARRIER_TRYWAIT_NOSLEEP_INTERNAL_SUPPORT_ENABLED)
#define FMHA_PTX_MBARRIER_TRYWAIT_NOSLEEP_INTERNAL_SUPPORT_ENABLED 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void named_barrier_arrive(uint32_t BARRIER_ID, uint32_t NUM_THREADS)
{
    if (NUM_THREADS > 1)
    {
        asm volatile("bar.arrive %0, %1;" : : "r"(BARRIER_ID), "r"(NUM_THREADS));
    }
}

inline __device__ void named_barrier_wait(uint32_t BARRIER_ID, uint32_t NUM_THREADS)
{
    if (NUM_THREADS > 1)
    {
        asm volatile("bar.sync %0, %1;" ::"r"(BARRIER_ID), "r"(NUM_THREADS));
    }
}

// it is executed per thread, i.e., each thread can call and init a barrier.
// need a bar.sync after using it.
inline __device__ void bar_create(void* bar_ptr, int init_count)
{

    unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);

    asm volatile(
        "{\n\t"
#if CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
        "mbarrier.init.shared.b64 [%1], %0; \n\t"
#else
        ".reg .s32                negCnt, count, expectedCount;\n\t"
        ".reg .s64                comboCnt; \n\t"
        "neg.s32                  negCnt, %0;\n\t "
        "and.b32                  count, negCnt, 0x7fffffff; \n\t"
        "and.b32                  expectedCount, negCnt, 0x3fffffff; \n\t"
        "mov.b64                  comboCnt, {expectedCount, count}; \n\t"
        "st.shared.s64            [%1], comboCnt; \n\t"
#endif
        "}"
        :
        : "r"(init_count), "r"(smem_ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Arrive_wait
{
public:
    inline __device__ Arrive_wait()
    {
        bar_base_ = NULL;
    }

    inline __device__ Arrive_wait(uint64_t* bar_base, int id = 0)
    {
        bar_base_ = bar_base;
        id_ = id;
    }

    inline __device__ uint64_t* get_bar_addr(int32_t id)
    {
        return reinterpret_cast<uint64_t*>(bar_base_ + id);
    }

    inline __device__ int bar_peek(int id, unsigned int bar_phase)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);
        uint32_t result32;
#if FMHA_PTX_MBARRIER_TRYWAIT_NOSLEEP_INTERNAL_SUPPORT_ENABLED
        asm volatile(
            "{\n\t"
            ".reg .pred       P3; \n\t"
            "mbarrier.try_wait.parity.nosleep.shared.b64 P3, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P3; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase));
#else
        // public ptx default heruistic generate SASS equal to with .nosleep in internal ptx
        asm volatile(
            "{\n\t"
            ".reg .pred       P3; \n\t"
            "mbarrier.try_wait.parity.shared.b64 P3, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P3; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase));
#endif
        return result32;
#else
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned int output_phase = (bar_ptr[0] >> 63) & 1;

        return output_phase != bar_phase;
#endif
    }

    inline __device__ int bar_peek(int id, unsigned int bar_phase, int pred)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);
        uint32_t result32;
#if FMHA_PTX_MBARRIER_TRYWAIT_NOSLEEP_INTERNAL_SUPPORT_ENABLED
        asm volatile(
            "{\n\t"
            ".reg .pred       P3; \n\t"
            ".reg .pred P2;\n\t"
            "setp.eq.u32 P2, %3, 1;\n\t"
            "@P2 mbarrier.try_wait.parity.nosleep.shared.b64 P3, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P3; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase), "r"(pred));
#else
        // public ptx default heruistic generate SASS equal to with .nosleep in internal ptx
        asm volatile(
            "{\n\t"
            ".reg .pred       P3; \n\t"
            ".reg .pred P2;\n\t"
            "setp.eq.u32 P2, %3, 1;\n\t"
            "@P2 mbarrier.try_wait.parity.shared.b64 P3, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P3; \n\t"
            "}"
            : "=r"(result32)
            : "r"(smem_ptr), "r"(bar_phase), "r"(pred));
#endif
        return result32;
#else
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned int output_phase = (bar_ptr[0] >> 63) & 1;

        return output_phase != bar_phase;
#endif
    }

    inline __device__ void bar_wait(int id, unsigned int bar_phase)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);
        uint32_t large_val = 0x989680;
        asm volatile(
            "{\n\t"
            ".reg .pred                P3; \n\t"
            "LAB_WAIT: \n\t"
            //"mbarrier.try_wait.parity.b64 P3, [%0], %1; \n\t"
            "mbarrier.try_wait.parity.shared.b64 P3, [%0], %1, %2; \n\t"
            "@P3                       bra.uni DONE; \n\t"
            "bra.uni                   LAB_WAIT; \n\t"
            "DONE: \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(bar_phase), "r"(large_val));
#else
        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);

        asm volatile(
            "{\n\t"
            ".reg .pred                P3; \n\t"
#ifdef CUDA_CP_ASYNC_MBARRIER_WAIT_ACTIVATED
            "mbarrier.test_wait.parity.shared.b64  P3, [%0], %1;\n\t"
#else
            ".reg .s32                 high, low; \n\t"
            ".reg .u32                 currentPhase; \n\t"
            "ld.volatile.shared.v2.s32 { low, high }, [%0]; \n\t"
            "shr.u32                   currentPhase, high, 31; \n\t"
            "setp.ne.u32               P3, currentPhase, %1; \n\t"
#endif
            "@P3                       bra.uni DONE; \n\t"
            "LAB_WAIT: \n\t"
#ifdef CUDA_CP_ASYNC_MBARRIER_WAIT_ACTIVATED
            "mbarrier.test_wait.parity.shared.b64  P3, [%0], %1;\n\t"
#else
            "ld.volatile.shared.v2.s32 { low, high }, [%0]; \n\t"
            "shr.u32                   currentPhase, high, 31; \n\t"
            "setp.ne.u32               P3, currentPhase, %1; \n\t"
#endif
            "@P3                       bra.uni DONE; \n\t"
            "bra.uni                   LAB_WAIT; \n\t"
            "DONE: \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(bar_phase));
#endif
    }

    // Set the expected_transaction_count and add 1 arrive count (1 transaction = 1 Byte)
    // This PTX maps to SYNCS.ARRIVES.TRANS64.A1TR.
    inline __device__ void bar_arrive_set_transactioncnt(int id, int expected_copy_bytes)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1; \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(expected_copy_bytes));
#endif
    }

    // Set the expected_transaction_count and add 1 arrive count (1 transaction = 1 Byte)
    // This PTX maps to SYNCS.ARRIVES.TRANS64.A1TR.
    inline __device__ void bar_arrive_set_transactioncnt(int id, int expected_copy_bytes, uint32_t pred)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, %2, 1;\n\t"
            "@p mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1; \n\t"
            "}"
            :
            : "r"(smem_ptr), "r"(expected_copy_bytes), "r"(pred));
#endif
    }

    // Sends barrier arrive notification to DSMEM
    // Note this uses a slightly different syntax compared to normal arrive
    // NOTE : Caller has to ensure that set_bar_base_dsmem has been called prior to using this
    // This is done as a compiler optimizations (since set barrier base is independent)
    inline __device__ void bar_arrive_dsmem(int const& id)
    {

#if CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED

        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        // TODO : check with PTX team on setctarank (currently emitting errors)
        // asm volatile("{\n\t"
        //"setctarank.shared.u32 %0, %1, %2;\n\t"
        //"}"
        // : "=r"(dst_ptr) : "r"(smem_ptr), "r"(cta_id));

        asm volatile(
            "{\n\t"
            "mbarrier.arrive.b64   _, [%0];\n\t"
            "}"
            :
            : "l"(bar_ptr));
#endif
    }

    // Just a predicated version of the above function
    // Manually inlining it - since the compiler generates BRA instructions at the moment
    // NOTE : Caller has to ensure that set_bar_base_dsmem has been called prior to using this
    // This is done as a compiler optimizations (since set barrier base is independent)
    inline __device__ void bar_arrive_dsmem(int const& id, uint32_t const& pred)
    {
#if CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
        asm volatile(
            "{\n\t"
            " .reg .pred p;\n\t"
            " .reg .s64 addr;\n\t"
            " .reg .b64 tmp;\n\t"
            "   setp.eq.u32 p, %2, 1;\n\t"
            "   mul.wide.s32 tmp, %0, 8;\n\t"
            "   add.s64 addr, tmp, %1;\n\t"
            "@p mbarrier.arrive.b64   _, [addr];\n\t"
            "}"
            :
            : "r"(id), "l"(bar_base_), "r"(pred));
#endif
    }

    // Sets up the base address for arrival with the correct ctaid in cga
    inline __device__ void set_bar_base_dsmem(uint32_t const& cta_id)
    {
        bar_base_ = reinterpret_cast<uint64_t*>(
            ((unsigned long long int) bar_base_ & 0xFFFFFFFFF0FFFFFFULL) + (cta_id << 24));
    }

    inline __device__ void bar_arrive_normal(int id, bool flag = true)
    {

#if CUDA_CP_ASYNC_ACTIVATED && !(CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED)
        asm("membar.cta;");
#endif

        // to make distance for the dependence between atoms.arrive and shfl
        if (flag == true)
        {

            uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
            unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);

#if CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED

            asm volatile(
                "{\n\t"
                ".reg .b64 state; \n\t"
                "mbarrier.arrive.shared.b64   state, [%0];\n\t"
                "}"
                :
                : "r"(smem_ptr));

#elif CUDA_CP_ASYNC_ACTIVATED

            asm volatile(
                "{\n\t"
                ".reg .b64  state; \n\t"
                "atom.shared.arrive.b64       state, [%0];"
                "}"
                :
                : "r"(smem_ptr));
#endif
        }
    }

    inline __device__ void bar_arrive_ldgsts(int id)
    {

        uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(bar_base_ + id);
        unsigned smem_ptr = __nvvm_get_smem_pointer(bar_ptr);

#if CUDA_CP_ASYNC_MBARRIER_ARRIVE_ACTIVATED
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_ptr));
#elif CUDA_CP_ASYNC_ACTIVATED
        asm volatile("cp.async.arrive.shared.b64 [%0];" : : "r"(smem_ptr));
#endif
    }

    inline __device__ uint64_t* bar_base()
    {
        return bar_base_;
    }

private:
    // smem barrier base pointer
    uint64_t* bar_base_;
    // barrier id
    int id_;
};

// Set the expected_transaction_count and add 1 arrive count (1 transaction = 1 Byte)
// This PTX maps to SYNCS.ARRIVES.TRANS64.A1TR.
inline __device__ void bar_arrive_set_transactioncnt(unsigned smem_ptr, unsigned expected_copy_bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_copy.shared.b64 _, [%0], %1; \n\t"
        "}"
        :
        : "r"(smem_ptr), "r"(expected_copy_bytes));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
