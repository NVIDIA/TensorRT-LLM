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

#include <cfloat>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES_PER_LDG>
struct Fragment_ldg
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_ldg<1>
{
    template <typename Fragment>
    static inline __device__ void ldg(Fragment& f, int ii, void const* ptr)
    {
        uint8_t tmp;
        fmha::ldg(tmp, ptr);
        f.u8(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_ldg<2>
{
    template <typename Fragment>
    static inline __device__ void ldg(Fragment& f, int ii, void const* ptr)
    {
        uint16_t tmp;
        fmha::ldg(tmp, ptr);
        f.u16(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_ldg<4>
{
    template <typename Fragment>
    static inline __device__ void ldg(Fragment& f, int ii, void const* ptr)
    {
        uint32_t tmp;
        fmha::ldg(tmp, ptr);
        f.reg(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_ldg<8>
{
    template <typename Fragment>
    static inline __device__ void ldg(Fragment& f, int ii, void const* ptr)
    {
        uint2 tmp;
        fmha::ldg(tmp, ptr);
        f.reg(2 * ii + 0) = tmp.x;
        f.reg(2 * ii + 1) = tmp.y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_ldg<16>
{
    template <typename Fragment>
    static inline __device__ void ldg(Fragment& f, int ii, void const* ptr)
    {
        uint4 tmp;
        fmha::ldg(tmp, ptr);
        f.reg(4 * ii + 0) = tmp.x;
        f.reg(4 * ii + 1) = tmp.y;
        f.reg(4 * ii + 2) = tmp.z;
        f.reg(4 * ii + 3) = tmp.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES_PER_LDS>
struct Fragment_lds
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_lds<2>
{
    template <typename Fragment>
    static inline __device__ void lds(Fragment& f, int ii, uint32_t ptr)
    {
        uint16_t tmp;
        fmha::lds(tmp, ptr);
        f.u16(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_lds<4>
{
    template <typename Fragment>
    static inline __device__ void lds(Fragment& f, int ii, uint32_t ptr)
    {
        uint32_t tmp;
        fmha::lds(tmp, ptr);
        f.reg(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_lds<8>
{
    template <typename Fragment>
    static inline __device__ void lds(Fragment& f, int ii, uint32_t ptr)
    {
        uint2 tmp;
        fmha::lds(tmp, ptr);
        f.reg(2 * ii + 0) = tmp.x;
        f.reg(2 * ii + 1) = tmp.y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_lds<16>
{
    template <typename Fragment>
    static inline __device__ void lds(Fragment& f, int ii, uint32_t ptr)
    {
        uint4 tmp;
        fmha::lds(tmp, ptr);
        f.reg(4 * ii + 0) = tmp.x;
        f.reg(4 * ii + 1) = tmp.y;
        f.reg(4 * ii + 2) = tmp.z;
        f.reg(4 * ii + 3) = tmp.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<>
// struct Fragment_lds<32> {
//     template< typename Fragment >
//     static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
//         uint4 tmp;
//         fmha::lds(tmp, ptr);
//         f.reg(8*ii+0) = tmp.x;
//         f.reg(8*ii+1) = tmp.y;
//         f.reg(8*ii+2) = tmp.z;
//         f.reg(8*ii+3) = tmp.w;
//
//         fmha::lds(tmp, static_cast<const char*>(ptr)+sizeof(uint4));
//         f.reg(8*ii+4) = tmp.x;
//         f.reg(8*ii+5) = tmp.y;
//         f.reg(8*ii+6) = tmp.z;
//         f.reg(8*ii+7) = tmp.w;
//     }
// };

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES_PER_STG>
struct Fragment_stg
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_stg<1>
{
    template <typename Fragment>
    static inline __device__ void stg(void* ptr, Fragment const& f, int ii = 0)
    {
        fmha::stg(ptr, f.u8(ii));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_stg<2>
{
    template <typename Fragment>
    static inline __device__ void stg(void* ptr, Fragment const& f, int ii = 0)
    {
        fmha::stg(ptr, f.u16(ii));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_stg<4>
{
    template <typename Fragment>
    static inline __device__ void stg(void* ptr, Fragment const& f, int ii = 0)
    {
        fmha::stg(ptr, f.reg(ii));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_stg<8>
{
    template <typename Fragment>
    static inline __device__ void stg(void* ptr, Fragment const& f, int ii = 0)
    {
        uint2 tmp;
        tmp.x = f.reg(2 * ii + 0);
        tmp.y = f.reg(2 * ii + 1);
        fmha::stg(ptr, tmp);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_stg<16>
{
    template <typename Fragment>
    static inline __device__ void stg(void* ptr, Fragment const& f, int ii = 0)
    {
        uint4 tmp;
        tmp.x = f.reg(4 * ii + 0);
        tmp.y = f.reg(4 * ii + 1);
        tmp.z = f.reg(4 * ii + 2);
        tmp.w = f.reg(4 * ii + 3);
        fmha::stg(ptr, tmp);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type_, int NUM_ELTS_, int BITS_PER_ELT_, int ALIGNMENT_>
struct Fragment_base_
{

    // The data type.
    using Data_type = Data_type_;
    // default input type
    using Input_type_ = Data_type_;

    // Does it store the array of elements.
    enum
    {
        HAS_ELTS = BITS_PER_ELT_ >= 8
    };

    // The number of elements.
    enum
    {
        NUM_ELTS = NUM_ELTS_
    };

    // The size of element in bits.
    enum
    {
        BITS_PER_ELT = BITS_PER_ELT_
    };

    // The size of byte of a single register.
    enum
    {
        BYTES_PER_REG = 4
    };

    // The size in bits.
    enum
    {
        BITS_PER_REG = BYTES_PER_REG * 8
    };

    // The number of registers needed to store the fragment.
    enum
    {
        NUM_REGS = Div_up<NUM_ELTS * BITS_PER_ELT, BITS_PER_REG>::VALUE
    };

    // The size in bytes (as returned by sizeof(Fragment_base<>).
    enum
    {
        SIZE_IN_BYTES = NUM_REGS * BYTES_PER_REG
    };

    // The alignment.
    enum
    {
        ALIGNMENT = ALIGNMENT_ > 0 ? ALIGNMENT_ : Min<NUM_REGS * BYTES_PER_REG, 16>::VALUE
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The type of the elements.
    typename Data_type_,
    // The number of elements.
    int NUM_ELTS_,
    // The size of each element in bits.
    int BITS_PER_ELT_,
    // The alignment if you want to force a value -- use 0 otherwise.
    int ALIGNMENT_,
    // The base class.
    typename Base_ = Fragment_base_<Data_type_, NUM_ELTS_, BITS_PER_ELT_, ALIGNMENT_>>
struct alignas(static_cast<int>(Base_::ALIGNMENT)) Fragment_base : public Base_
{

    // The size of a load/store.
    enum
    {
        BYTES_PER_LOAD_STORE = Base_::NUM_REGS * sizeof(uint32_t)
    };

    // Clear the fragment. Using PTX in that code seems to produce better SASS...
    inline __device__ void clear()
    {
#pragma unroll
        for (int ii = 0; ii < Base_::NUM_REGS; ++ii)
        {
            asm volatile("mov.u32 %0, 0; \n" : "=r"(this->reg(ii)) :);
        }
    }

    // Load from global memory.
    inline __device__ void ldg(void const* ptr)
    {
        Fragment_ldg<Base_::SIZE_IN_BYTES>::ldg(*this, 0, ptr);
    }

    // Load from shared memory.
    inline __device__ void lds(uint32_t ptr)
    {
        Fragment_lds<Base_::SIZE_IN_BYTES>::lds(*this, 0, ptr);
    }

    // Immutable access to a register.
    inline __device__ uint32_t const& reg(int ii) const
    {
        return this->regs_[ii];
    }

    // Mutable access to a register.
    inline __device__ uint32_t& reg(int ii)
    {
        return this->regs_[ii];
    }

    // Set the fragment with a scalar
    inline __device__ void set(uint32_t value)
    {
#pragma unroll
        for (int ii = 0; ii < Base_::NUM_REGS; ++ii)
        {
            this->reg(ii) = value;
        }
    }

    // Store to global memory.
    inline __device__ void stg(void* ptr) const
    {
        Fragment_stg<Base_::SIZE_IN_BYTES>::stg(ptr, *this, 0);
    }

    // Immutable access to a byte.
    inline __device__ uint8_t u8(int ii) const
    {
        return reinterpret_cast<uint8_t const*>(&this->regs_[0])[ii];
    }

    // Mutable access to a u8.
    inline __device__ uint8_t& u8(int ii)
    {
        return reinterpret_cast<uint8_t*>(&this->regs_[0])[ii];
    }

    // Immutable access to a half-word..
    inline __device__ uint16_t u16(int ii) const
    {
        return reinterpret_cast<uint16_t const*>(&this->regs_[0])[ii];
    }

    // Mutable access to a half-word.
    inline __device__ uint16_t& u16(int ii)
    {
        return reinterpret_cast<uint16_t*>(&this->regs_[0])[ii];
    }

    // Immutable access to a word.
    inline __device__ uint32_t u32(int ii) const
    {
        return reinterpret_cast<uint32_t const*>(&this->regs_[0])[ii];
    }

    // Mutable access to a word.
    inline __device__ uint32_t& u32(int ii)
    {
        return reinterpret_cast<uint32_t*>(&this->regs_[0])[ii];
    }

    // Immutable access to a word.
    inline __device__ uint2 u64(int ii) const
    {
        return reinterpret_cast<uint2 const*>(&this->regs_[0])[ii];
    }

    // Mutable access to a word.
    inline __device__ uint2& u64(int ii)
    {
        return reinterpret_cast<uint2*>(&this->regs_[0])[ii];
    }

    // The storage in registers.
    //
    // NOTE: Instead of using only an array of uint32_t, we could use a union so we could either
    // access the registers or the elements. We found that for:
    //
    // union {
    //   uint16_t elts_[4]; uint32_t regs_[2];
    // };
    //
    // The compiler does not always produce a final structure of 8B. So, for the moment we are
    // going to go only with the regs_ array and use reinterpret_cast<> to access elements (see
    // below). It may be worth revisiting that when time permits.
    uint32_t regs_[Base_::NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type_, int NUM_ELTS_, int ALIGNMENT_ = 0>
struct Fragment : public Fragment_base<Data_type_, NUM_ELTS_, 8 * sizeof(Data_type_), ALIGNMENT_>
{

    // Immutable access to the elements.
    inline __device__ Data_type_ const& elt(int ii) const
    {
        return reinterpret_cast<Data_type_ const*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    inline __device__ Data_type_& elt(int ii)
    {
        return reinterpret_cast<Data_type_*>(&this->regs_[0])[ii];
    }

    // Immutable access to the elements with a cast.
    template <typename Cast_type>
    inline __device__ Cast_type const& elt_as(int ii) const
    {
        return reinterpret_cast<Cast_type const*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    template <typename Cast_type>
    inline __device__ Cast_type& elt_as(int ii)
    {
        return reinterpret_cast<Cast_type*>(&this->regs_[0])[ii];
    }

    // Add another fragment.
    inline __device__ void add(Fragment const& other)
    {
#pragma unroll
        for (int ii = 0; ii < NUM_ELTS_; ++ii)
        {
            this->elt(ii) += other.elt(ii);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Layout>
struct Fragment_a
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Volta_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Volta_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Turing_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 4>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Turing_hmma_fp32_traits, Layout> : public Fragment<uint16_t, 4>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Turing_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ampere_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ampere_hmma_bf16_bf16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ampere_hmma_fp32_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ampere_hmma_bf16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ampere_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ada_qmma_e4m3_fp32_traits, Layout> : public Fragment<e4m3_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_a<Ada_qmma_e4m3_fp16_traits, Layout> : public Fragment<e4m3_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Layout>
struct Fragment_b
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Volta_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Volta_hmma_fp16_16x16x16_traits, Layout> : public Fragment<uint16_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Volta_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Turing_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 4>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Turing_hmma_fp32_traits, Layout> : public Fragment<uint16_t, 4>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Turing_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ampere_hmma_fp16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ampere_hmma_bf16_bf16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ampere_hmma_fp32_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ampere_hmma_bf16_traits, Layout> : public Fragment<uint16_t, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ampere_imma_int8_int32_traits, Layout> : public Fragment<int8_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ada_qmma_e4m3_fp32_traits, Layout> : public Fragment<e4m3_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
struct Fragment_b<Ada_qmma_e4m3_fp16_traits, Layout> : public Fragment<e4m3_t, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Fragment_accumulator
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Volta_hmma_fp16_traits> : public Fragment<uint16_t, 8>
{

    // The traits.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Fragment<uint16_t, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Traits, Row>;
    using Fragment_b = Fragment_b<Traits, Col>;

    // HMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(0)), "+r"(this->reg(1)), "+r"(this->reg(2)), "+r"(this->reg(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(0)), "+r"(this->reg(1)), "+r"(this->reg(2)), "+r"(this->reg(3))
            : "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(2)), "r"(b.reg(3)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Volta_hmma_fp16_16x16x16_traits> : public Fragment<uint16_t, 16>
{

    // The base class.
    using Base = Fragment<uint16_t, 16>;

    // The fragments.
    using Fragment_a = Fragment_a<Volta_hmma_fp16_traits, Row>;
    using Fragment_b = Fragment_b<Volta_hmma_fp16_16x16x16_traits, Row>;

    // HMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
        // K = 0..3 for threads 0..7 and 16..23 and K = 4..7 for 8..15 and 24..31.
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(0)), "+r"(this->reg(1)), "+r"(this->reg(2)), "+r"(this->reg(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(4)), "+r"(this->reg(5)), "+r"(this->reg(6)), "+r"(this->reg(7))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(2)), "r"(b.reg(3)));

        // K = 8..11 for threads 0..7 and 16..23 and K = 12..15 for 8..15 and 24..31.
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(0)), "+r"(this->reg(1)), "+r"(this->reg(2)), "+r"(this->reg(3))
            : "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(4)), "r"(b.reg(5)));
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+r"(this->reg(4)), "+r"(this->reg(5)), "+r"(this->reg(6)), "+r"(this->reg(7))
            : "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(6)), "r"(b.reg(7)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Volta_imma_int8_int32_traits> : public Fragment<int32_t, 8>
{

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Volta_imma_int8_int32_traits, Row>;
    using Fragment_b = Fragment_b<Volta_imma_int8_int32_traits, Col>;

    // IMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            asm volatile(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \n"
                "    {%0, %1}, \n"
                "    {%2}, \n"
                "    {%3}, \n"
                "    {%0, %1}; \n"
                : "+r"(this->reg(2 * i + 0)), "+r"(this->reg(2 * i + 1))
                : "r"(a.reg(i / 2)), "r"(b.reg(i % 2)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Turing_hmma_fp16_traits> : public Fragment<uint16_t, 8>
{

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(
        Fragment_a<Turing_hmma_fp16_traits, Layout_a> const& a, Fragment_b<Turing_hmma_fp16_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1}, \n"
            "    {%2, %3}, \n"
            "    {%4}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(0)), "+r"(reg(1))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(0)));
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1}, \n"
            "    {%2, %3}, \n"
            "    {%4}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(2)), "+r"(reg(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(1)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Turing_hmma_fp32_traits> : public Fragment<float, 8>
{

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    inline __device__ void mul(float const other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) *= other;
        }
    }

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(
        Fragment_a<Turing_hmma_fp32_traits, Layout_a> const& a, Fragment_b<Turing_hmma_fp32_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(0)), "+f"(elt(1)), "+f"(elt(2)), "+f"(elt(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(0)));
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(4)), "+f"(elt(5)), "+f"(elt(6)), "+f"(elt(7))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(b.reg(1)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Turing_imma_int8_int32_traits> : public Fragment<int32_t, 8>
{

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Turing_imma_int8_int32_traits, Row>;
    using Fragment_b = Fragment_b<Turing_imma_int8_int32_traits, Col>;

    // IMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            asm volatile(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \n"
                "    {%0, %1}, \n"
                "    {%2}, \n"
                "    {%3}, \n"
                "    {%0, %1}; \n"
                : "+r"(this->reg(2 * i + 0)), "+r"(this->reg(2 * i + 1))
                : "r"(a.reg(i / 2)), "r"(b.reg(i % 2)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Ampere_hmma_fp16_traits> : public Fragment<uint16_t, 8>
{

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(
        Fragment_a<Ampere_hmma_fp16_traits, Layout_a> const& a, Fragment_b<Ampere_hmma_fp16_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1}, \n"
            "    {%2, %3, %4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(0)), "+r"(reg(1))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \n"
            "    {%0, %1}, \n"
            "    {%2, %3, %4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(2)), "+r"(reg(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(2)), "r"(b.reg(3)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// BF16 MMA must accumulate with at least FP32
template <>
struct Fragment_accumulator<Ampere_hmma_bf16_bf16_traits> : public Fragment<bf16_t, 8>
{

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(Fragment_a<Ampere_hmma_bf16_bf16_traits, Layout_a> const& a,
        Fragment_b<Ampere_hmma_bf16_bf16_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "    {%0, %1}, \n"
            "    {%2, %3, %4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(0)), "+r"(reg(1))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "    {%0, %1}, \n"
            "    {%2, %3, %4, %5}, \n"
            "    {%6, %7}, \n"
            "    {%0, %1}; \n"
            : "+r"(reg(2)), "+r"(reg(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(2)), "r"(b.reg(3)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Ampere_hmma_fp32_traits> : public Fragment<float, 8>
{

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    inline __device__ void mul(float const other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) *= other;
        }
    }

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(
        Fragment_a<Ampere_hmma_fp32_traits, Layout_a> const& a, Fragment_b<Ampere_hmma_fp32_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(0)), "+f"(elt(1)), "+f"(elt(2)), "+f"(elt(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(4)), "+f"(elt(5)), "+f"(elt(6)), "+f"(elt(7))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(2)), "r"(b.reg(3)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// BF16 MMA must accumulate with at least FP32
template <>
struct Fragment_accumulator<Ampere_hmma_bf16_traits> : public Fragment<float, 8>
{

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    inline __device__ void mul(float const other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) *= other;
        }
    }

    // Do the HMMA.
    template <typename Layout_a, typename Layout_b>
    inline __device__ void mma(
        Fragment_a<Ampere_hmma_bf16_traits, Layout_a> const& a, Fragment_b<Ampere_hmma_bf16_traits, Layout_b> const& b)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(0)), "+f"(elt(1)), "+f"(elt(2)), "+f"(elt(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(0)), "r"(b.reg(1)));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5, %6, %7}, \n"
            "    {%8, %9}, \n"
            "    {%0, %1, %2, %3}; \n"
            : "+f"(elt(4)), "+f"(elt(5)), "+f"(elt(6)), "+f"(elt(7))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(2)), "r"(b.reg(3)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Ampere_imma_int8_int32_traits> : public Fragment<int32_t, 8>
{

    // The base class.
    using Base = Fragment<int32_t, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Ampere_imma_int8_int32_traits, Row>;
    using Fragment_b = Fragment_b<Ampere_imma_int8_int32_traits, Col>;

    // IMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \n"
                "    {%0, %1, %2, %3}, \n"
                "    {%4, %5, %6, %7}, \n"
                "    {%8, %9}, \n"
                "    {%0, %1, %2, %3}; \n"
                : "+r"(reg(i * 4 + 0)), "+r"(reg(i * 4 + 1)), "+r"(reg(i * 4 + 2)), "+r"(reg(i * 4 + 3))
                : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(i * 2)), "r"(b.reg(i * 2 + 1)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Ada_qmma_e4m3_fp32_traits> : public Fragment<float, 8>
{

    // The base class.
    using Base = Fragment<float, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Ada_qmma_e4m3_fp32_traits, Row>;
    using Fragment_b = Fragment_b<Ada_qmma_e4m3_fp32_traits, Col>;

    // IMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \n"
                "    {%0, %1, %2, %3}, \n"
                "    {%4, %5, %6, %7}, \n"
                "    {%8, %9}, \n"
                "    {%0, %1, %2, %3}; \n"
                : "+r"(reg(i * 4 + 0)), "+r"(reg(i * 4 + 1)), "+r"(reg(i * 4 + 2)), "+r"(reg(i * 4 + 3))
                : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(i * 2)), "r"(b.reg(i * 2 + 1)));
#else
            asm volatile("trap;\n");
#endif
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Fragment_accumulator<Ada_qmma_e4m3_fp16_traits> : public Fragment<uint16_t, 8>
{

    // The base class.
    using Base = Fragment<uint16_t, 8>;

    // The fragments.
    using Fragment_a = Fragment_a<Ada_qmma_e4m3_fp16_traits, Row>;
    using Fragment_b = Fragment_b<Ada_qmma_e4m3_fp16_traits, Col>;

    // IMMA.
    inline __device__ void mma(Fragment_a const& a, Fragment_b const& b)
    {
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 \n"
                "    {%0, %1}, \n"
                "    {%2, %3, %4, %5}, \n"
                "    {%6, %7}, \n"
                "    {%0, %1}; \n"
                : "+r"(reg(i * 2 + 0)), "+r"(reg(i * 2 + 1))
                : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)), "r"(b.reg(i * 2)), "r"(b.reg(i * 2 + 1)));
#else
            asm volatile("trap;\n");
#endif
        }
    }
};

template <typename Traits, typename Cta_tile, bool Sage = false>
struct Tile_o_normalizer
{

    // The fragment accumulator.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = 2 * MMAS_M
    };

    // The number of registers per thread
    enum
    {
        REGS_PER_THREAD = 4
    };

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // softmax data bytes
    enum
    {
        BYTES_PER_ELEMENT = sizeof(float)
    };

    // Initialize the attention sinks.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : attention_sink_value_(params.attention_sinks != nullptr ? params.attention_sinks[binfo.bidh] : -FLT_MAX)
    {
    }

    // Update the sum when attention sinks are used.
    inline __device__ void update_sum(float const (&max)[ROWS_PER_THREAD], float (&sum)[ROWS_PER_THREAD])
    {
#pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i)
        {
            sum[i] += expf(attention_sink_value_ - max[i]);
        }
    }

    // Update o.
    inline __device__ void update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float (&curr_max)[ROWS_PER_THREAD],
        float const (&prev_max)[ROWS_PER_THREAD], float (&sum)[ROWS_PER_THREAD])
    {
#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            uint32_t alpha[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                curr_max[jj] = fmax(prev_max[jj], curr_max[jj]);
                float a = expf(prev_max[jj] - curr_max[jj]);
                sum[jj] *= a;
                // Convert back to FP16x2.
                alpha[ii] = fmha::float2_to_half2(a, a);
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators in FP16x2.
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hmul2(alpha[ii & 1], acc_o_pair);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float alpha[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                curr_max[jj] = fmax(prev_max[jj], curr_max[jj]);
                alpha[ii] = expf(prev_max[jj] - curr_max[jj]);
                sum[jj] *= alpha[ii];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The registers.
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Do the math in Fp32.
                    acc_o_pair.x = alpha[ii & 1] * acc_o_pair.x;
                    acc_o_pair.y = alpha[ii & 1] * acc_o_pair.y;

                    // Convert back to Fp16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }

    // Update o.
    inline __device__ void final_update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float (&sum)[ROWS_PER_THREAD])
    {

#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            uint32_t beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                float b = (sum[jj] == 0.f || sum[jj] != sum[jj]) ? 1.f : 1.f / sum[jj];
                // Convert back to FP16x2.
                beta[ii] = fmha::float2_to_half2(b, b);
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators in FP16x2.
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hmul2(acc_o_pair, beta[ii & 1]);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The diviser.
                beta[ii] = (sum[jj] == 0.f || sum[jj] != sum[jj]) ? 1.f : 1.f / sum[jj];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The registers.
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Do the math in Fp32.
                    acc_o_pair.x = acc_o_pair.x * beta[ii & 1];
                    acc_o_pair.y = acc_o_pair.y * beta[ii & 1];

                    // Convert back to Fp16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }

    // Attention sink value.
    float attention_sink_value_;
};

template <typename Traits, typename Cta_tile>
struct Tile_o_normalizer_fp32
{

    // The fragment accumulator.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in the M dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    // The number of MMAs in the N dimension.
    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = 2 * MMAS_M
    };

    // The number of registers per thread.
    enum
    {
        REGS_PER_THREAD = 8
    };

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // softmax data bytes
    enum
    {
        BYTES_PER_ELEMENT = sizeof(float)
    };

    // Initialize the attention sinks.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer_fp32(Params const& params, Block_info const& binfo)
        : attention_sink_value_(params.attention_sinks != nullptr ? params.attention_sinks[binfo.bidh] : -FLT_MAX)
    {
    }

    // Update the sum when attention sinks are used.
    inline __device__ void update_sum(float const (&max)[ROWS_PER_THREAD], float (&sum)[ROWS_PER_THREAD])
    {
#pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i)
        {
            sum[i] += expf(attention_sink_value_ - max[i]);
        }
    }

    // Update o.
    inline __device__ void update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float (&curr_max)[ROWS_PER_THREAD],
        float const (&prev_max)[ROWS_PER_THREAD], float (&sum)[ROWS_PER_THREAD])
    {

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float alpha[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                curr_max[jj] = fmax(prev_max[jj], curr_max[jj]);
                alpha[ii] = expf(prev_max[jj] - curr_max[jj]);
                sum[jj] *= alpha[ii];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The register     for O.
                    float acc_o_f = acc_o[mi][ni].elt(ii);
                    // Compute the next accumulator.
                    acc_o_f = alpha[(ii & 2) / 2] * acc_o_f;
                    // Update the accumulator.
                    acc_o[mi][ni].elt(ii) = acc_o_f;
                }
            }
        }
    }

    // Update o after P * V
    inline __device__ void final_update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float (&sum)[ROWS_PER_THREAD])
    {

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;

                // The diviser.
                beta[ii] = (sum[jj] == 0.f || sum[jj] != sum[jj]) ? 1.f : 1.f / sum[jj];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The register for O.
                    float acc_o_f = acc_o[mi][ni].elt(ii);
                    // Compute the next accumulator.
                    acc_o_f = acc_o_f * beta[(ii & 2) / 2];
                    // Update the accumulator.
                    acc_o[mi][ni].elt(ii) = acc_o_f;
                }
            }
        }
    }

    // Attention sink value.
    float attention_sink_value_;
};

template <typename Cta_tile>
struct Tile_o_normalizer<Ampere_hmma_fp32_traits, Cta_tile>
    : public Tile_o_normalizer_fp32<Ampere_hmma_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Tile_o_normalizer_fp32<Traits, Cta_tile>;

    // The ctor.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }
};

template <typename Cta_tile>
struct Tile_o_normalizer<Ampere_hmma_bf16_traits, Cta_tile>
    : public Tile_o_normalizer_fp32<Ampere_hmma_bf16_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Tile_o_normalizer_fp32<Traits, Cta_tile>;

    // The ctor.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }
};

// The attention sinks are not enabled for Volta.
template <typename Cta_tile>
struct Tile_o_normalizer<Volta_hmma_fp16_16x16x16_traits, Cta_tile>
{

    // The traits.
    using Traits = Volta_hmma_fp16_16x16x16_traits;

    // The fragments.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = MMAS_M
    };

    // The number of registers per thread
    enum
    {
        REGS_PER_THREAD = 8
    };

    // Update o.
    inline __device__ void update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float (&curr_max)[ROWS_PER_THREAD],
        float const (&prev_max)[ROWS_PER_THREAD], float (&sum)[ROWS_PER_THREAD])
    {
#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors.
            uint32_t alpha;
            // Update the curr_max.
            curr_max[mi] = fmax(prev_max[mi], curr_max[mi]);
            // The multiplier.
            float a = expf(prev_max[mi] - curr_max[mi]);
            // The accumulated sum.
            sum[mi] *= a;
            // Convert back to FP16.
            alpha = fmha::float2_to_half2(a, a);

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators packed in FP16x2.
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hmul2(acc_o_pair, alpha);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Update the curr_max.
            curr_max[mi] = fmax(prev_max[mi], curr_max[mi]);
            // The multiplier.
            float alpha = expf(prev_max[mi] - curr_max[mi]);
            // The accumulated sum.
            sum[mi] *= alpha;

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators. Convert from FP16x2 to FP32x2.
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Apply the scaling.
                    acc_o_pair.x = alpha * acc_o_pair.x;
                    acc_o_pair.y = alpha * acc_o_pair.y;

                    // Update the register after converting back to FP16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }

    // Update o.
    inline __device__ void final_update(Fragment_accu (&acc_o)[MMAS_M][MMAS_N], float const (&sum)[ROWS_PER_THREAD])
    {
#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors.
            uint32_t beta;
            // The divisor.
            float b = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
            // Convert back to FP16.
            beta = fmha::float2_to_half2(b, b);

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators packed in FP16x2.
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hmul2(acc_o_pair, beta);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // The divisor.
            float beta = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The registers.
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Do the math in Fp32.
                    acc_o_pair.x = acc_o_pair.x * beta;
                    acc_o_pair.y = acc_o_pair.y * beta;

                    // Convert back to Fp16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }
};

template <typename Cta_tile>
struct Tile_o_normalizer<Ada_qmma_e4m3_fp32_traits, Cta_tile>
    : public Tile_o_normalizer_fp32<Ada_qmma_e4m3_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Tile_o_normalizer_fp32<Traits, Cta_tile>;

    // The ctor.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    // Update the sum.
    inline __device__ void update_sum(float const (&max)[Base::ROWS_PER_THREAD], float (&sum)[Base::ROWS_PER_THREAD])
    {
// Take the log2f(Traits::SOFTMAX_FP_QUANT_SCALE) into account as the same scale has been applied to sum.
#pragma unroll
        for (int i = 0; i < Base::ROWS_PER_THREAD; ++i)
        {
            sum[i] += expf(this->attention_sink_value_ - max[i]) * Traits::SOFTMAX_FP_QUANT_SCALE;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Tile_o_normalizer<Ada_qmma_e4m3_fp32_traits, Cta_tile, true>
    : public Tile_o_normalizer_fp32<Ada_qmma_e4m3_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Tile_o_normalizer_fp32<Traits, Cta_tile>;

    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in the M dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    // The number of MMAs in the N dimension.
    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of registers per thread.
    enum
    {
        REGS_PER_THREAD = 8
    };

    // The ctor.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_normalizer(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    inline __device__ void merge(Fragment_accu (&acc_dst)[MMAS_M][MMAS_N], Fragment_accu (&acc_src)[MMAS_M][MMAS_N])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    acc_dst[mi][ni].elt(ii) += acc_src[mi][ni].elt(ii);
                }
            }
        }
    }

    template <typename Params>
    inline __device__ void move_to_first_block(Params const& params, int bidb, int bidh)
    {
        int scale_iter = bidb * params.h * params.sage.v.max_nblock + bidh * params.sage.v.max_nblock;

        params_scale_v_iter = reinterpret_cast<float const*>(params.sage.v.scales + scale_iter);
        params_scale_v_ = __ldg(params_scale_v_iter);
    }

    inline __device__ void move_to_next_block()
    {
        params_scale_v_iter += 1;
        params_scale_v_ = __ldg(params_scale_v_iter);
    }

    inline __device__ void apply_scale(Fragment_accu (&acc_o)[MMAS_M][MMAS_N])
    {
        float const scale = reinterpret_cast<float const&>(params_scale_v_);

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    float acc_o_f = acc_o[mi][ni].elt(ii);
                    acc_o_f = scale * acc_o_f;
                    acc_o[mi][ni].elt(ii) = acc_o_f;
                }
            }
        }
    }

    float const* params_scale_v_iter;
    float params_scale_v_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Mma_tile>
struct Softmax_saver
{

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = 2 * MMAS_M
    };

    // The number of registers per thread
    enum
    {
        REGS_PER_THREAD = 4
    };

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // softmax data bytes
    enum
    {
        BYTES_PER_ELEMENT = sizeof(float)
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Softmax_saver(Params const& params, Block_info const& binfo)
        : actual_q_len_(binfo.actual_q_seqlen)
        , softmax_sum_ptr_(reinterpret_cast<char*>(params.softmax_stats_ptr))
        , softmax_stats_stride_in_bytes_(params.softmax_stats_stride_in_bytes)
    {
        softmax_max_ptr_ = reinterpret_cast<char*>(params.softmax_stats_ptr);

        int warp = threadIdx.x / Cta_tile::THREADS_PER_WARP;
        int lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;
        // MMA row0 index (8x4 thread layout)

        int m_per_mma = 32 / Mma_tile::THREADS_PER_MMA_N * 2;
        row0_ = (warp % WARPS_M) * m_per_mma + (lane / 4);
        // Decide whether to store the lse values
        store_softmax_ = (lane % 4 == 0 && int(warp / WARPS_M) == 0);

        // assume fixed seq length for the batch
        size_t const bh_offset = (binfo.sum_s * params.h + binfo.bidh) * sizeof(float) * 2;
        softmax_max_ptr_ += bh_offset + row0_ * params.softmax_stats_stride_in_bytes;
        softmax_sum_ptr_ += bh_offset + row0_ * params.softmax_stats_stride_in_bytes + sizeof(float);
    };

    inline __device__ void store(int q_loop, float* p_sum, float* p_max)
    {

        if (store_softmax_)
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                float sum0 = p_sum[mi * 2];
                float sum1 = p_sum[mi * 2 + 1];
                float max0 = p_max[mi * 2];
                float max1 = p_max[mi * 2 + 1];

                int row_offset = q_loop * Cta_tile::M + mi * Mma_tile::M_PER_MMA_PER_CTA;
                if (row0_ + row_offset < actual_q_len_)
                {
                    fmha::stg(softmax_max_ptr_ + row_offset * softmax_stats_stride_in_bytes_, max0);
                    fmha::stg(softmax_sum_ptr_ + row_offset * softmax_stats_stride_in_bytes_, sum0);
                }
                if (row0_ + row_offset + 8 < actual_q_len_)
                {
                    fmha::stg(softmax_max_ptr_ + (row_offset + 8) * softmax_stats_stride_in_bytes_, max1);
                    fmha::stg(softmax_sum_ptr_ + (row_offset + 8) * softmax_stats_stride_in_bytes_, sum1);
                }
            }
        }
    }

    // ptr (total_token_q, h, 2) float
    char* softmax_sum_ptr_ = nullptr;
    char* softmax_max_ptr_ = nullptr;

    // the first row's idx
    int row0_;
    // actual seq length
    int const actual_q_len_ = 0;
    int const softmax_stats_stride_in_bytes_ = 0;

    // store lse or not
    bool store_softmax_ = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Flash Attention: default applied to Turing, Ampere fp16 traits

template <typename Traits, typename Cta_tile>
struct Fragment_updater
{

    // The fragment accumulator.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = 2 * MMAS_M
    };

    // The number of registers per thread
    enum
    {
        REGS_PER_THREAD = 4
    };

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // softmax data bytes
    enum
    {
        BYTES_PER_ELEMENT = sizeof(float)
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater(Params const& params, Block_info const& binfo)
        : actual_seqlen_(binfo.actual_seqlen)
        , softmax_lse_ptr_(reinterpret_cast<char*>(params.lse_ptr)) // [b, h, s]
    {
        int warp = threadIdx.x / Cta_tile::THREADS_PER_WARP;
        int lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;
        // MMA row0 index (8x4 thread layout)
        row0_ = (warp % WARPS_M) * Mma_tile::M_PER_MMA + (lane / 4);
        // Decide whether to store the lse values
        store_lse_ = (lane % 4 == 0 && int(warp / WARPS_M) == 0);

        // assume fixed seq length for the batch
        size_t const bh_offset = (binfo.bidb * params.h + binfo.bidh) * binfo.actual_seqlen * BYTES_PER_ELEMENT;
        softmax_lse_ptr_ += bh_offset + row0_ * BYTES_PER_ELEMENT;
    };

    // init all statistics
    inline __device__ Fragment_updater()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            curr_max_[row_i] = -HUGE_VALF;
            prev_max_[row_i] = -HUGE_VALF;
            prev_sum_[row_i] = 0.0f;
            curr_sum_[row_i] = 0.0f;
        }
    }

    // Update o.
    inline __device__ void update_o(
        Fragment_accu (&acc_o)[MMAS_M][MMAS_N], Fragment_accu const (&local_acc_o)[MMAS_M][MMAS_N])
    {
#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            uint32_t alpha[2], beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                float a = prev_sum_[jj] * __expf(prev_max_[jj] - curr_max_[jj]);
                // The diviser.
                float b = (curr_sum_[jj] == 0.f || curr_sum_[jj] != curr_sum_[jj]) ? 1.f : 1.f / curr_sum_[jj];
                // Convert back to FP16x2.
                alpha[ii] = fmha::float2_to_half2(a, a);
                beta[ii] = fmha::float2_to_half2(b, b);
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators in FP16x2.
                    uint32_t local_o_pair = local_acc_o[mi][ni].reg(ii);
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hfma2(alpha[ii & 1], acc_o_pair, local_o_pair);
                    acc_o_pair = fmha::hmul2(acc_o_pair, beta[ii & 1]);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float alpha[2], beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                alpha[ii] = prev_sum_[jj] * __expf(prev_max_[jj] - curr_max_[jj]);
                // The diviser.
                beta[ii] = (curr_sum_[jj] == 0.f || curr_sum_[jj] != curr_sum_[jj]) ? 1.f : 1.f / curr_sum_[jj];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The registers.
                    float2 local_o_pair = fmha::half2_to_float2(local_acc_o[mi][ni].reg(ii));
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Do the math in Fp32.
                    acc_o_pair.x = (alpha[ii & 1] * acc_o_pair.x + local_o_pair.x) * beta[ii & 1];
                    acc_o_pair.y = (alpha[ii & 1] * acc_o_pair.y + local_o_pair.y) * beta[ii & 1];

                    // Convert back to Fp16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }

    // Update max scale
    inline __device__ void update_acc_max()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            float pre_curr_max_ = curr_max_[row_i];
            curr_max_[row_i] = fmaxf(prev_max_[row_i], curr_max_[row_i]);
            prev_max_[row_i] = pre_curr_max_;
        }
    }

    // Update max scale
    inline __device__ void update_acc_sum()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            float pre_curr_sum_ = curr_sum_[row_i];
            curr_sum_[row_i] = __expf(prev_max_[row_i] - curr_max_[row_i]) * curr_sum_[row_i] + prev_sum_[row_i];
            prev_sum_[row_i] = pre_curr_sum_;
        }
    }

    inline __device__ void store(int q_loop)
    {

        if (store_lse_)
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                float row0_lse = curr_max_[mi * 2] + __logf(curr_sum_[mi * 2]);
                float row1_lse = curr_max_[mi * 2 + 1] + __logf(curr_sum_[mi * 2 + 1]);
                int row_offset = q_loop * Cta_tile::M + mi * Mma_tile::M_PER_MMA_PER_CTA;
                if (row0_ + row_offset < actual_seqlen_)
                {
                    fmha::stg(softmax_lse_ptr_ + row_offset * BYTES_PER_ELEMENT, row0_lse);
                }
                if (row0_ + row_offset + 8 < actual_seqlen_)
                {
                    fmha::stg(softmax_lse_ptr_ + (row_offset + 8) * BYTES_PER_ELEMENT, row1_lse);
                }
            }
        }
    }

    // Update scales.
    float curr_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    float curr_sum_[ROWS_PER_THREAD] = {0};
    float prev_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    ;
    float prev_sum_[ROWS_PER_THREAD] = {0};

    // ptr
    char* softmax_lse_ptr_ = nullptr;

    // the first row's idx
    int row0_ = 0;
    // actual seq length
    int const actual_seqlen_ = 0;

    // store lse or not
    bool store_lse_ = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Flash attention to update the accumulators in the 2nd GEMM when we accumulate in FP32.
// Support both hmma_fp32 and ampere_hmma_bf16
template <typename Traits, typename Cta_tile>
struct Fragment_updater_ampere_fp32
{

    // The fragment accumulator.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in the M dimension.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    // The number of MMAs in the N dimension.
    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = 2 * MMAS_M
    };

    // The number of registers per thread.
    enum
    {
        REGS_PER_THREAD = 8
    };

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // softmax data bytes
    enum
    {
        BYTES_PER_ELEMENT = sizeof(float)
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater_ampere_fp32(Params const& params, Block_info const& binfo)
        : actual_seqlen_(binfo.actual_seqlen)
        , softmax_lse_ptr_(reinterpret_cast<char*>(params.lse_ptr)) // [b, h, s]
    {
        int warp = threadIdx.x / Cta_tile::THREADS_PER_WARP;
        int lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;
        // MMA row0 index (8x4 thread layout)
        row0_ = (warp % WARPS_M) * Mma_tile::M_PER_MMA + (lane / 4);
        // Decide whether to store the lse values
        store_lse_ = (lane % 4 == 0 && int(warp / WARPS_M) == 0);

        // assume fixed seq length for the batch
        size_t const bh_offset = (binfo.bidb * params.h + binfo.bidh) * binfo.actual_seqlen * BYTES_PER_ELEMENT;
        softmax_lse_ptr_ += bh_offset + row0_ * BYTES_PER_ELEMENT;
    };

    // init all statistics
    inline __device__ Fragment_updater_ampere_fp32()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            curr_max_[row_i] = -HUGE_VALF;
            prev_max_[row_i] = -HUGE_VALF;
            prev_sum_[row_i] = 0.0f;
            curr_sum_[row_i] = 0.0f;
        }
    }

    // Update o after P * V
    inline __device__ void update_o(
        Fragment_accu (&acc_o)[MMAS_M][MMAS_N], Fragment_accu const (&local_acc_o)[MMAS_M][MMAS_N])
    {

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float alpha[2], beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                alpha[ii] = prev_sum_[jj] * __expf(prev_max_[jj] - curr_max_[jj]);
                // The diviser.
                beta[ii] = (curr_sum_[jj] == 0.f || curr_sum_[jj] != curr_sum_[jj]) ? 1.f : 1.f / curr_sum_[jj];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {

                    // The register from P.
                    float local_acc_o_f = local_acc_o[mi][ni].elt(ii);
                    // The register for O.
                    float acc_o_f = acc_o[mi][ni].elt(ii);
                    // Compute the next accumulator.
                    acc_o_f = (alpha[(ii & 2) / 2] * acc_o_f + local_acc_o_f) * beta[(ii & 2) / 2];
                    // Update the accumulator.
                    acc_o[mi][ni].elt(ii) = acc_o_f;
                }
            }
        }
    }

    // Update o before P * V
    inline __device__ void update_o(Fragment_accu (&acc_o)[MMAS_M][MMAS_N])
    {

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors for the 2 rows.
            float alpha[2], beta[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii)
            {
                // The row.
                int jj = 2 * mi + ii;
                // The multiplier.
                alpha[ii] = prev_sum_[jj] * __expf(prev_max_[jj] - curr_max_[jj]);
                // The diviser.
                beta[ii] = (curr_sum_[jj] == 0.f || curr_sum_[jj] != curr_sum_[jj]) ? 1.f : 1.f / curr_sum_[jj];
            }

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {

                    // The register for O.
                    float acc_o_f = acc_o[mi][ni].elt(ii);
                    // Compute the next accumulator.
                    acc_o_f = alpha[(ii & 2) / 2] * acc_o_f * beta[(ii & 2) / 2];
                    // Update the accumulator.
                    acc_o[mi][ni].elt(ii) = acc_o_f;
                }
            }
        }
    }

    // Update max scale
    inline __device__ void update_acc_max()
    {
#pragma unroll
        for (int ii = 0; ii < ROWS_PER_THREAD; ++ii)
        {
            float curr_max = curr_max_[ii];
            curr_max_[ii] = fmaxf(prev_max_[ii], curr_max);
            prev_max_[ii] = curr_max;
        }
    }

    // Update max scale
    inline __device__ void update_acc_sum()
    {
#pragma unroll
        for (int ii = 0; ii < ROWS_PER_THREAD; ++ii)
        {
            float curr_sum = curr_sum_[ii];
            curr_sum_[ii] = __expf(prev_max_[ii] - curr_max_[ii]) * curr_sum_[ii] + prev_sum_[ii];
            prev_sum_[ii] = curr_sum;
        }
    }

    inline __device__ void store(int q_loop)
    {

        if (store_lse_)
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                float row0_lse = curr_max_[mi * 2] + __logf(curr_sum_[mi * 2]);
                float row1_lse = curr_max_[mi * 2 + 1] + __logf(curr_sum_[mi * 2 + 1]);
                int row_offset = q_loop * Cta_tile::M + mi * Mma_tile::M_PER_MMA_PER_CTA;
                if (row0_ + row_offset < actual_seqlen_)
                {
                    fmha::stg(softmax_lse_ptr_ + row_offset * BYTES_PER_ELEMENT, row0_lse);
                }
                if (row0_ + row_offset + 8 < actual_seqlen_)
                {
                    fmha::stg(softmax_lse_ptr_ + (row_offset + 8) * BYTES_PER_ELEMENT, row1_lse);
                }
            }
        }
    }

    // Update scales.
    float curr_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    float curr_sum_[ROWS_PER_THREAD] = {0};
    float prev_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    float prev_sum_[ROWS_PER_THREAD] = {0};

    // ptr
    char* softmax_lse_ptr_ = nullptr;

    // the first row's idx
    int row0_ = 0;
    // actual seq length
    int const actual_seqlen_ = 0;

    // store lse or not
    bool store_lse_ = false;
};

template <typename Cta_tile>
struct Fragment_updater<Ampere_hmma_fp32_traits, Cta_tile>
    : public Fragment_updater_ampere_fp32<Ampere_hmma_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Fragment_updater_ampere_fp32<Traits, Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    // Default ctor
    Fragment_updater() = default;
};

template <typename Cta_tile>
struct Fragment_updater<Ampere_hmma_bf16_traits, Cta_tile>
    : public Fragment_updater_ampere_fp32<Ampere_hmma_bf16_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Fragment_updater_ampere_fp32<Traits, Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    // Default ctor
    Fragment_updater() = default;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Fragment_updater<Turing_hmma_fp32_traits, Cta_tile>
    : public Fragment_updater_ampere_fp32<Turing_hmma_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Turing_hmma_fp32_traits;
    // The base class.
    using Base = Fragment_updater_ampere_fp32<Traits, Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    // Default ctor
    Fragment_updater() = default;
};

template <typename Cta_tile>
struct Fragment_updater<Ada_qmma_e4m3_fp32_traits, Cta_tile>
    : public Fragment_updater_ampere_fp32<Ada_qmma_e4m3_fp32_traits, Cta_tile>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Fragment_updater_ampere_fp32<Traits, Cta_tile>;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Fragment_updater(Params const& params, Block_info const& binfo)
        : Base(params, binfo)
    {
    }

    // Default ctor
    Fragment_updater() = default;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Fragment_updater<Volta_hmma_fp16_16x16x16_traits, Cta_tile>
{

    // The traits.
    using Traits = Volta_hmma_fp16_16x16x16_traits;

    // The fragments.
    using Fragment_accu = Fragment_accumulator<Traits>;

    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::VALID_MMAS_N
    };

    // The number of rows per thread.
    enum
    {
        ROWS_PER_THREAD = MMAS_M
    };

    // The number of registers per thread
    enum
    {
        REGS_PER_THREAD = 8
    };

    // init all statistics
    inline __device__ Fragment_updater()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            curr_max_[row_i] = -HUGE_VALF;
            prev_max_[row_i] = -HUGE_VALF;
            prev_sum_[row_i] = 0.0f;
            curr_sum_[row_i] = 0.0f;
        }
    }

    // Update o.
    inline __device__ void update_o(
        Fragment_accu (&acc_o)[MMAS_M][MMAS_N], Fragment_accu const (&local_acc_o)[MMAS_M][MMAS_N])
    {
#ifdef HALF_ACCUMULATION_FOR_FLASH_ATTENTION // Half accumulation

#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // Precompute the scaling factors.
            uint32_t alpha, beta;
            // The multiplier.
            float a = prev_sum_[mi] * __expf(prev_max_[mi] - curr_max_[mi]);
            // The diviser.
            float b = (curr_sum_[mi] == 0.f || curr_sum_[mi] != curr_sum_[mi]) ? 1.f : 1.f / curr_sum_[mi];
            // Convert back to FP16.
            alpha = fmha::float2_to_half2(a, a);
            beta = fmha::float2_to_half2(b, b);

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators packed in FP16x2.
                    uint32_t local_o_pair = local_acc_o[mi][ni].reg(ii);
                    uint32_t acc_o_pair = acc_o[mi][ni].reg(ii);

                    // Apply the scaling.
                    acc_o_pair = fmha::hmul2(fmha::hfma2(alpha, acc_o_pair, local_o_pair), beta);

                    // Update the register.
                    acc_o[mi][ni].reg(ii) = acc_o_pair;
                }
            }
        }
#else // Float accumulation
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {

            // The multiplier.
            float alpha = prev_sum_[mi] * __expf(prev_max_[mi] - curr_max_[mi]);
            // The diviser.
            float beta = (curr_sum_[mi] == 0.f || curr_sum_[mi] != curr_sum_[mi]) ? 1.f : 1.f / curr_sum_[mi];

#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < REGS_PER_THREAD; ++ii)
                {
                    // The accumulators. Convert from FP16x2 to FP32x2.
                    float2 local_o_pair = fmha::half2_to_float2(local_acc_o[mi][ni].reg(ii));
                    float2 acc_o_pair = fmha::half2_to_float2(acc_o[mi][ni].reg(ii));

                    // Apply the scaling.
                    acc_o_pair.x = (alpha * acc_o_pair.x + local_o_pair.x) * beta;
                    acc_o_pair.y = (alpha * acc_o_pair.y + local_o_pair.y) * beta;

                    // Update the register after converting back to FP16x2.
                    acc_o[mi][ni].reg(ii) = fmha::float2_to_half2(acc_o_pair);
                }
            }
        }
#endif // defined HALF_ACCUMULATION_FOR_FLASH_ATTENTION
    }

    // Update max scale
    inline __device__ void update_acc_max()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            float pre_curr_max_ = curr_max_[row_i];
            curr_max_[row_i] = fmaxf(prev_max_[row_i], curr_max_[row_i]);
            prev_max_[row_i] = pre_curr_max_;
        }
    }

    // Update max scale
    inline __device__ void update_acc_sum()
    {
#pragma unroll
        for (int row_i = 0; row_i < ROWS_PER_THREAD; ++row_i)
        {
            float pre_curr_sum_ = curr_sum_[row_i];
            curr_sum_[row_i] = __expf(prev_max_[row_i] - curr_max_[row_i]) * curr_sum_[row_i] + prev_sum_[row_i];
            prev_sum_[row_i] = pre_curr_sum_;
        }
    }

    // updater scales
    float curr_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    float curr_sum_[ROWS_PER_THREAD] = {0};
    float prev_max_[ROWS_PER_THREAD] = {-HUGE_VALF};
    float prev_sum_[ROWS_PER_THREAD] = {0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data_type_, int SIZE_IN_BYTES_>
struct Fragment_from_size_in_bytes
{
    using Type = Fragment<Data_type_, SIZE_IN_BYTES_ / static_cast<int>(sizeof(Data_type_))>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SIZE_IN_BYTES_>
struct Fragment_from_size_in_bytes<bool, SIZE_IN_BYTES_>
{
    using Type = Fragment<bool, SIZE_IN_BYTES_ * 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment, int M, int N>
inline __device__ void clear(Fragment (&frag)[M][N])
{
#pragma unroll
    for (int mi = 0; mi < M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < N; ++ni)
        {
            frag[mi][ni].clear();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Accumulator_type, int WARPS_K>
struct Clear_accumulator
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_K>
struct Clear_accumulator<uint16_t, WARPS_K>
{
    template <typename Acc, int M, int N>
    static inline __device__ void apply(Acc (&acc)[M][N], bool = false)
    {
        fmha::clear(acc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_K>
struct Clear_accumulator<fmha::bf16_t, WARPS_K>
{
    template <typename Acc, int M, int N>
    static inline __device__ void apply(Acc (&acc)[M][N], bool = false)
    {
        fmha::clear(acc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_K>
struct Clear_accumulator<float, WARPS_K>
{
    template <typename Acc, int M, int N>
    static inline __device__ void apply(Acc (&acc)[M][N], bool = false)
    {
        fmha::clear(acc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_K>
struct Clear_accumulator<int32_t, WARPS_K>
{
    template <typename Acc, int M, int N>
    static inline __device__ void apply(Acc (&acc)[M][N], bool enable_i2f_trick = true)
    {
#if defined(USE_I2F_EMULATION_TRICK)
        if (enable_i2f_trick)
        {
#pragma unroll
            for (int mi = 0; mi < M; ++mi)
            {
#pragma unroll
                for (int ni = 0; ni < N; ++ni)
                {
#pragma unroll
                    for (int ii = 0; ii < Acc::NUM_REGS; ++ii)
                    {
                        acc[mi][ni].reg(ii) = uint32_t(FP32_I2F_MAGIC_NUMBER_HEX) / WARPS_K;
                    }
                }
            }
        }
        else
#endif // defined(USE_I2F_EMULATION_TRICK)
        {
            fmha::clear(acc);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
