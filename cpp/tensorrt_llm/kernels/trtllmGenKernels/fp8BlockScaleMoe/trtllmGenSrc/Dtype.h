/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "DtypeDecl.h"

//// FIX
#include "tensorrt_llm/common/assert.h" // #include <trtllm/gen/GenCtx.h>
#include "tensorrt_llm/common/logger.h"
#define TLLM_CHECK_ERROR(cond, ...)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "TRTLLM-GEN kernel launch failed");                                            \
        }                                                                                                              \
    } while (0)

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// The supported namespaces.
enum class Ns
{
    None,
    CudaPtx,
    Cute,
    Cutlass,
    CutlassArch,
    CutlassGemm,
    CutlassGemmKernelDetail,
    TrtllmDev
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dtypeToCutlassString(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::Bfloat16: return "cutlass::bfloat16_t";
    case Dtype::Bool: return "bool";
    case Dtype::E0m3: return "cutlass::float_e0m3_t";
    case Dtype::E2m1: return "cutlass::float_e2m1_t";
    case Dtype::E2m3: return "cutlass::float_e2m3_t";
    case Dtype::E3m2: return "cutlass::float_e3m2_t";
    case Dtype::E4m3: return "cutlass::float_e4m3_t";
    case Dtype::E5m2: return "cutlass::float_e5m2_t";
    case Dtype::Fp16: return "cutlass::half_t";
    case Dtype::Fp32: return "float";
    case Dtype::Int8: return "int8_t";
    case Dtype::Int32: return "int32_t";
    case Dtype::Int64: return "int64_t";
    case Dtype::UE8m0: return "cutlass::float_ue8m0_t";
    case Dtype::UInt8: return "uint8_t";
    case Dtype::UInt16: return "uint16_t";
    case Dtype::UInt32: return "uint32_t";
    case Dtype::UInt64: return "uint64_t";
    case Dtype::UInt128: return "cutlass::uint128_t";
    case Dtype::Void: return "void";
    default: TLLM_CHECK_ERROR(false, "Unsupported type"); return "Error";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// For logging and error reporting
inline std::string dtypeToString(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::Bfloat16: return "Bfloat16";
    case Dtype::Bool: return "Bool";
    case Dtype::E0m3: return "E0m3";
    case Dtype::E2m1: return "E2m1";
    case Dtype::E2m3: return "E2m3";
    case Dtype::E3m2: return "E3m2";
    case Dtype::E4m3: return "E4m3";
    case Dtype::E5m2: return "E5m2";
    case Dtype::Fp16: return "Fp16";
    case Dtype::Fp32: return "Fp32";
    case Dtype::Int8: return "Int8";
    case Dtype::Int32: return "Int32";
    case Dtype::Int64: return "Int64";
    case Dtype::UE8m0: return "UE8m0";
    case Dtype::UInt8: return "UInt8";
    case Dtype::UInt16: return "UInt16";
    case Dtype::UInt32: return "UInt32";
    case Dtype::UInt64: return "UInt64";
    case Dtype::UInt128: return "UInt128";
    case Dtype::Void: return "Void";
    default: TLLM_CHECK_ERROR(false, "Unsupported type"); return "Error";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dtypeToPtxString(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::Bfloat16: return "bf16";
    case Dtype::Bool: return "pred";
    case Dtype::E0m3: return "e0m3";
    case Dtype::E2m1: return "e2m1";
    case Dtype::E2m3: return "e2m3";
    case Dtype::E3m2: return "e3m2";
    case Dtype::E4m3: return "e4m3";
    case Dtype::E5m2: return "e5m2";
    case Dtype::Fp16: return "f16";
    case Dtype::Fp32: return "f32";
    case Dtype::Int8: return "s8";
    case Dtype::Int32: return "s32";
    case Dtype::Int64: return "s64";
    case Dtype::UInt8: return "u8";
    case Dtype::UInt16: return "u16";
    case Dtype::UInt32: return "u32";
    case Dtype::UInt64: return "u64";
    case Dtype::UInt128:
    case Dtype::Void:
    default: TLLM_CHECK_ERROR(false, "Unsupported type"); return "Error";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The number of bytes in a data type?
inline int dtypeGetNumBytes(Dtype dtype)
{
    TLLM_CHECK_ERROR(dtypeGetNumBits(dtype) % 8 == 0, "Sub-byte types not supported");
    return dtypeGetNumBits(dtype) / 8;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline Dtype dtypeGetPackedUInt(Dtype dtypeElt, int numElts)
{
    auto const packSizeInBytes = dtypeGetNumBits(dtypeElt) * numElts / /* bits */ 8;
    switch (packSizeInBytes)
    {
    case 1: return Dtype::UInt8;
    case 2: return Dtype::UInt16;
    case 4: return Dtype::UInt32;
    case 8: return Dtype::UInt64;
    case 16: return Dtype::UInt128;
    default: TLLM_CHECK_ERROR(false, "Unsupported pack size in bytes: ", packSizeInBytes);
    }
    return Dtype::UInt8;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline Dtype dtypeFind(std::vector<T> const& v)
{
    TLLM_CHECK_ERROR(v.size() > 0, "Cannot determine the type of elements in an empty vector");
    Dtype dtype = v[0]->getDtype();
    for (size_t ii = 1; ii < v.size(); ++ii)
    {
        Dtype otherDtype = v[ii]->getDtype();
        TLLM_CHECK_ERROR(dtype == otherDtype, "Cannot determine the dtype as types are different");
    }
    return dtype;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline Dtype dtypeFromString(std::string const& str)
{
    if (str == "Bfloat16")
    {
        return Dtype::Bfloat16;
    }
    else if (str == "Bool")
    {
        return Dtype::Bool;
    }
    else if (str == "E0m3")
    {
        return Dtype::E0m3;
    }
    else if (str == "E2m1")
    {
        return Dtype::E2m1;
    }
    else if (str == "E2m3")
    {
        return Dtype::E2m3;
    }
    else if (str == "E3m2")
    {
        return Dtype::E3m2;
    }
    else if (str == "E4m3")
    {
        return Dtype::E4m3;
    }
    else if (str == "E5m2")
    {
        return Dtype::E5m2;
    }
    else if (str == "Fp16")
    {
        return Dtype::Fp16;
    }
    else if (str == "Fp32")
    {
        return Dtype::Fp32;
    }
    else if (str == "Int8")
    {
        return Dtype::Int8;
    }
    else if (str == "Int32")
    {
        return Dtype::Int32;
    }
    else if (str == "Int64")
    {
        return Dtype::Int64;
    }
    else if (str == "UE8m0")
    {
        return Dtype::UE8m0;
    }
    else if (str == "UInt8")
    {
        return Dtype::UInt8;
    }
    else if (str == "UInt16")
    {
        return Dtype::UInt16;
    }
    else if (str == "UInt32")
    {
        return Dtype::UInt32;
    }
    else if (str == "UInt64")
    {
        return Dtype::UInt64;
    }
    else if (str == "UInt128")
    {
        return Dtype::UInt128;
    }
    else if (str == "Void")
    {
        return Dtype::Void;
    }
    else
    {
        TLLM_LOG_ERROR("Unknown Dtype");
    }
    return Dtype::Void;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Does the format use block scaling?
inline bool dtypeIsBlockFmt(Dtype dtype)
{
    constexpr uint32_t kMask = 0xffu << 24;
    return static_cast<bool>((static_cast<uint32_t>(dtype) & kMask) >> 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type a floating-point type?
inline bool dtypeIsFloat(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 16;
    return dtype != Dtype::Void && 0 == (static_cast<uint32_t>(dtype) & kMask);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type an 8-bit floating-point type?
inline bool dtypeIsFp8(Dtype dtype)
{
    return dtype == Dtype::E4m3 || dtype == Dtype::E5m2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type an integer type?
inline bool dtypeIsInt(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 16;
    return (dtype != Dtype::Bool) && (0 != (static_cast<uint32_t>(dtype) & kMask));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type signed?
inline bool dtypeIsSigned(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 20;
    return (0 != (static_cast<uint32_t>(dtype) & kMask));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline Dtype dtypeBlockSfType(Dtype dtype)
{
    TLLM_CHECK_ERROR(dtypeIsBlockFmt(dtype), "Not a block format: ", dtypeToString(dtype));

    switch (dtype)
    {
    case Dtype::E0m3:
    case Dtype::E2m1: return Dtype::E4m3;
    case Dtype::MxE2m1: return Dtype::UE8m0;
    default: TLLM_LOG_ERROR("Unknown scaling factor type"); return Dtype::Void;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class Type
{
public:
    // Ctor.
    Type(Dtype dtype, int numElts = 1)
        : mDtype{dtype}
        , mNumElts{numElts}
    {
    }

    // Are two types equal?
    inline bool equals(Type const& other) const
    {
        return (mDtype == other.mDtype) && (mNumElts == other.mNumElts);
    }

    // The data type.
    inline Dtype getDtype() const
    {
        return mDtype;
    }

    // The number of elements.
    inline int getNumElts() const
    {
        return mNumElts;
    }

    // Is it a bool?
    inline bool isBool() const
    {
        return mDtype == Dtype::Bool;
    }

    // Is it an bool scalar?
    inline bool isBoolScalar() const
    {
        return isBool() && isScalar();
    }

    // Is it an bool vector?
    inline bool isBoolVec() const
    {
        return isBool() && isVec();
    }

    // Is it an float type?
    inline bool isFloat() const
    {
        return dtypeIsFloat(mDtype);
    }

    // Is it an float scalar?
    inline bool isFloatScalar() const
    {
        return isFloat() && isScalar();
    }

    // Is it an float vector?
    inline bool isFloatVec() const
    {
        return isFloat() && isVec();
    }

    // Is it an int type?
    inline bool isInt() const
    {
        return dtypeIsInt(mDtype);
    }

    // Is it an int scalar?
    inline bool isIntScalar() const
    {
        return isInt() && isScalar();
    }

    // Is it an int vector?
    inline bool isIntVec() const
    {
        return isInt() && isVec();
    }

    // Is it a scalar type?
    inline bool isScalar() const
    {
        return mNumElts == 1;
    }

    // Is it a signed type?
    inline bool isSigned() const
    {
        return dtypeIsSigned(mDtype);
    }

    // Is it a vector type?
    inline bool isVec() const
    {
        return mNumElts > 1;
    }

    // Is it the void type?
    inline bool isVoid() const
    {
        return mDtype == Dtype::Void;
    }

private:
    // The data type.
    Dtype const mDtype;
    // The number of elements.
    int const mNumElts;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
