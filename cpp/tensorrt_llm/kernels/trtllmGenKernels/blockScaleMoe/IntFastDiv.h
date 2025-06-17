/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cute/config.hpp>

#include <stdexcept>

namespace trtllm::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// IntFastDiv class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

class IntFastDiv
{
public:
    // we allow a default constructor (and over-writing the divisor with assignment)
    CUTE_HOST
    IntFastDiv()
        : mDivisor(1)
        , mMagicM(0)
        , mMagicS(-1)
        , mAddSign(1)
    {
    }

    CUTE_HOST
    IntFastDiv(int divisor)
        : mDivisor(divisor)
    {
        if (mDivisor == 0)
            throw std::runtime_error("IntFastDiv: cannot divide by 0");
        updateMagicNumbers();
    }

    CUTE_HOST
    IntFastDiv& operator=(int divisor)
    {
        this->mDivisor = divisor;
        if (this->mDivisor == 0)
            throw std::runtime_error("IntFastDiv: cannot divide by 0");
        updateMagicNumbers();
        return *this;
    }

    CUTE_HOST_DEVICE
    operator int() const
    {
        return mDivisor;
    }

private:
    int mDivisor;
    int mMagicM;
    int mMagicS;
    int mAddSign;

    // Hacker'mMagicS Delight, Second Edition, Chapter 10, Integer Division By Constants
    CUTE_HOST
    void updateMagicNumbers()
    {
        if (mDivisor == 1)
        {
            mMagicM = 0;
            mMagicS = -1;
            mAddSign = 1;
            return;
        }
        else if (mDivisor == -1)
        {
            mMagicM = 0;
            mMagicS = -1;
            mAddSign = -1;
            return;
        }

        int p;
        unsigned int tmpAd, tmpAnc, delta, q1, r1, q2, r2, t;
        unsigned const two31 = 0x80000000;
        tmpAd = abs(mDivisor);
        t = two31 + ((unsigned int) mDivisor >> 31);
        tmpAnc = t - 1 - t % tmpAd;
        p = 31;
        q1 = two31 / tmpAnc;
        r1 = two31 - q1 * tmpAnc;
        q2 = two31 / tmpAd;
        r2 = two31 - q2 * tmpAd;
        do
        {
            ++p;
            q1 = 2 * q1;
            r1 = 2 * r1;
            if (r1 >= tmpAnc)
            {
                ++q1;
                r1 -= tmpAnc;
            }
            q2 = 2 * q2;
            r2 = 2 * r2;
            if (r2 >= tmpAd)
            {
                ++q2;
                r2 -= tmpAd;
            }
            delta = tmpAd - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        this->mMagicM = q2 + 1;
        if (mDivisor < 0)
            this->mMagicM = -this->mMagicM;
        this->mMagicS = p - 32;

        if ((mDivisor > 0) && (mMagicM < 0))
            mAddSign = 1;
        else if ((mDivisor < 0) && (mMagicM > 0))
            mAddSign = -1;
        else
            mAddSign = 0;
    }

    CUTE_HOST_DEVICE
    friend int operator/(int const dividend, IntFastDiv const& divisor);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
int operator/(int const dividend, IntFastDiv const& divisor)
{
    int q;
#ifdef __CUDA_ARCH__
    asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(divisor.mMagicM), "r"(dividend));
#else
    q = (((unsigned long long) ((long long) divisor.mMagicM * (long long) dividend)) >> 32);
#endif
    q += dividend * divisor.mAddSign;
    if (divisor.mMagicS >= 0)
    {
        q >>= divisor.mMagicS; // we rely on this to be implemented as arithmetic shift
        q += (((unsigned int) q) >> 31);
    }
    return q;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
int operator%(int const dividend, IntFastDiv const& divisor)
{
    int quotient = dividend / divisor;
    int remainder = dividend - quotient * divisor;
    return remainder;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace trtllm::dev
