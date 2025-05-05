/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

// code in this file based on https://github.com/milakov/int_fastdiv
// note: above repo is not added as a submodule because we slightly update APIs

#include <cute/config.hpp>

namespace trtllm::dev
{

class IntFastDiv
{
public:
    // divisor != 0
    CUTE_HOST_DEVICE
    IntFastDiv(int divisor = 0)
        : d(divisor)
    {
        update_magic_numbers();
    }

    CUTE_HOST_DEVICE
    IntFastDiv& operator=(int divisor)
    {
        this->d = divisor;
        update_magic_numbers();
        return *this;
    }

    CUTE_HOST_DEVICE
    operator int() const
    {
        return d;
    }

private:
    int d;
    int M;
    int s;
    int n_add_sign;

    // Hacker's Delight, Second Edition, Chapter 10, Integer Division By Constants
    CUTE_HOST_DEVICE
    void update_magic_numbers()
    {
        if (d == 1)
        {
            M = 0;
            s = -1;
            n_add_sign = 1;
            return;
        }
        else if (d == -1)
        {
            M = 0;
            s = -1;
            n_add_sign = -1;
            return;
        }

        int p;
        unsigned int tmp_ad, tmp_anc, delta, q1, r1, q2, r2, t;
        unsigned const two31 = 0x80000000;
        tmp_ad = (d == 0) ? 1 : abs(d);
        t = two31 + ((unsigned int) d >> 31);
        tmp_anc = t - 1 - t % tmp_ad;
        p = 31;
        q1 = two31 / tmp_anc;
        r1 = two31 - q1 * tmp_anc;
        q2 = two31 / tmp_ad;
        r2 = two31 - q2 * tmp_ad;
        do
        {
            ++p;
            q1 = 2 * q1;
            r1 = 2 * r1;
            if (r1 >= tmp_anc)
            {
                ++q1;
                r1 -= tmp_anc;
            }
            q2 = 2 * q2;
            r2 = 2 * r2;
            if (r2 >= tmp_ad)
            {
                ++q2;
                r2 -= tmp_ad;
            }
            delta = tmp_ad - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        this->M = q2 + 1;
        if (d < 0)
            this->M = -this->M;
        this->s = p - 32;

        if ((d > 0) && (M < 0))
            n_add_sign = 1;
        else if ((d < 0) && (M > 0))
            n_add_sign = -1;
        else
            n_add_sign = 0;
    }

    CUTE_HOST_DEVICE
    friend int operator/(int const divident, IntFastDiv const& divisor);
};

CUTE_HOST_DEVICE
int operator/(int const n, IntFastDiv const& divisor)
{
    int q;
#ifdef __CUDA_ARCH__
    asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(divisor.M), "r"(n));
#else
    q = (((unsigned long long) ((long long) divisor.M * (long long) n)) >> 32);
#endif
    q += n * divisor.n_add_sign;
    if (divisor.s >= 0)
    {
        q >>= divisor.s; // we rely on this to be implemented as arithmetic shift
        q += (((unsigned int) q) >> 31);
    }
    return q;
}

CUTE_HOST_DEVICE
int operator%(int const n, IntFastDiv const& divisor)
{
    int quotient = n / divisor;
    int remainder = n - quotient * divisor;
    return remainder;
}

} // namespace trtllm::dev
