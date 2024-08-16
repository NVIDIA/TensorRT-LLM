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

#include <tuple>

#ifdef __CUDACC__ // for CUDA
#define FT_DEV_CEXPR __device__ __host__ inline constexpr
#else
#define FT_DEV_CEXPR inline constexpr
#endif

//----------------------------------------------------------------------------
// Cn: constant integer
//----------------------------------------------------------------------------

template <auto value_>
struct Cn : public std::integral_constant<decltype(value_), value_>
{
};

template <auto value_>
constexpr auto cn = Cn<value_>();

//----------------------------------------------------------------------------
// Operators for Cn
//----------------------------------------------------------------------------

template <auto value_>
FT_DEV_CEXPR auto operator+(Cn<value_>)
{
    return cn<+value_>;
}

template <auto value_>
FT_DEV_CEXPR auto operator-(Cn<value_>)
{
    return cn<-value_>;
}

template <auto value_>
FT_DEV_CEXPR auto operator!(Cn<value_>)
{
    return cn<!value_>;
}

template <auto value_>
FT_DEV_CEXPR auto operator~(Cn<value_>)
{
    return cn<~value_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator+(Cn<a_>, Cn<b_>)
{
    return cn<a_ + b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator-(Cn<a_>, Cn<b_>)
{
    return cn<a_ - b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator*(Cn<a_>, Cn<b_>)
{
    return cn<a_ * b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator/(Cn<a_>, Cn<b_>)
{
    return cn<a_ / b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator%(Cn<a_>, Cn<b_>)
{
    return cn<a_ % b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator<<(Cn<a_>, Cn<b_>)
{
    return cn<(a_ << b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator>>(Cn<a_>, Cn<b_>)
{
    return cn<(a_ >> b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator<(Cn<a_>, Cn<b_>)
{
    return cn<(a_ < b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator<=(Cn<a_>, Cn<b_>)
{
    return cn<(a_ <= b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator>(Cn<a_>, Cn<b_>)
{
    return cn<(a_ > b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator>=(Cn<a_>, Cn<b_>)
{
    return cn<(a_ >= b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator==(Cn<a_>, Cn<b_>)
{
    return cn<(a_ == b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator!=(Cn<a_>, Cn<b_>)
{
    return cn<(a_ != b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator^(Cn<a_>, Cn<b_>)
{
    return cn<a_ ^ b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator&(Cn<a_>, Cn<b_>)
{
    return cn<a_ & b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator&&(Cn<a_>, Cn<b_>)
{
    return cn < a_ && b_ > ;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator|(Cn<a_>, Cn<b_>)
{
    return cn<a_ | b_>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto operator||(Cn<a_>, Cn<b_>)
{
    return cn < a_ || b_ > ;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator*(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator/(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator%(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator<<(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator>>(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator&(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> operator&&(Cn<a_>, B_)
{
    return cn<a_>;
}

template <class A_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ == 0, Cn<b_>> operator*(A_, Cn<b_>)
{
    return cn<b_>;
}

template <class A_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ == +1, Cn<decltype(b_)(0)>> operator%(A_, Cn<b_>)
{
    return cn<decltype(b_)(0)>;
}

template <class A_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ == -1, Cn<decltype(b_)(0)>> operator%(A_, Cn<b_>)
{
    return cn<decltype(b_)(0)>;
}

template <class A_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ == 0, Cn<b_>> operator&(A_, Cn<b_>)
{
    return cn<b_>;
}

template <class A_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ == 0, Cn<b_>> operator&&(A_, Cn<b_>)
{
    return cn<b_>;
}

//----------------------------------------------------------------------------
// div_up & round_up
//----------------------------------------------------------------------------

template <class T_>
FT_DEV_CEXPR auto cexpr_abs(T_ a_) // abs is not constexpr until C++20
{
    return a_ >= cn<0> ? +a_ : -a_;
}

template <class T_, class U_>
FT_DEV_CEXPR auto div_up(T_ a_, U_ b_)
{
    auto tmp = a_ >= cn<0> ? a_ + (cexpr_abs(b_) - cn<1>) : a_ - (cexpr_abs(b_) - cn<1>);

    return tmp / b_;
}

template <class T_, class U_>
FT_DEV_CEXPR auto round_up(T_ a_, U_ b_)
{
    auto tmp = a_ >= cn<0> ? a_ + (cexpr_abs(b_) - cn<1>) : a_ - (cexpr_abs(b_) - cn<1>);

    return tmp - tmp % b_;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto div_up(Cn<a_>, Cn<b_>)
{
    return cn<div_up(a_, b_)>;
}

template <auto a_, auto b_>
FT_DEV_CEXPR auto round_up(Cn<a_>, Cn<b_>)
{
    return cn<round_up(a_, b_)>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> div_up(Cn<a_>, B_)
{
    return cn<a_>;
}

template <auto a_, class B_>
FT_DEV_CEXPR std::enable_if_t<a_ == 0, Cn<a_>> round_up(Cn<a_>, B_)
{
    return cn<a_>;
}

//----------------------------------------------------------------------------
// IsTuple: std::tuple, but not std::pair, std::array, etc.
//----------------------------------------------------------------------------

template <class T_>
struct IsTuple : public std::false_type
{
};

template <class... Ts_>
struct IsTuple<std::tuple<Ts_...>> : public std::true_type
{
};

template <class T_>
struct IsTuple<const T_> : public IsTuple<T_>
{
};

template <class T_>
struct IsTuple<T_&> : public IsTuple<T_>
{
};

template <class T_>
struct IsTuple<T_&&> : public IsTuple<T_>
{
};

template <class T_>
constexpr bool IsTuple_v = IsTuple<T_>::value;

// vim: ts=2 sw=2 sts=2 et sta
