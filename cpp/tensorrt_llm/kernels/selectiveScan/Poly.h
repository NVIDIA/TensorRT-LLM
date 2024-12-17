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
#pragma nv_diag_suppress 177, 550

#include <numeric>

#include "Cn.h"

//----------------------------------------------------------------------------
// Rn: ranged integer (with size and multiplier)
//----------------------------------------------------------------------------

enum Kind
{
    NONE,
    UNROLL,
    ID,
};

template <auto kind_ = NONE, auto size_ = 0, auto multiplier_ = decltype(size_)(1)>
class Ranged
{
    static_assert(multiplier_);

public:
    typedef decltype(size_) type;

    FT_DEV_CEXPR Ranged(type var_ = 0)
        : var(var_)
    {
    }

    static constexpr auto min = type(size_ && multiplier_ < 0 ? size_ - 1 : 0) * multiplier_;
    static constexpr auto max = type(size_ && multiplier_ > 0 ? size_ - 1 : 0) * multiplier_;
    static constexpr auto abs = max - min;

    static constexpr bool minInf = (size_ == 0 && multiplier_ < 0);
    static constexpr bool maxInf = (size_ == 0 && multiplier_ > 0);
    static constexpr bool inf = (size_ == 0);

    static constexpr auto zero = decltype(size_)(0);
    static constexpr auto ZERO = decltype(size_ * multiplier_)(0);

    static constexpr Kind kind = kind_;
    static constexpr type size = size_;
    static constexpr auto multiplier = multiplier_;

    type var;

    FT_DEV_CEXPR
    auto operator+() const
    {
        return *this;
    }

    FT_DEV_CEXPR
    auto operator-() const
    {
        return Ranged<kind_, size_, -multiplier_>{var};
    }
};

template <auto kind_ = NONE, auto size_ = std::is_same_v<decltype(kind_), Kind> ? 0 : decltype(kind_)(1),
    auto multiplier_ = decltype(size_)(1)>
using Rn = std::conditional_t<std::is_same_v<decltype(kind_), Kind>, Ranged<kind_, size_, multiplier_>,
    Ranged<NONE, kind_, size_>>;

//----------------------------------------------------------------------------
// Poly: polynomial integer
//----------------------------------------------------------------------------

template <auto bias_, class... Ts_>
class Poly
{
public:
    typedef std::tuple<Ts_...> Terms;

    FT_DEV_CEXPR Poly(Cn<bias_>, Terms ts_)
        : terms(ts_)
    {
    }

    FT_DEV_CEXPR Poly(Cn<bias_>, Ts_... ts_)
        : terms(ts_...)
    {
    }

    FT_DEV_CEXPR Poly(Terms ts_)
        : terms(ts_)
    {
    }

    FT_DEV_CEXPR Poly(Ts_... ts_)
        : terms(ts_...)
    {
    }

    static constexpr auto min = (bias_ + ... + Ts_::min);
    static constexpr auto max = (bias_ + ... + Ts_::max);

    static constexpr bool minInf = (false || ... || Ts_::minInf);
    static constexpr bool maxInf = (false || ... || Ts_::maxInf);

    static constexpr auto zero = decltype(bias_)(0);
    static constexpr auto ZERO = decltype((bias_ + ... + Ts_::zero))(0);

    static constexpr auto bias = bias_;

    Terms terms;

    FT_DEV_CEXPR
    auto operator+() const
    {
        return *this;
    }

    FT_DEV_CEXPR
    auto operator-() const
    {
        return negateImp(std::index_sequence_for<Ts_...>());
    }

    template <auto b_>
    FT_DEV_CEXPR auto mul(Cn<b_>) const
    {
        return mulImp(cn<b_>, cn<sizeof...(Ts_)>);
    }

    template <auto b_>
    FT_DEV_CEXPR auto operator/(Cn<b_>) const
    {
        return divImp(cn<b_>, cn<sizeof...(Ts_)>);
    }

    template <auto b_>
    FT_DEV_CEXPR auto mod(Cn<b_>) const
    {
        return modImp(cn<b_>, cn<sizeof...(Ts_)>);
    }

    template <Kind kind_>
    FT_DEV_CEXPR auto filter(Cn<kind_>) const
    {
        return filterImp(cn<kind_>, cn<sizeof...(Ts_)>);
    }

    template <auto b_>
    FT_DEV_CEXPR auto filterDiv(Cn<b_>) const
    {
        if constexpr (b_ == 0)
            return *this; // return itself if indivisible
        else if constexpr (!divisible(cn<b_>))
            return *this; // return itself if indivisible
        else
            return filterDivImp(cn<b_>, cn<sizeof...(Ts_)>);
    }

    template <auto b_>
    FT_DEV_CEXPR static bool divisible(Cn<b_>)
    {
        static_assert(b_);

        constexpr auto dMin = divisibleMin(cn<b_>, std::index_sequence_for<Ts_...>());
        constexpr auto dMax = divisibleMax(cn<b_>, std::index_sequence_for<Ts_...>());
        constexpr auto iMin = indivisibleMin(cn<b_>, std::index_sequence_for<Ts_...>());
        constexpr auto iMax = indivisibleMax(cn<b_>, std::index_sequence_for<Ts_...>());

        return dMin == 0 && iMin == 0 && bias_ >= 0 && iMax + bias_ % b_ < cexpr_abs(b_)
            || dMax == 0 && iMax == 0 && bias_ <= 0 && iMin + bias_ % b_ > -cexpr_abs(b_);
    }

    template <Kind kind_>
    FT_DEV_CEXPR static bool hasOnly(Cn<kind_>)
    {
        return hasOnlyImp(cn<kind_>, cn<sizeof...(Ts_)>);
    }

private:
    template <std::size_t... is_>
    FT_DEV_CEXPR auto negateImp(std::index_sequence<is_...>) const
    {
        return Poly<-bias_, decltype(-std::get<is_>(terms))...>{cn<-bias_>, std::tuple{-std::get<is_>(terms)...}};
    }

    template <auto b_, std::size_t... is_>
    FT_DEV_CEXPR static auto divisibleMin(Cn<b_>, std::index_sequence<is_...>)
    {
        return (zero + ...
            + (std::tuple_element_t<is_, Terms>::multiplier % b_ && std::tuple_element_t<is_, Terms>::size != 1
                    ? std::tuple_element_t<is_, Terms>::ZERO
                    : std::tuple_element_t<is_, Terms>::min
                        + std::tuple_element_t<is_, Terms>::minInf * -cexpr_abs(b_)));
    }

    template <auto b_, std::size_t... is_>
    FT_DEV_CEXPR static auto divisibleMax(Cn<b_>, std::index_sequence<is_...>)
    {
        return (zero + ...
            + (std::tuple_element_t<is_, Terms>::multiplier % b_ && std::tuple_element_t<is_, Terms>::size != 1
                    ? std::tuple_element_t<is_, Terms>::ZERO
                    : std::tuple_element_t<is_, Terms>::max
                        + std::tuple_element_t<is_, Terms>::maxInf * cexpr_abs(b_)));
    }

    template <auto b_, std::size_t... is_>
    FT_DEV_CEXPR static auto indivisibleMin(Cn<b_>, std::index_sequence<is_...>)
    {
        return (zero + ...
            + (std::tuple_element_t<is_, Terms>::multiplier % b_ && std::tuple_element_t<is_, Terms>::size != 1
                    ? std::tuple_element_t<is_, Terms>::min + std::tuple_element_t<is_, Terms>::minInf * -cexpr_abs(b_)
                            > -cexpr_abs(b_)
                        ? std::tuple_element_t<is_, Terms>::min
                            + std::tuple_element_t<is_, Terms>::minInf * -cexpr_abs(b_)
                        : -cexpr_abs(b_) + std::gcd(cexpr_abs(b_), std::tuple_element_t<is_, Terms>::multiplier)
                    : std::tuple_element_t<is_, Terms>::ZERO));
    }

    template <auto b_, std::size_t... is_>
    FT_DEV_CEXPR static auto indivisibleMax(Cn<b_>, std::index_sequence<is_...>)
    {
        return (zero + ...
            + (std::tuple_element_t<is_, Terms>::multiplier % b_ && std::tuple_element_t<is_, Terms>::size != 1
                    ? std::tuple_element_t<is_, Terms>::max + std::tuple_element_t<is_, Terms>::maxInf * cexpr_abs(b_)
                            < cexpr_abs(b_)
                        ? std::tuple_element_t<is_, Terms>::max
                            + std::tuple_element_t<is_, Terms>::maxInf * cexpr_abs(b_)
                        : cexpr_abs(b_) - std::gcd(cexpr_abs(b_), std::tuple_element_t<is_, Terms>::multiplier)
                    : std::tuple_element_t<is_, Terms>::ZERO));
    }

    template <Kind kind_, auto i_>
    FT_DEV_CEXPR static bool hasOnlyImp(Cn<kind_>, Cn<i_>)
    {
        if constexpr (i_ == 0)
            return true;
        else if constexpr (std::tuple_element_t<i_ - 1, Terms>::kind != kind_
            && std::tuple_element_t<i_ - 1, Terms>::size != 1)
            return false;
        else
            return hasOnlyImp(cn<kind_>, cn<i_ - 1>);
    }

    template <auto b_, auto i_>
    FT_DEV_CEXPR auto mulImp(Cn<b_>, Cn<i_>) const
    {
        if constexpr (i_ == 0)
            return cn<bias_ * b_>;
        else
            return mulImp(cn<b_>, cn<i_ - 1>) + std::get<i_ - 1>(terms) * cn<b_>;
    }

    template <auto b_, auto i_>
    FT_DEV_CEXPR auto divImp(Cn<b_>, Cn<i_>) const
    {
        static_assert(b_);
        static_assert(divisible(cn<b_>));

        if constexpr (i_ == 0)
            return cn<bias_ / b_>;
        else if constexpr (std::tuple_element_t<i_ - 1, Terms>::abs >= cexpr_abs(b_)
            || std::tuple_element_t<i_ - 1, Terms>::inf)
            return divImp(cn<b_>, cn<i_ - 1>) + std::get<i_ - 1>(terms) / cn<b_>;
        else
            return divImp(cn<b_>, cn<i_ - 1>);
    }

    template <auto b_, auto i_>
    FT_DEV_CEXPR auto modImp(Cn<b_>, Cn<i_>) const
    {
        static_assert(b_);
        static_assert(divisible(cn<b_>));

        if constexpr (i_ == 0)
            return cn<bias_ % b_>;
        else if constexpr (std::tuple_element_t<i_ - 1, Terms>::multiplier % b_
            && std::tuple_element_t<i_ - 1, Terms>::size != 1)
            return modImp(cn<b_>, cn<i_ - 1>) + std::get<i_ - 1>(terms) % cn<b_>;
        else
            return modImp(cn<b_>, cn<i_ - 1>);
    }

    template <Kind kind_, auto i_>
    FT_DEV_CEXPR auto filterImp(Cn<kind_>, Cn<i_>) const
    {
        if constexpr (i_ == 0)
            return Poly<zero>{};
        else if constexpr (std::tuple_element_t<i_ - 1, Terms>::kind == kind_
            && std::tuple_element_t<i_ - 1, Terms>::size != 1)
            return filterImp(cn<kind_>, cn<i_ - 1>) + std::get<i_ - 1>(terms);
        else
            return filterImp(cn<kind_>, cn<i_ - 1>);
    }

    template <auto b_, auto i_>
    FT_DEV_CEXPR auto filterDivImp(Cn<b_>, Cn<i_>) const
    {
        static_assert(b_);
        static_assert(divisible(cn<b_>));

        if constexpr (i_ == 0)
            return Poly<decltype(bias_)(bias_ / b_ * b_)>{};
        else if constexpr (std::tuple_element_t<i_ - 1, Terms>::abs >= cexpr_abs(b_)
            || std::tuple_element_t<i_ - 1, Terms>::inf)
            return filterDivImp(cn<b_>, cn<i_ - 1>) + std::get<i_ - 1>(terms);
        else
            return filterDivImp(cn<b_>, cn<i_ - 1>);
    }
};

// constructs Poly from Cn and Rns
template <auto value_, Kind... kinds_, auto... sizes_, auto... multipliers_>
Poly(Cn<value_>, Ranged<kinds_, sizes_, multipliers_>...) -> Poly<value_, Rn<kinds_, sizes_, multipliers_>...>;
// constructs Poly from Rns
template <Kind... kinds_, auto... sizes_, auto... multipliers_>
Poly(Ranged<kinds_, sizes_, multipliers_>...) -> Poly<false, Rn<kinds_, sizes_, multipliers_>...>;

//----------------------------------------------------------------------------
// Operators for Rn and Poly
//----------------------------------------------------------------------------

/* We should never use int * Rn
template <Kind k_,auto z_,auto m_,class T_> FT_DEV_CEXPR std::enable_if_t<std::is_integral_v<T_>,
Rn<k_,decltype(z_*T_{})(0),m_>> operator * (T_ a_, Ranged<k_,z_,m_> x_) { return Rn<k_,decltype(z_*T_{})(0),m_>{x_.var *
a_}; } template <Kind k_,auto z_,auto m_,class T_> FT_DEV_CEXPR std::enable_if_t<std::is_integral_v<T_>,
Rn<k_,decltype(z_*T_{})(0),m_>> operator * (Ranged<k_,z_,m_> x_, T_ b_) { return Rn<k_,decltype(z_*T_{})(0),m_>{x_.var *
b_}; }
*/

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR std::enable_if_t<a_ != 0, Rn<k_, z_, m_ * a_>> operator*(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return Rn<k_, z_, m_ * a_>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ != 0, Rn<k_, z_, m_ * b_>> operator*(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, z_, m_ * b_>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR std::enable_if_t<m_ % b_ == 0, Rn<k_, z_, m_ / b_>> operator/(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, z_, m_ / b_>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR std::enable_if_t<m_ % b_ == 0, Cn<m_ % b_>> operator%(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return cn<m_ % b_>;
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR Rn<k_, z_, (m_ << b_)> operator<<(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, z_, (m_ << b_)>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR std::enable_if_t<m_ % (1 << b_) == 0, Rn<k_, z_, (m_ >> b_)>> operator>>(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, z_, (m_ >> b_)>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR
    std::enable_if_t<(Rn<k_, z_, m_>::abs < cexpr_abs(b_) && !Rn<k_, z_, m_>::inf && m_ % b_ != 0), Cn<m_ / b_>>
    operator/(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return cn<m_ / b_>;
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR
    std::enable_if_t<(Rn<k_, z_, m_>::abs < cexpr_abs(b_) && !Rn<k_, z_, m_>::inf && m_ % b_ != 0), Rn<k_, z_, m_>>
    operator%(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, z_, m_>{x_.var};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR
    std::enable_if_t<(Rn<k_, z_, m_>::abs < (1 << b_) && !Rn<k_, z_, m_>::inf && m_ % (1 << b_) != 0), Cn<(m_ >> b_)>>
    operator>>(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return cn<(m_ >> b_)>;
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR
    std::enable_if_t<(Rn<k_, z_, m_>::abs >= cexpr_abs(b_) || Rn<k_, z_, m_>::inf) && (m_ % b_ != 0 && b_ % m_ == 0),
        Rn<k_, div_up(z_, cexpr_abs(b_ / m_)), b_ / m_ / cexpr_abs(b_ / m_)>>
    operator/(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, div_up(z_, cexpr_abs(b_ / m_)), b_ / m_ / cexpr_abs(b_ / m_)>{x_.var / cexpr_abs(b_ / m_)};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR
    std::enable_if_t<(Rn<k_, z_, m_>::abs >= cexpr_abs(b_) || Rn<k_, z_, m_>::inf) && (m_ % b_ != 0 && b_ % m_ == 0),
        Rn<k_, cexpr_abs(b_ / m_), m_>>
    operator%(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, cexpr_abs(b_ / m_), m_>{x_.var % cexpr_abs(b_ / m_)};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR std::enable_if_t<(m_ > 0) && // correct only when positive
        (Rn<k_, z_, m_>::abs >= (1 << b_) || Rn<k_, z_, m_>::inf) && (m_ % (1 << b_) != 0 && (1 << b_) % m_ == 0),
    Rn<k_, div_up(z_, cexpr_abs((1 << b_) / m_)), (1 << b_) / m_ / cexpr_abs((1 << b_) / m_)>>
operator>>(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return Rn<k_, div_up(z_, cexpr_abs((1 << b_) / m_)), (1 << b_) / m_ / cexpr_abs((1 << b_) / m_)>{
        x_.var / cexpr_abs((1 << b_) / m_)};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator+(Ranged<k_, z_, m_> a_, Cn<b_>)
{
    return Poly{cn<b_>, std::tuple{a_}};
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator-(Ranged<k_, z_, m_> a_, Cn<b_>)
{
    return Poly{cn<-b_>, std::tuple{a_}};
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator+(Cn<a_>, Ranged<k_, z_, m_> b_)
{
    return Poly{cn<a_>, std::tuple{b_}};
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator-(Cn<a_>, Ranged<k_, z_, m_> b_)
{
    return Poly{cn<a_>, std::tuple{-b_}};
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator+(Poly<A_, Ts_...> a_, Cn<b_>)
{
    return Poly{cn<A_ + b_>, a_.terms};
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator-(Poly<A_, Ts_...> a_, Cn<b_>)
{
    return Poly{cn<A_ - b_>, a_.terms};
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator+(Cn<a_>, Poly<B_, Ts_...> b_)
{
    return Poly{cn<a_ + B_>, b_.terms};
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator-(Cn<a_>, Poly<B_, Ts_...> b_)
{
    return Poly{cn<a_ - B_>, (-b_).terms};
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator+(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return Poly{cn<false>, std::tuple{a_, b_}};
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator-(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return Poly{cn<false>, std::tuple{a_, -b_}};
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator+(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return Poly{cn<A_>, std::tuple_cat(a_.terms, std::tuple{b_})};
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator-(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return Poly{cn<A_>, std::tuple_cat(a_.terms, std::tuple{-b_})};
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator+(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return Poly{cn<B_>, std::tuple_cat(std::tuple{a_}, b_.terms)};
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator-(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return Poly{cn<-B_>, std::tuple_cat(std::tuple{a_}, (-b_).terms)};
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator+(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return Poly{cn<A_ + B_>, std::tuple_cat(a_.terms, b_.terms)};
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator-(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return Poly{cn<A_ - B_>, std::tuple_cat(a_.terms, (-b_).terms)};
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR std::enable_if_t<a_ != 0, decltype(std::declval<Poly<B_, Ts_...>>().mul(cn<a_>))> operator*(
    Cn<a_>, Poly<B_, Ts_...> x_)
{
    return x_.mul(cn<a_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ != 0, decltype(std::declval<Poly<A_, Ts_...>>().mul(cn<b_>))> operator*(
    Poly<A_, Ts_...> x_, Cn<b_>)
{
    return x_.mul(cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR std::enable_if_t<b_ != +1 && b_ != -1, decltype(std::declval<Poly<A_, Ts_...>>().mod(cn<b_>))> operator%(
    Poly<A_, Ts_...> x_, Cn<b_>)
{
    return x_.mod(cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator<<(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return x_ * cn<(1 << b_)>;
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator>>(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return x_ / cn<(1 << b_)>;
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator*(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return Ranged < kA == kB ? kA : NONE, zA * zB, mA * mB > {a_.var * b_.var};
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator*(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return Ranged < Poly<A_, Ts_...>::hasOnly(cn<k_>) ? k_ : NONE > {get(a_) * b_.var} * cn<m_>;
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator*(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return Ranged < Poly<B_, Ts_...>::hasOnly(cn<k_>) ? k_ : NONE > {a_.var * get(b_)} * cn<m_>;
}

/* We should never use Poly * Poly
template <auto A_,class...TsA, auto B_,class...TsB>
FT_DEV_CEXPR
auto operator * (Poly<A_,TsA...> a_, Poly<B_,TsB...> b_)
{
  return Ranged<Poly<A_,TsA...>::hasOnly(cn<UNROLL>) &&
                Poly<B_,TsB...>::hasOnly(cn<UNROLL>) ? UNROLL : (
                Poly<A_,TsA...>::hasOnly(cn<ID>) &&
                Poly<B_,TsB...>::hasOnly(cn<ID>) ? ID : NONE)>{get(a_) * get(b_)};
}
*/

//----------------------------------------------------------------------------
// get() for Cn, Rn and Poly
//----------------------------------------------------------------------------

template <auto value_>
FT_DEV_CEXPR auto get(Cn<value_>)
{
    return value_;
}

template <Kind kind_, auto size_, auto multiplier_>
FT_DEV_CEXPR auto get(Ranged<kind_, size_, multiplier_> x_)
{
    if constexpr (size_ == 1)
        return Rn<kind_, size_, multiplier_>::ZERO;
    else
        return x_.var * multiplier_;
}

template <auto bias_, class... Ts_, std::size_t... is_>
FT_DEV_CEXPR auto getImp(Poly<bias_, Ts_...> x_, std::index_sequence<is_...>)
{
    return (bias_ + ... + get(std::get<is_>(x_.terms)));
}

template <Kind kind_, auto bias_, class... Ts_>
FT_DEV_CEXPR auto get(Poly<bias_, Ts_...> x_)
{
    return getImp(x_.filter(cn<kind_>),
        std::make_index_sequence<std::tuple_size_v<typename decltype(x_.filter(cn<kind_>))::Terms>>());
}

template <auto bias_, class... Ts_>
FT_DEV_CEXPR auto get(Poly<bias_, Ts_...> x_)
{
    return bias_ + get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_);
}

//----------------------------------------------------------------------------
// Comparison operators for Rn and Poly
//----------------------------------------------------------------------------

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto ltzeroImp(Poly<A_, Ts_...> x_)
{
    constexpr decltype(A_) p2 = A_ ? ((-A_) ^ (-A_ - 1)) / 2 + 1 : 0;

    constexpr auto n1 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<1>))::Terms>;
    constexpr auto n2 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<p2>))::Terms>;
    constexpr auto nA = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<A_>))::Terms>;

    if constexpr (Poly<A_, Ts_...>::min >= 0 && !Poly<A_, Ts_...>::minInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::max < 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<true>;

    else if constexpr (A_ < 0 && nA < n2 && nA < n1)
        return ltzeroImp((x_ - cn<A_>) .filterDiv(cn<A_>) + cn<A_>);
    else if constexpr (A_ < 0 && n2 < n1)
        return ltzeroImp((x_ - cn<A_>) .filterDiv(cn<p2>) + cn<A_>);

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min >= 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max < 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return true;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) < -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) > A_;
}

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto lezeroImp(Poly<A_, Ts_...> x_)
{
    constexpr decltype(A_) p2 = A_ ? ((+A_) ^ (+A_ - 1)) / 2 + 1 : 0;

    constexpr auto n1 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<1>))::Terms>;
    constexpr auto n2 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<p2>))::Terms>;
    constexpr auto nA = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<A_>))::Terms>;

    if constexpr (Poly<A_, Ts_...>::min > 0 && !Poly<A_, Ts_...>::minInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::max <= 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<true>;

    else if constexpr (A_ > 0 && nA < n2 && nA < n1)
        return lezeroImp((x_ - cn<A_>) .filterDiv(cn<A_>) + cn<A_>);
    else if constexpr (A_ > 0 && n2 < n1)
        return lezeroImp((x_ - cn<A_>) .filterDiv(cn<p2>) + cn<A_>);

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min > 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max <= 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return true;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) <= -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) >= A_;
}

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto gtzeroImp(Poly<A_, Ts_...> x_)
{
    constexpr decltype(A_) p2 = A_ ? ((+A_) ^ (+A_ - 1)) / 2 + 1 : 0;

    constexpr auto n1 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<1>))::Terms>;
    constexpr auto n2 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<p2>))::Terms>;
    constexpr auto nA = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<A_>))::Terms>;

    if constexpr (Poly<A_, Ts_...>::max <= 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::min > 0 && !Poly<A_, Ts_...>::minInf)
        return cn<true>;

    else if constexpr (A_ > 0 && nA < n2 && nA < n1)
        return gtzeroImp((x_ - cn<A_>) .filterDiv(cn<A_>) + cn<A_>);
    else if constexpr (A_ > 0 && n2 < n1)
        return gtzeroImp((x_ - cn<A_>) .filterDiv(cn<p2>) + cn<A_>);

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max <= 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min > 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return true;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) > -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) < A_;
}

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto gezeroImp(Poly<A_, Ts_...> x_)
{
    constexpr decltype(A_) p2 = A_ ? ((-A_) ^ (-A_ - 1)) / 2 + 1 : 0;

    constexpr auto n1 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<1>))::Terms>;
    constexpr auto n2 = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<p2>))::Terms>;
    constexpr auto nA = std::tuple_size_v<typename decltype((x_ - cn<A_>) .filterDiv(cn<A_>))::Terms>;

    if constexpr (Poly<A_, Ts_...>::max < 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::min >= 0 && !Poly<A_, Ts_...>::minInf)
        return cn<true>;

    else if constexpr (A_ < 0 && nA < n2 && nA < n1)
        return gezeroImp((x_ - cn<A_>) .filterDiv(cn<A_>) + cn<A_>);
    else if constexpr (A_ < 0 && n2 < n1)
        return gezeroImp((x_ - cn<A_>) .filterDiv(cn<p2>) + cn<A_>);

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max < 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min >= 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return true;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) >= -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) <= A_;
}

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto eqzeroImp(Poly<A_, Ts_...> x_)
{
    if constexpr (Poly<A_, Ts_...>::min > 0 && !Poly<A_, Ts_...>::minInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::max < 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<false>;
    else if constexpr (Poly<A_, Ts_...>::min == 0 && !Poly<A_, Ts_...>::minInf && Poly<A_, Ts_...>::max == 0
        && !Poly<A_, Ts_...>::maxInf)
        return cn<true>;

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min > 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max < 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return false;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min == 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf
        && A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max == 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return true;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) == -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) == A_;
}

template <auto A_, class... Ts_>
FT_DEV_CEXPR auto nezeroImp(Poly<A_, Ts_...> x_)
{
    if constexpr (Poly<A_, Ts_...>::min > 0 && !Poly<A_, Ts_...>::minInf)
        return cn<true>;
    else if constexpr (Poly<A_, Ts_...>::max < 0 && !Poly<A_, Ts_...>::maxInf)
        return cn<true>;
    else if constexpr (Poly<A_, Ts_...>::min == 0 && !Poly<A_, Ts_...>::minInf && Poly<A_, Ts_...>::max == 0
        && !Poly<A_, Ts_...>::maxInf)
        return cn<false>;

    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min > 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf)
        return true;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max < 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return true;
    else if (A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::min == 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::minInf
        && A_ + get<UNROLL>(x_) + decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::max == 0
        && !decltype(x_.filter(cn<ID>) + x_.filter(cn<NONE>))::maxInf)
        return false;

    else if constexpr (A_ < 0)
        return get<ID>(x_) + get<UNROLL>(x_) + get<NONE>(x_) != -A_;
    else
        return get<ID>(-x_) + get<UNROLL>(-x_) + get<NONE>(-x_) != A_;
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator<(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return ltzeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator<=(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return lezeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator>(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return gtzeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator>=(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return gezeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator==(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return eqzeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto a_>
FT_DEV_CEXPR auto operator!=(Cn<a_>, Ranged<k_, z_, m_> x_)
{
    return nezeroImp(cn<a_> - x_);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator<(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return ltzeroImp(x_ - cn<b_>);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator<=(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return lezeroImp(x_ - cn<b_>);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator>(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return gtzeroImp(x_ - cn<b_>);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator>=(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return gezeroImp(x_ - cn<b_>);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator==(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return eqzeroImp(x_ - cn<b_>);
}

template <Kind k_, auto z_, auto m_, auto b_>
FT_DEV_CEXPR auto operator!=(Ranged<k_, z_, m_> x_, Cn<b_>)
{
    return nezeroImp(x_ - cn<b_>);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator<(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return ltzeroImp(cn<a_> - x_);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator<=(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return lezeroImp(cn<a_> - x_);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator>(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return gtzeroImp(cn<a_> - x_);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator>=(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return gezeroImp(cn<a_> - x_);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator==(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return eqzeroImp(cn<a_> - x_);
}

template <auto B_, class... Ts_, auto a_>
FT_DEV_CEXPR auto operator!=(Cn<a_>, Poly<B_, Ts_...> x_)
{
    return nezeroImp(cn<a_> - x_);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator<(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return ltzeroImp(x_ - cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator<=(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return lezeroImp(x_ - cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator>(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return gtzeroImp(x_ - cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator>=(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return gezeroImp(x_ - cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator==(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return eqzeroImp(x_ - cn<b_>);
}

template <auto A_, class... Ts_, auto b_>
FT_DEV_CEXPR auto operator!=(Poly<A_, Ts_...> x_, Cn<b_>)
{
    return nezeroImp(x_ - cn<b_>);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator<(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return ltzeroImp(a_ - b_);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator<=(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return lezeroImp(a_ - b_);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator>(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return gtzeroImp(a_ - b_);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator>=(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return gezeroImp(a_ - b_);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator==(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return eqzeroImp(a_ - b_);
}

template <Kind kA, auto zA, auto mA, Kind kB, auto zB, auto mB>
FT_DEV_CEXPR auto operator!=(Ranged<kA, zA, mA> a_, Ranged<kB, zB, mB> b_)
{
    return nezeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator<(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return ltzeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator<=(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return lezeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator>(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return gtzeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator>=(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return gezeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator==(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return eqzeroImp(a_ - b_);
}

template <auto A_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator!=(Poly<A_, Ts_...> a_, Ranged<k_, z_, m_> b_)
{
    return nezeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator<(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return ltzeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator<=(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return lezeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator>(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return gtzeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator>=(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return gezeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator==(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return eqzeroImp(a_ - b_);
}

template <auto B_, class... Ts_, Kind k_, auto z_, auto m_>
FT_DEV_CEXPR auto operator!=(Ranged<k_, z_, m_> a_, Poly<B_, Ts_...> b_)
{
    return nezeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator<(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return ltzeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator<=(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return lezeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator>(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return gtzeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator>=(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return gezeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator==(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return eqzeroImp(a_ - b_);
}

template <auto A_, class... TsA, auto B_, class... TsB>
FT_DEV_CEXPR auto operator!=(Poly<A_, TsA...> a_, Poly<B_, TsB...> b_)
{
    return nezeroImp(a_ - b_);
}

//----------------------------------------------------------------------------
// swizzle() for Poly
//----------------------------------------------------------------------------

template <int mode_, /*  mode_,     16, */ class A_>
FT_DEV_CEXPR auto swizzle(A_ a_)
{
    return swizzle<mode_, mode_, 16>(Poly{a_});
}

template <int mode_, /*  mode_,     16, */ class A_, class B_>
FT_DEV_CEXPR auto swizzle(A_ a_, B_ b_)
{
    return swizzle<mode_, mode_, 16>(Poly{a_}, Poly{b_});
}

template <int mode_, int line_, /*  16, */ class A_>
FT_DEV_CEXPR auto swizzle(A_ a_)
{
    return swizzle<mode_, line_, 16>(Poly{a_});
}

template <int mode_, int line_, /*  16, */ class A_, class B_>
FT_DEV_CEXPR auto swizzle(A_ a_, B_ b_)
{
    return swizzle<mode_, line_, 16>(Poly{a_}, Poly{b_});
}

template <int mode_, int line_, int unit_, class A_>
FT_DEV_CEXPR auto swizzle(A_ a_)
{
    return swizzle<mode_, line_, unit_>(Poly{a_});
}

template <int mode_, int line_, int unit_, class A_, class B_>
FT_DEV_CEXPR auto swizzle(A_ a_, B_ b_)
{
    return swizzle<mode_, line_, unit_>(Poly{a_}, Poly{b_});
}

template <int mode_, int line_, int unit_, auto biasA_, class... TsA_>
FT_DEV_CEXPR auto swizzle(Poly<biasA_, TsA_...> a_)
{
    static_assert((mode_ & (mode_ - 1)) == 0);
    static_assert((unit_ & (unit_ - 1)) == 0);
    static_assert(mode_ >= unit_);

    if constexpr (decltype(a_)::divisible(cn<line_>))
    {
        if constexpr (decltype(Poly{a_ / cn<line_>})::divisible(cn<mode_ / unit_>)
            && decltype(Poly{a_ % cn<line_>})::divisible(cn<unit_>))
        {
            if constexpr (mode_ == unit_)
                return get(a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<UNROLL>))
                return biasA_ + get<ID>(a_) + (get<UNROLL>(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>))
                    + get<NONE>(a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<ID>))
                return biasA_ + (get<ID>(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>)) + get<UNROLL>(a_)
                    + get<NONE>(a_);
            else
                return get(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>);
        }
#if 1
        else if constexpr (decltype(Poly{a_ % cn<line_>})::divisible(cn<unit_>))
        {
            if constexpr (mode_ == unit_)
                return get(a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<UNROLL>))
                return biasA_ + get<ID>(a_)
                    + (get<UNROLL>(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>) +get<NONE>(a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<ID>))
                return biasA_ + (get<ID>(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>) +get<UNROLL>(a_)
                    + get<NONE>(a_);
#endif
            else
                return get(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>;
#if 1
        }
        else
        {
            return get(a_) ^ get(a_) / cn<line_> % cn<mode_ / unit_> * cn<unit_>;
        }
#endif
    }
    else
    {
        return get(a_) ^ get(a_) / cn<line_> % cn<mode_ / unit_> * cn<unit_>;
    }
}

template <int mode_, int line_, int unit_, auto biasA_, class... TsA_, auto biasB_, class... TsB_>
FT_DEV_CEXPR auto swizzle(Poly<biasA_, TsA_...> a_, Poly<biasB_, TsB_...> b_)
{
    static_assert((mode_ & (mode_ - 1)) == 0);
    static_assert((unit_ & (unit_ - 1)) == 0);
    static_assert(mode_ >= unit_);

    if constexpr (decltype(a_)::divisible(cn<line_>))
    {
        if constexpr (decltype(Poly{a_ / cn<line_>})::divisible(cn<mode_ / unit_>)
            && decltype(Poly{a_ % cn<line_>})::divisible(cn<unit_>))
        {
            if constexpr (mode_ == unit_)
                return get(b_ + a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<UNROLL>))
                return biasB_ + biasA_ + get<ID>(b_ + a_)
                    + (get<UNROLL>(b_) + (get<UNROLL>(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>)))
                    + get<NONE>(b_ + a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<ID>))
                return biasB_ + biasA_
                    + (get<ID>(b_) + (get<ID>(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>)))
                    + get<UNROLL>(b_ + a_) + get<NONE>(b_ + a_);
            else
                return get(b_) + (get(a_) ^ get(a_ / cn<line_> % cn<mode_ / unit_> * cn<unit_>));
        }
#if 1
        else if constexpr (decltype(Poly{a_ % cn<line_>})::divisible(cn<unit_>))
        {
            if constexpr (mode_ == unit_)
                return get(b_ + a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<UNROLL>))
                return biasB_ + biasA_ + get<ID>(b_ + a_)
                    + (get<UNROLL>(b_) + (get<UNROLL>(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>) )
                    + get<NONE>(b_ + a_);
            else if constexpr (decltype(Poly{a_ % cn<line_> / cn<unit_>})::hasOnly(cn<ID>))
                return biasB_ + biasA_
                    + (get<ID>(b_) + (get<ID>(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>) )
                    + get<UNROLL>(b_ + a_) + get<NONE>(b_ + a_);
#endif
            else
                return get(b_) + (get(a_) ^ get(a_ / cn<line_>) % cn<mode_ / unit_> * cn<unit_>);
#if 1
        }
        else
        {
            if constexpr (mode_ == unit_)
                return get(b_ + a_);
            else
                return get(b_) + (get(a_) ^ get(a_) / cn<line_> % cn<mode_ / unit_> * cn<unit_>);
        }
#endif
    }
    else
    {
        if constexpr (mode_ == unit_)
            return get(b_ + a_);
        else
            return get(b_) + (get(a_) ^ get(a_) / cn<line_> % cn<mode_ / unit_> * cn<unit_>);
    }
}

// vim: ts=2 sw=2 sts=2 et sta
