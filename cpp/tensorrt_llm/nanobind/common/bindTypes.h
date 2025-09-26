/*
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace NanobindUtils
{

namespace nb = nanobind;

template <typename T>
void bindSet(nb::module_& m, std::string const& name)
{
    nb::class_<T>(m, name.c_str())
        .def(nb::init<>())
        .def("clear", &T::clear)
        .def("size", &T::size)
        .def("insert", [](T& s, typename T::value_type const& value) { s.insert(value); })
        .def("erase", nb::overload_cast<typename T::value_type const&>(&T::erase))
        .def("__len__", [](T const& lst) { return lst.size(); })
        .def("__contains__", [](T const& s, typename T::value_type x) { return s.find(x) != s.end(); })
        .def(
            "__iter__", [](T& s) { return nb::make_iterator(nb::type<T>(), "iterator", s.begin(), s.end()); },
            nb::keep_alive<0, 1>())
        .def("__eq__", [](T const& s, T const& other) { return s == other; })
        .def("__getstate__",
            [](T const& v)
            {
                /* Return a tuple that fully encodes the state of the object */
                return nb::make_tuple(std::vector<typename T::value_type>(v.begin(), v.end()));
            })
        .def("__setstate__",
            [](T& v, nb::tuple const& t)
            {
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                /* Create a new C++ instance */
                T s;
                /* Assign any additional state */
                auto state_list = nb::cast<std::vector<typename T::value_type>>(t[0]);
                for (auto& item : state_list)
                {
                    s.insert(item);
                }
                new (&v) T(s);
            });
}

} // namespace NanobindUtils
