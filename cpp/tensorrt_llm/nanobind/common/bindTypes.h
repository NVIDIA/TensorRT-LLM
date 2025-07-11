/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/pybind/common/customCasters.h"
#include <pybind11/pybind11.h>

namespace PybindUtils
{

namespace py = pybind11;

template <typename T>
void bindList(py::module& m, std::string const& name)
{
    py::class_<T>(m, name.c_str())
        .def(py::init())
        .def("push_back", [](T& lst, const typename T::value_type& value) { lst.push_back(value); })
        .def("pop_back", [](T& lst) { lst.pop_back(); })
        .def("push_front", [](T& lst, const typename T::value_type& value) { lst.push_front(value); })
        .def("pop_front", [](T& lst) { lst.pop_front(); })
        .def("__len__", [](T const& lst) { return lst.size(); })
        .def(
            "__iter__", [](T& lst) { return py::make_iterator(lst.begin(), lst.end()); }, py::keep_alive<0, 1>())
        .def("__getitem__",
            [](T const& lst, size_t index)
            {
                if (index >= lst.size())
                    throw py::index_error();
                auto it = lst.begin();
                std::advance(it, index);
                return *it;
            })
        .def("__setitem__",
            [](T& lst, size_t index, const typename T::value_type& value)
            {
                if (index >= lst.size())
                    throw py::index_error();
                auto it = lst.begin();
                std::advance(it, index);
                *it = value;
            });
}

template <typename T>
void bindSet(py::module& m, std::string const& name)
{
    py::class_<T>(m, name.c_str())
        .def(py::init())
        .def("clear", &T::clear)
        .def("size", &T::size)
        .def("insert", [](T& s, typename T::value_type const& value) { s.insert(value); })
        .def("erase", py::overload_cast<typename T::value_type const&>(&T::erase))
        .def("__len__", [](T const& lst) { return lst.size(); })
        .def("__contains__", [](T const& s, typename T::value_type x) { return s.find(x) != s.end(); })
        .def(
            "__iter__", [](T& s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
        .def("__eq__", [](T const& s, T const& other) { return s == other; })
        .def(py::pickle(
            [](T const& s) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(std::vector<typename T::value_type>(s.begin(), s.end()));
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                /* Create a new C++ instance */
                T s;
                /* Assign any additional state */
                auto state_list = t[0].cast<std::vector<typename T::value_type>>();
                for (auto& item : state_list)
                {
                    s.insert(item);
                }
                return s;
            }));
}

} // namespace PybindUtils
