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
            "__iter__", [](T& s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>());
}

} // namespace PybindUtils
