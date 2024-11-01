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

#include "opaqueBindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace PybindUtils
{
template <typename T>
void makeAlgorithmBindings(py::module_& m)
{
    py::class_<T>(m, T::name).def(py::init()).def("forward", &T::forward).def("name", [](T const&) { return T::name; });
}

template <typename T>
void instantiatePybindAlgorithm(py::module_& m);
} // namespace PybindUtils

#define INSTANTIATE_ALGORITHM(TYPE)                                                                                    \
    template <>                                                                                                        \
    void PybindUtils::instantiatePybindAlgorithm<TYPE>(py::module_ & m)                                                \
    {                                                                                                                  \
        makeAlgorithmBindings<TYPE>(m);                                                                                \
    };
