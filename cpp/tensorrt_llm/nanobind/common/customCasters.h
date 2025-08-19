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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/DLConvertor.h>
#include <c10/util/ArrayRef.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <deque>

// Pybind requires to have a central include in order for type casters to work.
// Opaque bindings add a type caster, so they have the same requirement.
// See the warning in https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

// Opaque bindings
NB_MAKE_OPAQUE(tensorrt_llm::batch_manager::ReqIdsSet)
NB_MAKE_OPAQUE(std::vector<tensorrt_llm::batch_manager::SlotDecoderBuffers>)
NB_MAKE_OPAQUE(std::vector<tensorrt_llm::runtime::SamplingConfig>)

namespace nb = nanobind;

// Custom casters
namespace NB_NAMESPACE
{

namespace detail
{

template <typename T, typename Alloc>
struct type_caster<std::deque<T, Alloc>>
{
    using Type = std::deque<T, Alloc>;
    NB_TYPE_CASTER(Type, const_name("List"));

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        sequence seq(src, nanobind::detail::borrow_t{});
        value.clear();
        make_caster<T> caster;
        for (auto const& item : seq)
        {
            if (!caster.from_python(item, flags, cleanup))
                return false;
            value.push_back(caster.operator T&());
        }
        return true;
    }

    static handle from_cpp(Type const& deque, rv_policy policy, cleanup_list* cleanup) noexcept
    {
        nb::list list;

        for (auto const& item : deque)
        {
            nb::object py_item = steal(make_caster<T>::from_cpp(item, policy, cleanup));
            if (!py_item)
                return {};
            list.append(py_item);
        }
        return list.release();
    }
};

template <typename T>
struct type_caster<tensorrt_llm::common::OptionalRef<T>>
{
    using value_conv = make_caster<T>;

    NB_TYPE_CASTER(tensorrt_llm::common::OptionalRef<T>, value_conv::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup)
    {
        if (src.is_none())
        {
            // If the Python object is None, create an empty OptionalRef
            value = tensorrt_llm::common::OptionalRef<T>();
            return true;
        }

        value_conv conv;
        if (!conv.from_python(src, flags, cleanup))
            return false;

        // Create an OptionalRef with a reference to the converted value
        value = tensorrt_llm::common::OptionalRef<T>(conv);
        return true;
    }

    static handle from_cpp(tensorrt_llm::common::OptionalRef<T> const& src, rv_policy policy, cleanup_list* cleanup)
    {
        if (!src.has_value())
            return none().release();

        return value_conv::from_cpp(*src, policy, cleanup);
    }
};

template <>
class type_caster<tensorrt_llm::executor::StreamPtr>
{
public:
    NB_TYPE_CASTER(tensorrt_llm::executor::StreamPtr, const_name("int"));

    bool from_python([[maybe_unused]] handle src, uint8_t flags, cleanup_list* cleanup)
    {
        auto stream_ptr = nanobind::cast<uintptr_t>(src);
        value = std::make_shared<tensorrt_llm::runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream_ptr));

        return true;
    }

    static handle from_cpp(
        tensorrt_llm::executor::StreamPtr const& src, rv_policy /* policy */, cleanup_list* /* cleanup */)
    {
        // Return cudaStream_t as integer.
        return PyLong_FromVoidPtr(src->get());
    }
};

template <>
struct type_caster<tensorrt_llm::executor::Tensor>
{
public:
    NB_TYPE_CASTER(tensorrt_llm::executor::Tensor, const_name("torch.Tensor"));

    // Convert PyObject(torch.Tensor) -> tensorrt_llm::executor::Tensor
    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup)
    {
        PyObject* obj = src.ptr();
        if (THPVariable_Check(obj))
        {
            at::Tensor const& t = THPVariable_Unpack(obj);
            value = tensorrt_llm::executor::detail::ofITensor(tensorrt_llm::runtime::TorchView::of(t));
            return true;
        }
        return false;
    }

    // Convert tensorrt_llm::executor::Tensor -> PyObject(torch.Tensor)
    static handle from_cpp(
        tensorrt_llm::executor::Tensor const& src, rv_policy /* policy */, cleanup_list* /* cleanup */)
    {
        return THPVariable_Wrap(tensorrt_llm::runtime::Torch::tensor(tensorrt_llm::executor::detail::toITensor(src)));
    }
};

template <>
struct type_caster<tensorrt_llm::runtime::ITensor::SharedPtr>
{
public:
    NB_TYPE_CASTER(tensorrt_llm::runtime::ITensor::SharedPtr, const_name("torch.Tensor"));

    // Convert PyObject(torch.Tensor) -> tensorrt_llm::runtime::ITensor::SharedPtr
    bool from_python(handle src, uint8_t, cleanup_list*)
    {
        PyObject* obj = src.ptr();
        if (THPVariable_Check(obj))
        {
            at::Tensor const& t = THPVariable_Unpack(obj);
            value = std::move(tensorrt_llm::runtime::TorchView::of(t));
            return true;
        }
        return false;
    }

    // Convert tensorrt_llm::runtime::ITensor::SharedPtr -> PyObject(torch.Tensor)
    static handle from_cpp(
        tensorrt_llm::runtime::ITensor::SharedPtr const& src, rv_policy /* policy */, cleanup_list* /* cleanup */)
    {
        if (src == nullptr)
        {
            return none().release();
        }
        return THPVariable_Wrap(tensorrt_llm::runtime::Torch::tensor(src));
    }
};

template <>
struct type_caster<tensorrt_llm::runtime::ITensor::SharedConstPtr>
{
public:
    NB_TYPE_CASTER(tensorrt_llm::runtime::ITensor::SharedConstPtr, const_name("torch.Tensor"));

    // Convert PyObject(torch.Tensor) -> tensorrt_llm::runtime::ITensor::SharedConstPtr
    bool from_python(handle src, uint8_t, cleanup_list*)
    {
        PyObject* obj = src.ptr();
        if (THPVariable_Check(obj))
        {
            at::Tensor const& t = THPVariable_Unpack(obj);
            value = std::move(tensorrt_llm::runtime::TorchView::of(t));
            return true;
        }
        return false;
    }

    // Convert tensorrt_llm::runtime::ITensor::SharedConstPtr -> PyObject(torch.Tensor)
    static handle from_cpp(
        tensorrt_llm::runtime::ITensor::SharedConstPtr const& src, rv_policy /* policy */, cleanup_list* /* cleanup */)
    {
        if (src == nullptr)
        {
            return none().release();
        }
        return THPVariable_Wrap(tensorrt_llm::runtime::Torch::tensor(
            reinterpret_cast<tensorrt_llm::runtime::ITensor::SharedPtr const&>(src)));
    }
};

template <>
struct type_caster<at::Tensor>
{
    NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"));

    bool from_python(nb::handle src, uint8_t, cleanup_list*) noexcept
    {
        PyObject* obj = src.ptr();
        if (THPVariable_Check(obj))
        {
            value = THPVariable_Unpack(obj);
            return true;
        }
        return false;
    }

    static handle from_cpp(at::Tensor src, rv_policy, cleanup_list*) noexcept
    {
        return THPVariable_Wrap(src);
    }
};

template <typename T>
struct type_caster<std::vector<std::reference_wrapper<T const>>>
{
    using VectorType = std::vector<std::reference_wrapper<T const>>;

    NB_TYPE_CASTER(VectorType, const_name("List[") + make_caster<T>::Name + const_name("]"));

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        // Not needed for our use case since we only convert C++ to Python
        return false;
    }

    static handle from_cpp(VectorType const& src, rv_policy policy, cleanup_list* cleanup) noexcept
    {

        std::vector<T> result;
        result.reserve(src.size());
        for (auto const& ref : src)
        {
            result.push_back(ref.get());
        }

        return make_caster<std::vector<T>>::from_cpp(result, policy, cleanup);
    }
};

template <>
struct type_caster<torch::ScalarType>
{
    NB_TYPE_CASTER(torch::ScalarType, const_name("torch.dtype"));

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        std::string dtype_name = nb::cast<std::string>(nb::str(src));
        if (dtype_name.substr(0, 6) == "torch.")
        {
            dtype_name = dtype_name.substr(6);
        }

        auto const& dtype_map = c10::getStringToDtypeMap();
        auto it = dtype_map.find(dtype_name);
        if (it != dtype_map.end())
        {
            value = it->second;
            return true;
        }

        return false;
    }

    static handle from_cpp(torch::ScalarType src, rv_policy policy, cleanup_list* cleanup)
    {
        throw std::runtime_error("from_cpp for torch::ScalarType is not implemented");
    }
};
} // namespace detail
} // namespace NB_NAMESPACE
