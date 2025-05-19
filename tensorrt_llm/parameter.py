# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import weakref
from typing import Optional, Sequence, Union

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on

from ._common import default_net
from ._utils import (copy_torch_to_numpy, np_dtype_to_trt, str_dtype_to_trt,
                     torch_to_numpy, trt_dtype_to_np, trt_dtype_to_torch)
from .functional import Tensor, constant
from .logger import logger
from .network import Network


class Parameter:
    _DEFAULT_DTYPE = trt.DataType.FLOAT

    def __init__(self,
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 shape: Sequence[int] = None,
                 dtype: Union[str, trt.DataType] = None,
                 is_buffer: bool = False,
                 prefer_managed=False):
        if dtype is None:
            logger.warning(
                f'Parameter dtype is None, using default dtype: {self._DEFAULT_DTYPE}, it is recommended to always specify dtype explicitly'
            )
        dtype = self._DEFAULT_DTYPE if dtype is None else dtype
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        self._dtype: trt.DataType = dtype
        if value is None:
            assert isinstance(shape, (
                list,
                tuple)), f"shape must be list or tuple, receive {(type(shape))}"
            self._shape = tuple(shape)
            self._value = None
        else:
            self._shape = value.shape
            self._value = self._regularize_value(value)
        self.is_buffer = is_buffer
        self._prefer_managed = prefer_managed
        self._tensor: Tensor = None
        self._network: weakref.ref = None
        self._name = None
        self.need_transpose = False

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    def _create_managed_tensor(self, network) -> Tensor:
        num = len(network._inputs)
        self._name = f"managed_constant_{num}"

        if self._value is None or (isinstance(self._value, np.ndarray)
                                   and not self._value.flags['C_CONTIGUOUS']):
            value_old = self._value
            self._value = np.empty(self._shape, trt_dtype_to_np(self._dtype))
            network._register_unfilled_weights(
                # use updated self._shape here
                self._name,
                self._value,
                value_old)
        return Tensor(name=self._name, dtype=self._dtype, shape=self._shape)

    def get_managed_tensor(self, network: Network) -> Tensor:
        if self._network is None or self._network() != network:
            self._network = weakref.ref(network)
            self._tensor = network.get_parameter_tensor(self)
            if self._tensor is None:
                self._tensor = self._create_managed_tensor(network)
                network.set_parameter_tensor(self, self._tensor)
        return self._tensor

    def _create_constant_tensor(self) -> Tensor:
        if (self._value is not None and isinstance(self._value, np.ndarray)
                and self._value.flags['C_CONTIGUOUS']):
            lower_type = None
            lower_shape = None
            # workaround for reinterpreted data type
            dtype = self._value.dtype
            if (self.dtype == trt.fp4 or self.dtype
                    == trt.fp8) and (dtype == np.uint8 or dtype == np.int8
                                     or dtype == np.int32 or dtype == np.int64):
                lower_type = self.dtype
                lower_shape = self.shape

            self._value = constant(self._value, lower_type, lower_shape)
            return self._value
        elif self._value is None or isinstance(self._value, np.ndarray):
            if self._dtype == trt.fp4:
                shape = list(self._shape)
                assert shape[
                    -1] % 16 == 0, "For FP4, the last dimension of the shape should be multiple of 16"
                shape[-1] = shape[-1] // 16
                dtype = np.int64
            else:
                shape = self._shape
                dtype = trt_dtype_to_np(self._dtype)
            ndarray = np.empty(shape, dtype)
            tensor = constant(ndarray, self._dtype, self._shape)
            default_net()._register_unfilled_weights(tensor.producer.name,
                                                     ndarray, self._value)
            return tensor

    def get_constant_tensor(self, network: Network) -> Tensor:
        if self._network is None or self._network() != network:
            self._network = weakref.ref(network)
            self._tensor = network.get_parameter_tensor(self)
            if self._tensor is None:
                self._tensor = self._create_constant_tensor()
                self._name = self._tensor.producer.name
                network.set_parameter_tensor(self, self._tensor)
        return self._tensor

    def get_tensor(self, network) -> Tensor:
        if self.is_managed(network):
            return self.get_managed_tensor(network)
        else:
            return self.get_constant_tensor(network)

    def is_managed(self, network):
        if network is None:
            network = default_net()
        return self._prefer_managed and network.plugin_config.manage_weights

    @property
    def value(self) -> Tensor:
        return self.get_tensor(default_net())

    @classmethod
    def xavier_init(cls, weights: np.ndarray):
        shape = weights.shape
        dtype = np_dtype_to_trt(weights.dtype)
        if len(shape) == 2:
            # Xavier initialization see https://paperswithcode.com/method/xavier-initialization
            v_range = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
        else:
            v_range = 0.1

        if dtype == trt.DataType.INT8 or dtype == trt.DataType.INT32 or dtype == trt.DataType.INT64:
            range_map = {
                trt.DataType.INT8: 128,
                trt.DataType.INT32: 2**31,
                trt.DataType.INT64: 2**63
            }
            upper = math.ceil(range_map[dtype] * v_range)
            value = torch.randint(-upper,
                                  upper, (shape),
                                  dtype=trt_dtype_to_torch(dtype),
                                  device='cuda')
            # value ~ U[int(-128 * v_range), int(128 * v_range)]
        elif dtype == trt.DataType.FP8:
            value = torch.rand((shape), device='cuda') * 2 - 1
            # value ~ U[-v_range, v_range]
            value = value * v_range
            value = value.to(trt_dtype_to_torch(dtype))
        else:
            value = torch.rand(
                (shape), dtype=trt_dtype_to_torch(dtype), device='cuda') * 2 - 1
            # value ~ U[-v_range, v_range]
            value = value * v_range

        copy_torch_to_numpy(value, weights)

    def is_inited(self) -> bool:
        return self._value is not None

    @property
    def raw_value(self) -> np.ndarray:
        if self._value is None:
            dtype = trt_dtype_to_np(self.dtype)
            self._value = np.empty(self.shape, dtype)
            Parameter.xavier_init(self._value)
        assert isinstance(
            self._value, np.ndarray
        ), "Must be np.ndarray. Proper usage: get parameter.raw_value before getting parameter.value"
        return self._value

    @value.setter
    def value(self, v: Union[np.ndarray, torch.Tensor]):
        v = self._regularize_value(v)

        if v.shape != self.shape and v.ndim == 0 and max(self.shape) == 1:
            # convert the scalar into a tensor which each dim is 1.
            v = v.reshape(self.shape)

        if self.dtype == trt.fp4:
            assert v.shape[:-1] == self.shape[:-1] and v.shape[-1] == self.shape[-1] // 2 // v.dtype.itemsize, \
                f'For FP4, the shape of the value should be the same as the original shape, ' \
                f'except the last dimension should be half of the original shape. ' \
                f'Updated: {v.shape}, original: {self.shape}'
        else:
            assert v.shape == self.shape, \
                f'The value updated is not the same shape as the original. ' \
                f'Updated: {v.shape}, original: {self.shape}'
        if (self.dtype == trt.fp4 or self.dtype
                == trt.fp8) and (v.dtype == np.int8 or v.dtype == np.uint8
                                 or v.dtype == np.int32 or v.dtype == np.int64):
            pass
        else:
            dtype = np_dtype_to_trt(v.dtype)
            if self.dtype != dtype:
                logger.warning(
                    f"Parameter was initialized as {self.dtype} but set to {dtype}"
                )
                self._dtype = dtype
        self._value = v

    def set_value_or_dummy(self, v: Union[np.ndarray, torch.Tensor]):
        v = self._regularize_value(v)
        if v.shape != self._shape:
            self.value = np.empty(self._shape, trt_dtype_to_np(self._dtype))
            return

        self.value = v

    def set_name(self, name: str, network):
        self._name = name
        if self.is_managed(network):
            self._get_weights(network).name = name
            return True
        else:
            return network.trt_network.set_weights_name(
                self._get_weights(network), name)

    def _get_weights(self, network) -> trt.Weights | Tensor | None:
        tensor = network.get_parameter_tensor(self)
        if self.is_managed(network):
            return tensor
        elif tensor is not None:
            tensor.producer.__class__ = trt.IConstantLayer
            return tensor.producer.weights
        else:
            return None

    def _regularize_value(self, value):
        if isinstance(value, np.ndarray):
            return value

        elif isinstance(value, torch.distributed.tensor.DTensor):
            return value.to_local().cpu().numpy()
        elif isinstance(value, torch.Tensor):
            return torch_to_numpy(value)
        raise TypeError(
            f'Expected numpy.ndarray or torch.Tensor, got {type(value)}')
