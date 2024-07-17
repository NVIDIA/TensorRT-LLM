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


class Parameter:
    _DEFAULT_DTYPE = trt.DataType.FLOAT

    def __init__(self,
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 shape: Sequence[int] = None,
                 dtype: Union[str, trt.DataType] = None,
                 is_buffer: bool = False):
        if dtype is None:
            logger.warning(
                f'Parameter dtype is None, using default dtype: {self._DEFAULT_DTYPE}, it is recommended to always specify dtype explicitly'
            )
        dtype = self._DEFAULT_DTYPE if dtype is None else dtype
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        self._dtype: trt.DataType = dtype
        if value is None:
            assert isinstance(shape, (list, tuple))
            self._shape = tuple(shape)
            self._value = None
        else:
            self._shape = value.shape
            self._value = self._regularize_value(value)
        self.is_buffer = is_buffer

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def value(self) -> Tensor:
        if (self._value is not None and isinstance(self._value, np.ndarray)
                and self._value.flags['C_CONTIGUOUS']):
            self._value = constant(self._value)
        elif self._value is None or isinstance(self._value, np.ndarray):
            dtype = trt_dtype_to_np(self.dtype)
            ndarray = np.empty(self.shape, dtype)
            value = self._value
            self._value = constant(ndarray)
            default_net()._register_unfilled_weights(self._value.producer.name,
                                                     ndarray, value)
        return self._value

    @classmethod
    def xavier_init(cls, weights: np.ndarray):
        shape = weights.shape
        dtype = np_dtype_to_trt(weights.dtype)
        if len(shape) == 2:
            # Xavier initialization see https://paperswithcode.com/method/xavier-initialization
            v_range = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
        else:
            v_range = 0.1

        if dtype == trt.DataType.INT8:
            upper = math.ceil(128 * v_range)
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

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
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

        assert v.shape == self.shape, \
            f'The value updated is not the same shape as the original. ' \
            f'Updated: {v.shape}, original: {self.shape}'
        dtype = np_dtype_to_trt(v.dtype)
        if self.dtype != dtype:
            logger.warning(
                f"Parameter was initialized as {self.dtype} but set to {dtype}")
        self._value = v

    def set_value_or_dummy(self, v: Union[np.ndarray, torch.Tensor]):
        v = self._regularize_value(v)
        if v.shape != self._shape:
            self.value = np.empty(self._shape, trt_dtype_to_np(self._dtype))
            return

        self.value = v

    def _get_weights(self) -> trt.Weights:
        if isinstance(self._value, Tensor):
            self._value.producer.__class__ = trt.IConstantLayer
            return self._value.producer.weights
        else:
            return None

    def _regularize_value(self, value):
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, torch.Tensor):
            return torch_to_numpy(value)
        raise TypeError(
            f'Expected numpy.ndarray or torch.Tensor, got {type(value)}')
