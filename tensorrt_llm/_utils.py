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
import copy
import gc
import inspect
import json
import math
import struct
import weakref
from dataclasses import asdict
from enum import EnumMeta
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
from packaging import version

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

# isort: off
import torch
import tensorrt as trt
# isort: on

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})


def torch_to_numpy(x: torch.Tensor):
    assert isinstance(x, torch.Tensor), \
        f'x must be a torch.Tensor object, but got {type(x)}.'
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).detach().cpu().numpy().view(np_bfloat16)
    elif x.dtype == torch.float8_e4m3fn:
        return x.view(torch.int8).detach().cpu().numpy().view(np_float8)
    else:
        return x.detach().cpu().numpy()


def numpy_to_torch(x):
    if x.dtype == np_bfloat16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    elif x.dtype == np_float8:
        return torch.from_numpy(x.view(np.int8)).view(torch.float8_e4m3fn)
    else:
        return torch.from_numpy(x)


def numpy_to_dtype(x, dtype: str):
    if str_dtype_to_np(dtype) == x.dtype:
        return x
    if x.dtype not in [np_bfloat16, np_float8
                       ] and dtype not in ['bfloat16', 'fp8']:
        return x.astype(str_dtype_to_np(dtype))
    else:
        return torch_to_numpy(numpy_to_torch(x).to(str_dtype_to_torch(dtype)))


fp32_array = partial(np.array, dtype=np.float32)
fp16_array = partial(np.array, dtype=np.float16)
int32_array = partial(np.array, dtype=np.int32)
int64_array = partial(np.array, dtype=np.int64)
bool_array = partial(np.array, dtype=np.bool_)


def dims_array(x):
    is_int64_dims = True
    try:
        trt.Dims([np.iinfo(np.int64).max])
    except TypeError:
        is_int64_dims = False
    return int64_array(x) if is_int64_dims else int32_array(x)


def bf16_array(x):
    x = torch.tensor(x, dtype=torch.bfloat16)
    x = torch_to_numpy(x)
    return x


def numpy_array(data, trt_dtype):
    # convenient wrapper due to numpy not support bf16 yet
    if trt_dtype == trt.bfloat16:
        return bf16_array(data)
    return np.array(data, trt_dtype_to_np(trt_dtype))


def copy_torch_to_numpy(x: torch.Tensor, ndarray: np.array):
    if x.dtype == torch.bfloat16:
        torch.from_numpy(ndarray.view(np.int16)).copy_(x.view(torch.int16))
    elif x.dtype == torch.float8_e4m3fn:
        torch.from_numpy(ndarray.view(np.int8)).copy_(x.view(torch.int8))
    else:
        torch.from_numpy(ndarray).copy_(x)
    return ndarray


def trt_version():
    return trt.__version__


# TRT supports strongly_typed in 9.1
def support_strongly_type():
    return version.parse(trt_version()) >= version.parse("9.1.0")


# Check if TRT version >= 10
def trt_gte_10():
    return version.parse(trt_version()).major > 9


def torch_version():
    return torch.__version__


_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int64=np.int64,
    int32=np.int32,
    int8=np.int8,
    bool=np.bool_,
    bfloat16=np_bfloat16,
    fp8=np_float8,
)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_str_to_torch_dtype_dict = dict(
    bfloat16=torch.bfloat16,
    float16=torch.float16,
    float32=torch.float32,
    int64=torch.int64,
    int32=torch.int32,
    int8=torch.int8,
    bool=torch.bool,
    fp8=torch.float8_e4m3fn,
)


def str_dtype_to_torch(dtype):
    ret = _str_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_dtype_to_str_dict = {v: k for k, v in _str_to_torch_dtype_dict.items()}


def torch_dtype_to_str(dtype):
    return _torch_dtype_to_str_dict[dtype]


_str_to_trt_dtype_dict = dict(float16=trt.float16,
                              float32=trt.float32,
                              int64=trt.int64,
                              int32=trt.int32,
                              int8=trt.int8,
                              bool=trt.bool,
                              bfloat16=trt.bfloat16,
                              fp8=trt.fp8)


def str_dtype_to_trt(dtype):
    ret = _str_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_str_dtype_dict = {v: k for k, v in _str_to_trt_dtype_dict.items()}


def trt_dtype_to_str(dtype: trt.DataType) -> str:
    assert isinstance(dtype, trt.DataType)
    return _trt_to_str_dtype_dict[dtype]


_np_to_trt_dtype_dict = {
    np.int8: trt.int8,
    np.int32: trt.int32,
    np.int64: trt.int64,
    np.float16: trt.float16,
    np.float32: trt.float32,
    np.bool_: trt.bool,

    # hash of np.dtype('int32') != np.int32
    np.dtype('int8'): trt.int8,
    np.dtype('int32'): trt.int32,
    np.dtype('int64'): trt.int64,
    np.dtype('float16'): trt.float16,
    np.dtype('float32'): trt.float32,
    np.dtype('bool'): trt.bool,
    np_bfloat16: trt.bfloat16,
    np_float8: trt.fp8,
}


def np_dtype_to_trt(dtype):
    ret = _np_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_np_dtype_dict = {
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.bool: np.bool_,
    trt.bfloat16: np_bfloat16,
    trt.fp8: np_float8,
}


def trt_dtype_to_np(dtype):
    ret = _trt_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_to_np_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.bfloat16: np_bfloat16,
    torch.float8_e4m3fn: np_float8,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def torch_dtype_to_np(dtype):
    ret = _torch_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int64: torch.int64,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.bfloat16: torch.bfloat16,
    trt.fp8: torch.float8_e4m3fn,
}


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def is_same_dtype(type_a: Union[str, trt.DataType],
                  type_b: Union[str, trt.DataType]) -> bool:
    if isinstance(type_a, str):
        type_a = str_dtype_to_trt(type_a)

    if isinstance(type_b, str):
        type_b = str_dtype_to_trt(type_b)

    return type_a == type_b


_torch_to_trt_dtype_dict = {
    torch.float16: trt.float16,
    torch.float32: trt.float32,
    torch.int64: trt.int64,
    torch.int32: trt.int32,
    torch.int8: trt.int8,
    torch.qint8: trt.int8,
    torch.bool: trt.bool,
    torch.bfloat16: trt.bfloat16
}


def torch_dtype_to_trt(dtype):
    ret = _torch_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask"""
    if not isinstance(dim, tuple):
        dim = (dim, )

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


def trt_axes_to_dim(axes: int) -> List[int]:
    """Converts tensorrt axes bitmask to dims"""
    dim = []
    for i in range(32):
        if axes & (1 << i):
            dim.append(i)

    return dim


def dim_resolve_negative(dim, ndim):
    if not isinstance(dim, tuple):
        dim = (dim, )
    pos = []
    for d in dim:
        if d < 0:
            d = ndim + d
        pos.append(d)
    return tuple(pos)


# mpi4py only exports MPI_COMM_TYPE_SHARED, so we define OMPI_COMM_TYPE_HOST here
OMPI_COMM_TYPE_HOST = 9


def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank() if ENABLE_MULTI_DEVICE else 0


def mpi_world_size():
    return mpi_comm().Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_barrier():
    mpi_comm().Barrier()


def mpi_broadcast(obj, root=0):
    return mpi_comm().bcast(obj, root)


def pad_vocab_size(vocab_size, tp_size):
    return int(math.ceil(vocab_size / tp_size) * tp_size)


def to_dict(obj):
    return copy.deepcopy(obj.__dict__)


def to_json_string(obj):
    if not isinstance(obj, dict):
        obj = to_dict(obj)
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def to_json_file(obj, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as writer:
        writer.write(to_json_string(obj))


def numpy_fp32_to_bf16(src):
    # Numpy doesn't support bfloat16 type
    # Convert float32 to bfloat16 manually and assign with bf16 abstract type
    original_shape = src.shape
    src = src.flatten()
    src = np.ascontiguousarray(src)

    assert src.dtype == np.float32
    dst = np.empty_like(src, dtype=np.uint16)
    for i in range(len(dst)):
        bytes = struct.pack('<f', src[i])
        dst[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return dst.reshape(original_shape).view(np_bfloat16)


_extra_attrs_by_object: Dict[int, Dict[str, Any]] = {}


def get_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return None
    extra_attrs = _extra_attrs_by_object[id(obj)]
    return extra_attrs.get(attr_name)


def _clean_extra_attrs(obj_id):
    if obj_id in _extra_attrs_by_object:
        del _extra_attrs_by_object[obj_id]


def set_extra_attr(obj, attr_name, value):
    if id(obj) not in _extra_attrs_by_object:
        _extra_attrs_by_object[id(obj)] = {}
        weakref.finalize(obj, _clean_extra_attrs, id(obj))
    _extra_attrs_by_object[id(obj)][attr_name] = value


def has_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return False
    return attr_name in _extra_attrs_by_object[id(obj)]


def set_obj_attrs(
    obj: torch.Tensor,
    ojb_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a object.

    This method is used to set attributes on a object. This method
    will not overwrite existing attributes.
    """
    if ojb_attrs is None:
        return
    for key, value in ojb_attrs.items():
        assert not hasattr(
            obj, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(obj, key, value)


def get_init_params(obj, cls=None):
    """
    Get all parameters in object's __init__.
    Use cls's __init__ as filter if cls provided.
    """
    names = None
    if cls is not None:
        names = set(list(inspect.signature(cls.__init__).parameters)[1:])
    return {
        name: getattr(obj, name)
        for name in list(inspect.signature(obj.__class__.__init__).parameters)
        [1:] if names is None or name in names
    }


def release_gc():
    ''' Release memory allocated by PyTorch and Python garbage collector explicitly and immediately.
    This could be used when some states might be kept in memory even after the variables are deleted.
    '''
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class DictConversion:

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        obj = cls()
        fields = obj.__dataclass_fields__
        for key, value in config.items():
            assert hasattr(obj, key)
            field_cls = fields[key].type
            if (isinstance(field_cls, type)
                    and issubclass(field_cls, DictConversion)
                    and isinstance(value, dict)):
                value = field_cls.from_dict(value)
            setattr(obj, key, value)
        return obj

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json_file(cls, file):
        with open(file) as f:
            return cls.from_dict(json.load(f))

    def set_defaults(self, **kwargs):
        for key, default in kwargs.items():
            value = getattr(self, key)
            if (value is None
                    or (isinstance(value, (list, dict)) and len(value) == 0)):
                setattr(self, key, default)


class BaseEnumMeta(EnumMeta):

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
