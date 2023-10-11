# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import json
import math
import time
from functools import partial

import numpy as np
import tensorrt as trt
import torch

from .logger import logger

fp32_array = partial(np.array, dtype=np.float32)
fp16_array = partial(np.array, dtype=np.float16)
int32_array = partial(np.array, dtype=np.int32)

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


def torch_to_numpy(x):
    if x.dtype != torch.bfloat16:
        return x.numpy()
    return x.view(torch.int16).numpy().view(np_bfloat16)


def trt_version():
    return trt.__version__


def torch_version():
    return torch.__version__


_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int32=np.int32,
    bfloat16=np_bfloat16,
)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_str_to_torch_dtype_dict = dict(
    bfloat16=torch.bfloat16,
    float16=torch.float16,
    float32=torch.float32,
    int32=torch.int32,
    int8=torch.int8,
)


def str_dtype_to_torch(dtype):
    ret = _str_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


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


_np_to_trt_dtype_dict = {
    np.int8: trt.int8,
    np.int32: trt.int32,
    np.float16: trt.float16,
    np.float32: trt.float32,

    # hash of np.dtype('int32') != np.int32
    np.dtype('int8'): trt.int8,
    np.dtype('int32'): trt.int32,
    np.dtype('float16'): trt.float16,
    np.dtype('float32'): trt.float32,
}


def np_dtype_to_trt(dtype):
    if trt_version() >= '7.0' and dtype == np.bool_:
        return trt.bool
    if trt_version() >= '9.0' and dtype == np_bfloat16:
        return trt.bfloat16
    ret = _np_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_np_dtype_dict = {
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.bool: np.bool_,
}


def trt_dtype_to_np(dtype):
    if trt_version() >= '9.0' and dtype == trt.bfloat16:
        return np_bfloat16
    ret = _trt_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_to_np_dtype_dict = {
    torch.float16: np.float16,
    torch.float32: np.float32,
}


def torch_dtype_to_np(dtype):
    ret = _torch_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
}


def trt_dtype_to_torch(dtype):
    if trt_version() >= '9.0' and dtype == trt.bfloat16:
        return torch.bfloat16
    ret = _trt_to_torch_dtype_dict.get(dtype)
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


def dim_resolve_negative(dim, ndim):
    if not isinstance(dim, tuple):
        dim = (dim, )
    pos = []
    for d in dim:
        if d < 0:
            d = ndim + d
        pos.append(d)
    return tuple(pos)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def deserialize_engine(path):
    runtime = trt.Runtime(logger.trt_logger)
    with open(path, 'rb') as f:
        logger.info(f'Loading engine from {path}...')
        tik = time.time()

        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine is not None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Engine loaded. Total time: {t}')
    return engine


def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank()


def mpi_world_size():
    return mpi_comm().Get_size()


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


_field_dtype_to_np_dtype_dict = {
    trt.PluginFieldType.FLOAT16: np.float16,
    trt.PluginFieldType.FLOAT32: np.float32,
    trt.PluginFieldType.FLOAT64: np.float64,
    trt.PluginFieldType.INT8: np.int8,
    trt.PluginFieldType.INT16: np.int16,
    trt.PluginFieldType.INT32: np.int32,
}


def field_dtype_to_np_dtype(dtype):
    ret = _field_dtype_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def convert_capsule_to_void_p(capsule):
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object, ctypes.c_char_p
    ]
    return ctypes.pythonapi.PyCapsule_GetPointer(capsule, None)


def get_nparray_from_void_p(void_pointer, elem_size, field_dtype):
    ctypes.pythonapi.PyMemoryView_FromMemory.restype = ctypes.py_object
    ctypes.pythonapi.PyMemoryView_FromMemory.argtypes = [
        ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_int
    ]
    logger.info(
        f'get_nparray: pointer = {void_pointer}, elem_size = {elem_size}')
    char_pointer = ctypes.cast(void_pointer, ctypes.POINTER(ctypes.c_char))
    np_dtype = field_dtype_to_np_dtype(field_dtype)
    buf_bytes = elem_size * np.dtype(np_dtype).itemsize
    logger.info(f'get_nparray: buf_bytes = {buf_bytes}')
    mem_view = ctypes.pythonapi.PyMemoryView_FromMemory(
        char_pointer, buf_bytes, 0)  # number 0 represents PyBUF_READ
    logger.info(
        f'get_nparray: mem_view = {mem_view}, field_dtype = {field_dtype}')
    buf = np.frombuffer(mem_view, np_dtype)
    return buf


def get_scalar_from_field(field):
    void_p = convert_capsule_to_void_p(field.data)
    np_array = get_nparray_from_void_p(void_p, 1, field.type)
    return np_array[0]
