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
import contextlib
import ctypes
import os
import platform
import time
from functools import wraps
from pathlib import Path

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on

from ._utils import str_dtype_to_trt
from .logger import logger
from .plugin import _load_plugin_lib

net = None

_inited = False


def _init(log_level=None):
    global _inited
    if _inited:
        return
    _inited = True
    # Move to __init__
    if log_level is not None:
        logger.set_level(log_level)

    # load plugin lib
    _load_plugin_lib()

    # load FT decoder layer
    project_dir = str(Path(__file__).parent.absolute())
    if platform.system() == "Windows":
        ft_decoder_lib = project_dir + '/libs/th_common.dll'
    else:
        ft_decoder_lib = project_dir + '/libs/libth_common.so'
    try:
        torch.classes.load_library(ft_decoder_lib)
    except Exception as e:
        msg = '\nFATAL: Decoding operators failed to load. This may be caused by the incompatibility between PyTorch and TensorRT-LLM. Please rebuild and install TensorRT-LLM.'
        raise ImportError(str(e) + msg)

    global net
    logger.info('TensorRT-LLM inited.')


def default_net():
    assert net, "Use builder to create network first, and use `set_network` or `net_guard` to set it to default"
    return net


def default_trtnet():
    return default_net().trt_network


def set_network(network):
    global net
    net = network


def switch_net_dtype(cur_dtype):
    prev_dtype = default_net().dtype
    default_net().dtype = cur_dtype
    return prev_dtype


@contextlib.contextmanager
def precision(dtype):
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)
    prev_dtype = switch_net_dtype(dtype)
    yield
    switch_net_dtype(prev_dtype)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open(path, 'wb') as f:
        f.write(engine)
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


class _BuildingFlag:

    def __enter__(self):
        os.environ['IS_BUILDING'] = '1'

    def __exit__(self, type, value, tb):
        del os.environ['IS_BUILDING']


def _is_building(f):
    '''Use this to decorate functions which are called during engine building/refiting process,
    otherwise, the plugin registration will fail.
    '''

    @wraps(f)
    def decorated(*args, **kwargs):
        with _BuildingFlag():
            return f(*args, **kwargs)

    return decorated
