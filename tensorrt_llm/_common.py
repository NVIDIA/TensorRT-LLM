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
from typing import TYPE_CHECKING

import numpy as np

# isort: off
import torch
import tensorrt as trt

# isort: on

if TYPE_CHECKING:
    from .network import Network
else:
    Network = None

from ._utils import str_dtype_to_trt
from .bindings import MpiComm
from .logger import logger
from .plugin import _load_plugin_lib

net = None

_inited = False


def _init(log_level: object = None) -> None:
    global _inited
    if _inited:
        return
    _inited = True
    # Move to __init__
    if log_level is not None:
        logger.set_level(log_level)

    if os.getenv("TRT_LLM_NO_LIB_INIT", "0") == "1":
        logger.info("Skipping TensorRT LLM init.")
        return

    logger.info("Starting TensorRT LLM init.")

    # load plugin lib
    _load_plugin_lib()

    # load FT decoder layer and torch custom ops
    project_dir = str(Path(__file__).parent.absolute())
    if platform.system() == "Windows":
        ft_decoder_lib = project_dir + "/libs/th_common.dll"
    else:
        ft_decoder_lib = project_dir + "/libs/libth_common.so"
    try:
        torch.classes.load_library(ft_decoder_lib)
        from ._torch.custom_ops import _register_fake

        _register_fake()
    except Exception as e:
        msg = (
            "\nFATAL: Decoding operators failed to load. This may be caused by an incompatibility "
            "between PyTorch and TensorRT-LLM. Please rebuild and install TensorRT-LLM."
        )
        raise ImportError(str(e) + msg)

    MpiComm.local_init()

    logger.info("TensorRT LLM inited.")


def default_net() -> Network:
    assert net, (
        "Use builder to create network first, and use `set_network` or `net_guard` to set it to default"
    )
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
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open(path, "wb") as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


def deserialize_engine(path):
    runtime = trt.Runtime(logger.trt_logger)
    with open(path, "rb") as f:
        logger.info(f"Loading engine from {path}...")
        tik = time.time()

        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine is not None

        tok = time.time()
        t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
        logger.info(f"Engine loaded. Total time: {t}")
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
    assert ret is not None, f"Unsupported dtype: {dtype}"
    return ret


def convert_capsule_to_void_p(capsule):
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(capsule, None)


def get_nparray_from_void_p(void_pointer, elem_size, field_dtype):
    ctypes.pythonapi.PyMemoryView_FromMemory.restype = ctypes.py_object
    ctypes.pythonapi.PyMemoryView_FromMemory.argtypes = [
        ctypes.c_char_p,
        ctypes.c_ssize_t,
        ctypes.c_int,
    ]
    logger.info(f"get_nparray: pointer = {void_pointer}, elem_size = {elem_size}")
    char_pointer = ctypes.cast(void_pointer, ctypes.POINTER(ctypes.c_char))
    np_dtype = field_dtype_to_np_dtype(field_dtype)
    buf_bytes = elem_size * np.dtype(np_dtype).itemsize
    logger.info(f"get_nparray: buf_bytes = {buf_bytes}")
    mem_view = ctypes.pythonapi.PyMemoryView_FromMemory(
        char_pointer, buf_bytes, 0
    )  # number 0 represents PyBUF_READ
    logger.info(f"get_nparray: mem_view = {mem_view}, field_dtype = {field_dtype}")
    buf = np.frombuffer(mem_view, np_dtype)
    return buf


def get_scalar_from_field(field):
    void_p = convert_capsule_to_void_p(field.data)
    np_array = get_nparray_from_void_p(void_p, 1, field.type)
    return np_array[0]


class _BuildingFlag:
    def __enter__(self):
        os.environ["IS_BUILDING"] = "1"

    def __exit__(self, type, value, tb):
        del os.environ["IS_BUILDING"]


def _is_building(f):
    """Use this to decorate functions which are called during engine building/refitting process,
    otherwise, the plugin registration will fail.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        with _BuildingFlag():
            return f(*args, **kwargs)

    return decorated


def check_max_num_tokens(
    max_num_tokens,
    opt_num_tokens,
    max_batch_size,
    max_input_len,
    max_seq_len,
    max_beam_width,
    remove_input_padding,
    enable_context_fmha,
    tokens_per_block,
    multiple_profiles,
):
    if not remove_input_padding:
        if max_num_tokens is not None or opt_num_tokens is not None:
            max_num_tokens = max_batch_size * max_seq_len
            logger.warning(
                "remove_input_padding is not enabled, the specified "
                "max_num_tokens/opt_num_tokens will be ignored."
            )
        return max_num_tokens, opt_num_tokens
    else:
        if max_num_tokens is None:
            max_num_tokens = max_seq_len * max_batch_size
            logger.warning(
                "remove_input_padding is enabled, while max_num_tokens "
                "is not set, setting to max_batch_size*max_seq_len. \n"
                "It may not be optimal to set max_num_tokens=max_batch_size*max_seq_len "
                "when remove_input_padding is enabled, because the number "
                "of packed input tokens are very likely to be smaller, "
                "we strongly recommend to set max_num_tokens according "
                "to your workloads."
            )
        if opt_num_tokens is None and not multiple_profiles:
            opt_num_tokens = min(max_batch_size * max_beam_width, max_num_tokens)
            logger.warning(
                "remove_input_padding is enabled, while opt_num_tokens "
                "is not set, setting to max_batch_size*max_beam_width. \n"
            )
        if max_num_tokens > 16384:
            logger.warning(
                "Specifying a `max_num_tokens` larger than 16384 is usually "
                "not recommended, we do not expect perf gain with that and too "
                "large `max_num_tokens` could possibly exceed the TensorRT "
                "tensor volume, causing runtime errors. "
                f"Got `max_num_tokens` = {max_num_tokens}"
            )
    if max_num_tokens > max_seq_len * max_batch_size:
        logger.warning(
            f"max_num_tokens ({max_num_tokens}) shouldn't be greater than "
            f"max_seq_len * max_batch_size ({max_seq_len * max_batch_size}), "
            f"specifying to max_seq_len * max_batch_size ({max_seq_len * max_batch_size})."
        )
        max_num_tokens = max_seq_len * max_batch_size
    if max_num_tokens < max_input_len and not enable_context_fmha:
        logger.warning(
            f"When enable_context_fmha is not turned on, max_num_tokens ({max_num_tokens}) "
            f"should be at least max_input_len ({max_input_len}), specifying to "
            f"max_input_len ({max_input_len})."
        )
        max_num_tokens = max_input_len
    elif max_num_tokens < tokens_per_block and enable_context_fmha:
        logger.warning(
            f"When enable_context_fmha is turned on, max_num_tokens ({max_num_tokens}) "
            f"should be at least tokens_per_block ({tokens_per_block}), specifying to "
            f"tokens_per_block ({tokens_per_block}). At this time, you also need to enable "
            f"context chunking at runtime, otherwise you may encounter errors."
        )
        max_num_tokens = tokens_per_block

    if opt_num_tokens is not None and opt_num_tokens > max_num_tokens:
        logger.warning(
            f"opt_num_tokens ({opt_num_tokens}) shouldn't be greater than "
            f"max_num_tokens ({max_num_tokens}), "
            f"specifying to max_num_tokens ({max_num_tokens})."
        )
        opt_num_tokens = max_num_tokens

    return max_num_tokens, opt_num_tokens
