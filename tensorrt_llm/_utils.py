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
import linecache
import math
import os
import socket
import struct
import tempfile
import trace
import weakref
from contextlib import contextmanager
from enum import EnumMeta
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import nvtx
from mpi4py import MPI
from mpi4py.util import pkl5
from packaging import version

# isort: off
import torch
import tensorrt as trt
# isort: on

from tensorrt_llm.bindings import DataType, GptJsonConfig, LayerType
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.logger import logger

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


def trt_gte(major: int, minor: int = 0):
    """
    Check if TRT version is greater than or equal to major.minor
    """
    trt_ver = version.parse(trt_version())
    return trt_ver.major >= major and trt_ver.minor >= minor


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


_str_to_binding_dtype_dict = dict(
    bfloat16=DataType.BF16,
    float16=DataType.HALF,
    float32=DataType.FLOAT,
    int64=DataType.INT64,
    int32=DataType.INT32,
    int8=DataType.INT8,
    bool=DataType.BOOL,
    fp8=DataType.FP8,
)
_binding_to_str_dtype = {v: k for k, v in _str_to_binding_dtype_dict.items()}

_binding_dtype_bits = {
    DataType.INT64: 64,
    DataType.FLOAT: 32,
    DataType.INT32: 32,
    DataType.BF16: 16,
    DataType.HALF: 16,
    DataType.BOOL: 8,
    DataType.FP8: 8,
    DataType.INT8: 8,
    DataType.UINT8: 8,
    DataType.NVFP4: 4,
}


def binding_layer_type_to_str(layer_type: LayerType) -> str:
    return layer_type.name.lower()


def binding_to_str_dtype(binding_dtype) -> str:
    ret = _binding_to_str_dtype.get(binding_dtype)
    assert ret is not None, f'Unsupported binding dtype: {binding_dtype}'
    return ret


def binding_dtype_size(dtype: DataType):
    return _binding_dtype_size[dtype]


def get_size_in_bytes(num_elements: int, dtype: DataType):
    total_num_bits = _binding_dtype_bits[dtype] * num_elements
    assert total_num_bits % 8 == 0, f"Total number of bits {total_num_bits} must be divisible by 8"
    return total_num_bits // 8


def str_dtype_to_binding(dtype):
    ret = _str_to_binding_dtype_dict.get(dtype)
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
                              fp8=trt.fp8,
                              nvfp4=trt.fp4)


def str_dtype_to_trt(dtype):
    if dtype == "fp4":
        # Special handling for FP4 since CI's trt version is not recent enough.
        if not hasattr(trt, 'fp4'):
            raise ValueError(
                "fp4 unsupported, trt version needs to be upgraded.")
        return trt.fp4

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


_np_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np_bfloat16: torch.bfloat16,
    np_float8: torch.float8_e4m3fn,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def np_dtype_to_torch(dtype):
    ret = _np_to_torch_dtype_dict.get(dtype)
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
    torch.float8_e4m3fn: trt.fp8,
    torch.qint8: trt.int8,
    torch.bool: trt.bool,
    torch.bfloat16: trt.bfloat16
}


def torch_dtype_to_trt(dtype):
    ret = _torch_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_to_binding_dtype_dict = {
    torch.float16: DataType.HALF,
    torch.float32: DataType.FLOAT,
    torch.int64: DataType.INT64,
    torch.int32: DataType.INT32,
    torch.int8: DataType.INT8,
    torch.float8_e4m3fn: DataType.FP8,
    torch.qint8: DataType.INT8,
    torch.bool: DataType.BOOL,
    torch.bfloat16: DataType.BF16
}


def torch_dtype_to_binding(dtype):
    ret = _torch_to_binding_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_dtype_to_np_typestr_dict = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int8: "|i1",
    torch.float8_e4m3fn: "|i1",
    torch.qint8: "|u1",
    torch.bool: "|b1",
    torch.bfloat16: "<f2",
}


def torch_dtype_to_np_typestr(dtype):
    ret = _torch_dtype_to_np_typestr_dict.get(dtype)
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


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


# mpi4py only exports MPI_COMM_TYPE_SHARED, so we define OMPI_COMM_TYPE_HOST here
OMPI_COMM_TYPE_HOST = 9

comm = pkl5.Intracomm(MPI.COMM_WORLD)


def set_mpi_comm(new_comm):
    global comm
    comm = new_comm


def mpi_comm():
    return comm


local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


def local_mpi_comm():
    return local_comm


# Global TorchDist instance for Ray orchestrator
_torch_comm = None


def set_torch_comm(torch_comm_instance):
    """Set global TorchDist instance"""
    global _torch_comm
    _torch_comm = torch_comm_instance


def torch_comm():
    """Get global TorchDist instance"""
    if _torch_comm is None:
        raise RuntimeError(
            "TorchDist not initialized. Call set_torch_comm() first.")
    return _torch_comm


def mpi_disabled() -> bool:
    """True if TLLM_DISABLE_MPI is set to "1", False otherwise."""
    return os.environ.get("TLLM_DISABLE_MPI") == "1"


def mpi_rank():
    if mpi_disabled():
        try:
            return torch.distributed.get_rank()
        except ValueError:
            # Fallback: return 0 when MPI is absent (Ray / Slurm PMIx)
            return 0
    return mpi_comm().Get_rank() if ENABLE_MULTI_DEVICE else 0


def global_mpi_rank():
    if mpi_disabled():
        # Fallback: return 0 when MPI is absent (Ray / Slurm PMIx)
        return 0

    return MPI.COMM_WORLD.Get_rank() if ENABLE_MULTI_DEVICE else 0


def global_mpi_size():
    return MPI.COMM_WORLD.Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_world_size():
    return mpi_comm().Get_size() if ENABLE_MULTI_DEVICE else 1


def local_mpi_rank():
    return local_comm.Get_rank() if ENABLE_MULTI_DEVICE else 0


def local_mpi_size():
    return local_comm.Get_size() if ENABLE_MULTI_DEVICE else 1


def default_gpus_per_node():
    num_gpus = torch.cuda.device_count()
    num_ranks = local_mpi_size()
    assert num_gpus > 0, "No GPU found on the node"
    if num_ranks > num_gpus:
        logger.warning(f"{num_ranks} MPI ranks will share {num_gpus} GPUs.")
    return min(num_ranks, num_gpus)


def mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Barrier()


def local_mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        local_comm.Barrier()


def mpi_broadcast(obj, root=0):
    return mpi_comm().bcast(obj, root) if global_mpi_size() > 1 else obj


def mpi_allgather(obj):
    return mpi_comm().allgather(obj) if ENABLE_MULTI_DEVICE else obj


def mpi_isend(buf, dest, tag=0):
    # isend in buf-like objects (e.g. numpy array)
    # return request handle if ENABLE_MULTI_DEVICE
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().Isend(buf, dest, tag=tag)
    return None


def mpi_send(buf, dest, tag=0):
    # send in buf-like objects (e.g. numpy array)
    # return request handle if ENABLE_MULTI_DEVICE
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Send(buf, dest, tag=tag)
    return None


def mpi_recv(buf, source, tag):
    # recv in buf-like object (e.g. numpy array)
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().Recv(buf, source, tag=tag)
    return None


def mpi_send_object(obj, dest, tag=0):
    if ENABLE_MULTI_DEVICE:
        mpi_comm().send(obj, dest=dest, tag=tag)


def mpi_isend_object(obj, dest, tag=0):
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().isend(obj, dest=dest, tag=tag)
    return None


def mpi_recv_object(source, tag):
    if ENABLE_MULTI_DEVICE:
        return mpi_comm().recv(source=source, tag=tag)
    return None


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


@lru_cache(maxsize=1)
def get_sm_version():
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


@lru_cache(maxsize=1)
def is_sm_100f(sm_version=None):
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version == 100 or sm_version == 103


def is_trace_enabled(env_var: str):
    value = os.environ.get(env_var, "-1")
    if value == "ALL":
        return True
    if value == "-1":
        # early return w/o calling global_mpi_rank() for Ray path
        return False
    try:
        return int(value) == global_mpi_rank()
    except ValueError:
        return False


def trace_func(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        def globaltrace(frame, why, arg):
            if why == "call":
                code = frame.f_code
                filename = frame.f_globals.get('__file__', None)
                if filename:
                    modulename = trace._modname(filename)
                    if modulename is not None:
                        ignore_it = tracer.ignore.names(filename, modulename)
                        if not ignore_it:
                            print(
                                f"[rank{rank}] --- path: {filename} , funcname: {code.co_name}"
                            )
                            return localtrace
                else:
                    return None

        def localtrace(frame, why, arg):
            if why == "line":
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                bname = os.path.basename(filename)
                print(
                    f"[rank{rank}] {bname}:{lineno}: {linecache.getline(filename, lineno)}",
                    end="")
            return localtrace

        ignoredirs = [
            os.path.dirname(package.__file__) for package in [os, torch, trace]
        ]
        tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)
        rank = global_mpi_rank()
        tracer.globaltrace = globaltrace
        tracer.localtrace = localtrace
        result = tracer.runfunc(func, *args, **kwargs)
        return result

    return wrapper


class BaseEnumMeta(EnumMeta):

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


def supports_inflight_batching(engine_dir):
    config_path = Path(engine_dir) / "config.json"
    json_config = GptJsonConfig.parse_file(config_path)
    model_config = json_config.model_config
    return model_config.supports_inflight_batching


class QuantModeWrapper:

    def __init__(self, objs):
        self.objs = objs

    def __getattr__(self, name):

        def method_wrapper(*args, **kwargs):
            result = False
            for obj in self.objs:
                attr = getattr(obj, name)
                if callable(attr):
                    result = result | attr(*args, **kwargs)
            return result

        return method_wrapper

    def __repr__(self):
        return f"QuantModeWrapper: ({self.objs})"

    def __str__(self):
        obj_strs = [str(obj) for obj in self.objs]
        return f"[{', '.join(obj_strs)}]"

    def __getitem__(self, index):
        return self.objs[index]


PYTHON_DEFAULT_GC_THRESHOLDS = gc.get_threshold()


@contextmanager
def customized_gc_thresholds(gen0_threshold: Optional[int] = None):
    try:
        if gen0_threshold:
            gc.set_threshold(gen0_threshold)
            logger.debug(
                f'Set Python GC threshold to customized value: {gen0_threshold}'
            )
        yield
    finally:
        if gen0_threshold:
            gc.set_threshold(*PYTHON_DEFAULT_GC_THRESHOLDS)
            logger.debug(
                f'Reset Python GC thresholds to default value: {PYTHON_DEFAULT_GC_THRESHOLDS}'
            )


@contextmanager
def _null_context_manager():
    yield


def nvtx_range(msg: str,
               color: str = "grey",
               domain: str = "TensorRT-LLM",
               category: Optional[str] = None):
    """
    Creates an NVTX range annotation for profiling.

    This function returns a context manager that marks the beginning and end of a
    range in NVIDIA Tools Extension (NVTX) profiling tools like Nsight Systems.

    Args:
        msg (str): The message/name for the NVTX range.
        color (str, optional): The color to use for the range in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the range. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the range. Defaults to None.

    Returns:
        contextmanager: A context manager that marks the NVTX range.
    """
    return nvtx.annotate(msg, color=color, domain=domain, category=category)


def nvtx_range_debug(msg: str,
                     color: str = "grey",
                     domain: str = "TensorRT-LLM",
                     category: Optional[str] = None):
    """
    Creates an NVTX range annotation for debugging purposes.

    Similar to nvtx_range, but only creates the range if specific environment
    variables are set, making it suitable for debug profiling.

    Args:
        msg (str): The message/name for the NVTX range.
        color (str, optional): The color to use for the range in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the range. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the range. Defaults to None.

    Returns:
        contextmanager: A context manager that either marks the NVTX range if enabled,
                        or a null context manager that does nothing if disabled.
    """
    if os.getenv("TLLM_LLMAPI_ENABLE_NVTX", "0") == "1" or \
            os.getenv("TLLM_NVTX_DEBUG", "0") == "1":
        return nvtx_range(msg, color=color, domain=domain, category=category)
    else:
        return _null_context_manager()


def nvtx_mark(msg: str,
              color: str = "grey",
              domain: str = "TensorRT-LLM",
              category: Optional[str] = None):
    """
    Creates an NVTX marker for profiling.

    This function places a single marker point in NVIDIA Tools Extension (NVTX)
    profiling tools like Nsight Systems, useful for marking specific events.

    Args:
        msg (str): The message/name for the NVTX marker.
        color (str, optional): The color to use for the marker in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the marker. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the marker. Defaults to None.
    """
    nvtx.mark(msg, color=color, category=category, domain=domain)


def volume(d: Sequence[int]):
    return np.prod(d)


class TensorWrapper:
    """
    A wrapper wraps raw data pointer to a tensor-like object. Could be compatibale with openai triton kernel and be converted to `torch.Tensor` with zero-copy overhead.
    """

    def __init__(
        self,
        data_ptr: int,
        dtype: Union[torch.dtype, str, np.dtype, trt.DataType],
        shape: Sequence[int],
        strides: Optional[Sequence[int]] = None,
    ):
        assert isinstance(data_ptr, int)
        self._data_ptr = data_ptr
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

    def data_ptr(self):
        return self._data_ptr

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return getattr(self, "_shape", None)

    @dtype.setter
    def dtype(self, dtype: Union[torch.dtype, str, np.dtype, trt.DataType]):
        if isinstance(dtype, torch.dtype):
            self._dtype = dtype
        elif isinstance(dtype, str):
            self._dtype = str_dtype_to_torch(dtype)
        elif isinstance(dtype, np.dtype):
            self._dtype = np_dtype_to_torch(dtype)
        elif isinstance(dtype, trt.DataType):
            self._dtype = trt_dtype_to_torch(dtype)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

    @shape.setter
    def shape(self, shape: Sequence[int]):
        self._shape = tuple(int(i) for i in shape)

    def numel(self):
        return volume(self.shape)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape":
            self.shape,
            "typestr":
            torch_dtype_to_np_typestr(self.dtype),
            "data": (self.data_ptr() if self.numel() > 0 else 0, False),
            "strides": [
                i * torch.tensor([], dtype=self.dtype).element_size()
                for i in self.strides
            ] if self.strides is not None else None,
            "version":
            3,
        }

    @staticmethod
    def from_trt_desc(desc: trt.PluginTensorDesc, pointer: int):
        return TensorWrapper(pointer, trt_dtype_to_torch(desc.type), desc.dims)


def convert_to_torch_tensor(
        tensor: Union[TensorWrapper, torch.Tensor]) -> torch.Tensor:
    """
    This function is to convert the `TensorWrapper` to torch.Tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor

    old_ptr = tensor.data_ptr()
    new_tensor = torch.as_tensor(tensor).view(tensor.dtype)
    new_ptr = new_tensor.data_ptr()
    if old_ptr != new_ptr:
        raise RuntimeError(
            "Data pointer mismatch after converting to torch.Tensor")
    return new_tensor


class KVCacheEventSerializer:

    @classmethod
    def get_event_serialize_func(cls, event_type):
        return {
            "KVCacheCreatedData": cls._created_to_json,
            "KVCacheStoredData": cls._stored_to_json,
            "KVCacheStoredBlockData": cls._stored_block_to_json,
            "KVCacheRemovedData": cls._removed_to_json,
            "KVCacheUpdatedData": cls._updated_to_json,
        }.get(event_type, None)

    @classmethod
    def serialize(cls, events):
        if events is None:
            return None

        if not isinstance(events, list):
            return cls.to_json_str(events)

        return [cls.to_json_str(event) for event in events]

    @classmethod
    def to_json_str(cls, event):
        if event is None:
            return {}

        event_type = type(event.data).__name__
        event_serialize_func = cls.get_event_serialize_func(event_type)
        if event_serialize_func is None:
            raise ValueError(f"Unknown KVCache event data type: {event_type}")

        json_str = {
            "event_id": event.event_id,
            "data": event_serialize_func(event.data),
            "window_size": event.window_size,
        }
        if event.attention_dp_rank is not None:
            json_str["attention_dp_rank"] = event.attention_dp_rank

        return json_str

    @staticmethod
    def _created_to_json(data):
        return {
            "type": "created",
            "num_blocks_per_cache_level": data.num_blocks_per_cache_level
        }

    @staticmethod
    def _stored_to_json(data):
        return {
            "type":
            "stored",
            "parent_hash":
            data.parent_hash,
            "blocks": [
                KVCacheEventSerializer._stored_block_to_json(block)
                for block in data.blocks
            ]
        }

    @staticmethod
    def _stored_block_to_json(data):
        return {
            "type":
            "stored_block",
            "block_hash":
            data.block_hash,
            "tokens": [
                KVCacheEventSerializer._unique_tokens_to_json(token)
                for token in data.tokens
            ],
            # "lora_id": data.lora_id, # TODO (shreyasm): enable serialization of lora_id
            "cache_level":
            data.cache_level,
            "priority":
            data.priority
        }

    @staticmethod
    def _removed_to_json(data):
        return {"type": "removed", "block_hashes": data.block_hashes}

    @staticmethod
    def _updated_to_json(data):
        return {
            "type":
            "updated",
            "block_hash":
            data.block_hash,
            "cache_level":
            KVCacheEventSerializer._event_diff_to_json(data.cache_level),
            "priority":
            KVCacheEventSerializer._event_diff_to_json(data.priority)
        }

    @staticmethod
    def _event_diff_to_json(data):
        return {
            "type": "event_diff",
            "new_value": data.new_value,
            "old_value": data.old_value
        }

    @staticmethod
    def _unique_tokens_to_json(data):
        return {
            "type": "unique_token",
            "token_id": data.token_id,
            "token_extra_id": data.token_extra_id
        }


def set_prometheus_multiproc_dir() -> object:
    # Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.10/python/sglang/srt/utils.py#L1266
    global prometheus_multiproc_dir
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.info("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"])
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.info(
        f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")


TORCH_PYBIND11_ABI = None


def torch_pybind11_abi() -> str:
    global TORCH_PYBIND11_ABI
    if TORCH_PYBIND11_ABI is None:
        TORCH_PYBIND11_ABI = f"{torch._C._PYBIND11_COMPILER_TYPE}{torch._C._PYBIND11_STDLIB}{torch._C._PYBIND11_BUILD_ABI}"
    return TORCH_PYBIND11_ABI


@lru_cache(maxsize=1)
def is_device_integrated() -> bool:
    """Check if the current GPU device is integrated (shares physical memory with CPU).

    Integrated GPU systems include DGX Spark and other unified memory architectures.
    This function caches the result to avoid repeated CUDA device property queries.

    Returns:
        bool: True if the GPU is integrated, False otherwise. Returns False if CUDA
              is not available.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties().is_integrated
