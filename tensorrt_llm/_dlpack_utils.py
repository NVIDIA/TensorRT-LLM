# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_char_p,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_void_p,
)

import torch


# Define data structures required for DLPack
class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", c_uint8),  # Data type code, e.g., 2 for float
        ("bits", c_uint8),  # Number of bits per element, e.g., 32
        ("lanes", c_uint16),  # Number of lanes, usually 1
    ]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", c_int),  # Device type, typically 2 for GPU
        ("device_id", c_int),  # Device ID, usually 0 for default GPU
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),  # Data pointer
        ("device", DLDevice),  # Device information
        ("ndim", c_int),  # Number of dimensions
        ("dtype", DLDataType),  # Data type
        ("shape", POINTER(c_int64)),  # Pointer to array of dimension sizes
        (
            "strides",
            POINTER(c_int64),
        ),  # Pointer to strides array (can be NULL for default contiguous layout)
        ("byte_offset", c_size_t),  # Byte offset (usually 0)
    ]


# Deleter type for DLManagedTensor
DLManagedTensorDeleter = CFUNCTYPE(None, POINTER(ctypes.c_void_p))  # Not used directly here


# Define DLManagedTensor structure, with deleter prototype void(*deleter)(DLManagedTensor*)
class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", CFUNCTYPE(None, POINTER(DLManagedTensor))),
]


# Rank of the tensors this module produces: sizes the block's shape/strides arrays and is
# reported as ndim, so all three move together.
_NDIM = 2


# The DLManagedTensor plus the shape and stride arrays it points at, in one contiguous
# allocation so a single free reclaims all three.
class _DLPackBlock(ctypes.Structure):
    _fields_ = [
        ("managed_tensor", DLManagedTensor),
        ("shape", c_int64 * _NDIM),
        ("strides", c_int64 * _NDIM),
    ]


_raw_calloc = ctypes.pythonapi.PyMem_RawCalloc
_raw_calloc.restype = c_void_p
_raw_calloc.argtypes = [c_size_t, c_size_t]

_raw_free = ctypes.pythonapi.PyMem_RawFree
_raw_free.restype = None
_raw_free.argtypes = [c_void_p]

_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = c_int
_capsule_is_valid.argtypes = [ctypes.py_object, c_char_p]

# py_object restype so ctypes adopts the reference PyCapsule_New returns (and raises if it
# returns NULL) instead of us leaking it.
_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.restype = ctypes.py_object
_new_capsule.argtypes = [c_void_p, c_char_p, c_void_p]


# The deleter DLPack requires the producer to supply: it releases the block, and only the
# block -- the tensor data stays owned by the caller.
#
# The consumer invokes this from its storage destructor, which for PyTorch runs inside
# ~TensorImpl, after THPVariable_clear has already freed the tensor's __dict__. So the block
# must not depend on any Python object's lifetime; hence raw memory owned by this deleter
# rather than a ctypes object kept alive by a Python reference.
@CFUNCTYPE(None, POINTER(DLManagedTensor))
def _free_dlpack_block(dmt_ptr):
    # The DLManagedTensor is the first member of the block, so their addresses coincide.
    _raw_free(dmt_ptr)


class CapsuleWrapper:
    """
    Holds the PyCapsule and owns its backing DLPack block until a consumer takes over.

    The block has exactly one owner. Importing the capsule (e.g. via
    torch.utils.dlpack.from_dlpack) hands ownership to the consumer, which calls
    _free_dlpack_block once the imported tensor dies -- possibly long after this wrapper is
    gone. An unconsumed capsule's block stays ours, and __del__ releases it.

    Telling those two cases apart relies on the DLPack requirement that a consumer rename
    the capsule to "used_dltensor" when it takes ownership; a consumer that imported without
    renaming would leave both sides believing they own the block.
    """

    def __init__(self, capsule, block_addr):
        self.capsule = capsule  # The main PyCapsule object that can be passed to other libraries
        self._block_addr = block_addr

    def __del__(self):
        # A capsule still answering to "dltensor" was never imported (see class docstring),
        # so its block is still ours to free.
        if _capsule_is_valid(self.capsule, b"dltensor"):
            _raw_free(self._block_addr)


def create_dlpack_capsule(ptr, segment_size, segment_stride, num_segments, torch_dtype, dev_id):
    """
    Parameters:
      ptr: GPU memory address obtained from cudaMalloc (Python int)
      segment_size: Memory size of each segments in bytes
      segment_stride: Memory stride size between segments in bytes
      num_segments: Number of segments
      torch_dtype: torch dtype
      dev_id: device id.
    Returns:
      A PyCapsule object compliant with DLPack specification, which can be directly converted to a
      tensor using torch.utils.dlpack.from_dlpack
    """
    bits_per_elements = 0
    dldata_type_code = 0
    # refer to https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L160
    if torch_dtype in [
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    ]:
        bits_per_elements = torch.finfo(torch_dtype).bits
        dldata_type_code = 2
    elif torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 0
    elif torch_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 1
    else:
        raise NotImplementedError(torch_dtype)
    bytes_per_element = bits_per_elements // 8
    # Raw memory rather than ctypes objects, because the consumer outlives every Python
    # reference held here -- see _free_dlpack_block for why.
    block_addr = _raw_calloc(1, ctypes.sizeof(_DLPackBlock))
    if not block_addr:
        raise MemoryError("Failed to allocate DLPack block")
    block = _DLPackBlock.from_address(block_addr)
    managed_tensor = block.managed_tensor
    # Shape (constructing a one-dimensional tensor here) and strides, in-place in the block
    block.shape[:] = (num_segments, segment_size // bytes_per_element)
    block.strides[:] = (segment_stride // bytes_per_element, 1)
    # Construct DLTensor
    dltensor = managed_tensor.dl_tensor
    dltensor.data = c_void_p(ptr)
    # Set device information: GPU (device_type=2) and device_id=dev_id (modify as needed)
    dltensor.device = DLDevice(device_type=2, device_id=dev_id)
    dltensor.ndim = _NDIM
    dltensor.dtype = DLDataType(code=dldata_type_code, bits=bits_per_elements, lanes=1)
    dltensor.shape = block.shape
    dltensor.strides = block.strides
    # byte_offset and manager_ctx stay 0/NULL from the calloc above.
    managed_tensor.deleter = _free_dlpack_block
    try:
        capsule = _new_capsule(block_addr, b"dltensor", None)
    except Exception:
        _raw_free(block_addr)
        raise
    # The wrapper owns the block until a consumer imports the capsule and takes it over
    return CapsuleWrapper(capsule, block_addr)


def pack_strided_memory(
    ptr: int, segment_size: int, segment_stride: int, num_segments: int, dtype: torch.dtype, dev_id
):
    """
    Pack GPU memory into a PyTorch tensor with specified stride.

    Parameters:
        ptr: GPU memory address obtained from cudaMalloc
        segment_size: Memory size of each segment in bytes
        segment_stride: Memory stride size between segments in bytes
        num_segments: Number of segments
        dtype: PyTorch data type for the resulting tensor
        dev_id: CUDA device ID

    Returns:
        PyTorch tensor that references the provided memory

    Note:
        This function creates a new DLPack capsule each time it's called,
        even with the same pointer. Each capsule is consumed only once.
    """
    # Create a new capsule each time
    capsule_wrapper = create_dlpack_capsule(
        ptr, segment_size, segment_stride, num_segments, dtype, dev_id
    )
    torch_tensor = torch.utils.dlpack.from_dlpack(capsule_wrapper.capsule)
    torch_tensor._capsule_wrapper = capsule_wrapper
    return torch_tensor
