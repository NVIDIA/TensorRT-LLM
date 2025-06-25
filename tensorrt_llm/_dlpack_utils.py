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
import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_void_p,
    pointer,
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


# A no-op deleter that doesn't perform any operation
@CFUNCTYPE(None, POINTER(DLManagedTensor))
def no_op_deleter(dmt_ptr):
    # You can also call cudaFree here if you want to free memory when the tensor's lifecycle ends
    pass


# Wrapper class to prevent Python garbage collection of DLPack-related objects
class CapsuleWrapper:
    """
    A wrapper class that holds references to the PyCapsule and its associated data.

    This class prevents Python's garbage collector from collecting the shape_array and
    managed_tensor objects while the capsule is still in use. It serves as a container
    to maintain the lifecycle of all DLPack-related objects.
    """

    def __init__(self, capsule, shape_array, managed_tensor):
        """
        Initialize the CapsuleWrapper with the necessary objects.

        Parameters:
            capsule: The PyCapsule object that follows the DLPack protocol
            shape_array: The array containing tensor shape information
            managed_tensor: The DLManagedTensor instance that the capsule points to
        """
        self.capsule = capsule  # The main PyCapsule object that can be passed to other libraries
        self._shape_array = shape_array  # Keep reference to prevent garbage collection
        self._managed_tensor = managed_tensor  # Keep reference to prevent garbage collection


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
    # Allocate space for shape (constructing a one-dimensional tensor here)
    ShapeArrayType = c_int64 * 2  # 1 dimension
    shape_array = ShapeArrayType(num_segments, segment_size // bytes_per_element)
    stride_array = ShapeArrayType(segment_stride // bytes_per_element, 1)
    # Set device information: GPU (device_type=2) and device_id=dev_id (modify as needed)
    device = DLDevice(device_type=2, device_id=dev_id)
    # Set data type
    dtype = DLDataType(code=dldata_type_code, bits=bits_per_elements, lanes=1)
    # Construct DLTensor
    dltensor = DLTensor()
    dltensor.data = c_void_p(ptr)
    dltensor.device = device
    dltensor.ndim = 2
    dltensor.dtype = dtype
    dltensor.shape = ctypes.cast(shape_array, POINTER(c_int64))
    dltensor.strides = ctypes.cast(stride_array, POINTER(c_int64))
    dltensor.byte_offset = 0
    # Construct DLManagedTensor and set deleter to no-op (you can also call cudaFree here)
    managed_tensor = DLManagedTensor()
    managed_tensor.dl_tensor = dltensor
    managed_tensor.manager_ctx = None
    managed_tensor.deleter = no_op_deleter
    # Note: Must ensure that shape_array and managed_tensor are not garbage collected by Python,
    # A simple way is to attach them to the capsule object.
    # Call PyCapsule_New to create capsule
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = c_void_p
    PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, c_void_p]
    # Allocate managed_tensor on the heap (note that pointer returns a pointer)
    managed_tensor_ptr = pointer(managed_tensor)
    # The capsule name must be "dltensor", as required by the DLPack specification
    capsule_ptr = PyCapsule_New(managed_tensor_ptr, b"dltensor", None)
    # Convert capsule_ptr to Python object
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object).value
    # To prevent shape_array and managed_tensor from being collected, we attach them as attributes to the capsule
    capsule_wrapper = CapsuleWrapper(capsule, shape_array, managed_tensor)
    return capsule_wrapper


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
