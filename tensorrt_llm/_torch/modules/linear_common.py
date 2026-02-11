# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tensor-parallel types and weight-loading utilities with minimal dependencies.

This module exists to avoid circular imports: it does not depend on
tensorrt_llm._torch.custom_ops or tensorrt_llm._torch.modules.fused_moe.
Consumers that need only TensorParallelMode and load_weight_shard (e.g. fused_moe
quantization) should import from here; linear.py re-exports these for backward
compatibility.
"""

from __future__ import annotations

import enum
import math
from typing import Optional, Union

import torch

from tensorrt_llm._utils import is_device_integrated
from tensorrt_llm.logger import logger


class TensorParallelMode(str, enum.Enum):
    COLUMN = "column"
    ROW = "row"

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0

    # Helper to shard the corresponding per-channel activation scales
    # Which shard along the dimension orthogonal to the weights
    @classmethod
    def flip(cls, mode):
        return cls.ROW if mode == cls.COLUMN else cls.COLUMN


def load_weight_shard(
    weight,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    tensor_parallel_mode: Optional[TensorParallelMode] = None,
    device: torch.device = torch.device("cpu"),
    return_slice_indices: bool = False,
) -> torch.Tensor:
    # Skip device transfers on integrated GPUs to conserve shared memory
    if weight.device.type != device.type and is_device_integrated():
        # For integrated GPU systems (e.g., DGX Spark), CPU and GPU share limited physical memory.
        # Avoiding device transfers reduces memory consumption and unnecessary data copies,
        # enabling support for larger models on memory-constrained systems.
        logger.warning_once(
            f"[load_weight_shard] Skipping device transfer from {weight.device} to {device} "
            "on integrated GPU to conserve shared memory.",
            key="load_weight_shard_skip_device_transfer_with_integrated_gpu",
        )
        device = weight.device
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor, indices: list[slice] | None = None):
            if indices is None:
                # Avoid unnecessary copy
                result = (tensor.to(device), [slice(d) for d in tensor.shape])
            else:
                result = (tensor[indices].to(device), indices)
            return result if return_slice_indices else result[0]

    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)
        ):
            return tensor[indices].to(device)
    else:
        raise ValueError(f"unsupported weight type: {type(weight)}")
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(d) for d in tensor_shape]
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))
