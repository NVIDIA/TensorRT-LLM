# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared capture, validation and compilation boundaries for MQA TopK."""

from __future__ import annotations
import importlib.util
from typing import Any, Tuple
import torch
from torch.cuda import get_device_capability as get_compute_capability

try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:
    _raw_stream = None
_CUTE_DSL_AVAILABLE = (
    importlib.util.find_spec("cutlass") is not None
    and importlib.util.find_spec("cutlass.cute") is not None
)


def _validate_topk_device(device: torch.device) -> None:
    """Reject devices outside the supported architectures."""
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        raise ValueError("TopK requires SM100 or SM103")


if _CUTE_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    for _definition_module in (".kernels.topk_metadata",):
        importlib.import_module(_definition_module, __package__)
    del _definition_module
_STREAM_CACHE: dict[Tuple[int, int], Any] = {}


def _byte_range(tensor: torch.Tensor) -> Tuple[int, int]:
    """Return the occupied positive-stride byte interval for alias checks."""
    elements = 1
    if tensor.numel() != 0:
        elements += sum(
            (
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
        )
    begin = tensor.data_ptr()
    return (begin, begin + elements * tensor.element_size())


def _overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _device_ordinal(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _current_stream(device: torch.device):
    device_index = _device_ordinal(device)
    raw_stream = (
        _raw_stream(device_index)
        if _raw_stream is not None
        else torch.cuda.current_stream(device_index).cuda_stream
    )
    key = (device_index, raw_stream)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = cuda.CUstream(raw_stream)
        _STREAM_CACHE[key] = stream
    return stream


def _compile_for_device(device: torch.device, kernel, *args):
    """Keep build context and generated architecture aligned with the tensor key."""
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        options = f"--gpu-arch sm_{major}{minor}a"
        max_registers = getattr(kernel, "ptxas_max_registers", None)
        if max_registers is not None:
            options += f" --ptxas-options=--maxrregcount={max_registers}"
        compiled = cute.compile(kernel, *args, options=options)
        compiled.to(_device_ordinal(device))
        return compiled


def _require_cute() -> None:
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("MQA TopK requires nvidia-cutlass-dsl")
