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

"""Engine-owned NCCL-window storage for zero-copy FP4 GEMM outputs."""

import os
from typing import Dict, List, Tuple

import torch
from torch import nn

from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_BufferSpec = Tuple[torch.device, torch.dtype, int]
_Registration = Tuple[nn.Module, torch.Tensor, int, torch.dtype]
_PreallocationKey = Tuple[Tuple[int, ...], str, int, torch.dtype, int]


class NCCLWindowTensorPool:
    """Own one persistent output buffer for each output signature.

    The current FP4 application consumes each output before another eligible
    GEMM with the same signature runs. That serialized lifetime lets all
    eligible modules and CUDA graphs share one address. Supporting concurrent
    producers will require an explicit workspace/lease contract instead.
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.capacity = 0
        self._registrations: List[_Registration] = []
        self._buffers: List[torch.Tensor] = []
        self._preallocation_keys: Tuple[_PreallocationKey, ...] = ()

        requested = os.environ.get("TLLM_NCCL_WINDOW_TENSOR_POOL", "0") == "1"
        zero_copy = os.environ.get("TLLM_NCCL_SYMMETRIC_ZERO_COPY", "1") == "1"
        self.enabled = requested and zero_copy and mapping.tp_size > 1
        self._log_stats = os.environ.get(
            "TLLM_NCCL_WINDOW_TENSOR_POOL_LOG_STATS", "0") == "1"

    def register(self, module: nn.Module, like: torch.Tensor,
                 output_width: int, dtype: torch.dtype) -> None:
        """Register a module buffer to bind when storage is reserved."""
        if self.enabled:
            self._registrations.append(
                (module, like, int(output_width), dtype))

    def disable(self) -> None:
        self.clear()
        self.enabled = False

    def reserve(self, capacity: int) -> None:
        """Allocate storage before tracing or CUDA-graph capture begins."""
        if not self.enabled or not self._registrations:
            return

        capacity = int(capacity)
        if capacity <= 0:
            return
        if self.capacity:
            if capacity != self.capacity:
                raise RuntimeError(
                    "cannot resize NCCL-window output storage after reservation"
                )
            return

        representatives: Dict[_BufferSpec, torch.Tensor] = {}
        for _, like, output_width, dtype in self._registrations:
            representatives.setdefault((like.device, dtype, output_width),
                                       like)

        buffers: Dict[_BufferSpec, torch.Tensor] = {}
        ordered_specs = sorted(
            representatives,
            key=lambda spec: (spec[0].type, spec[0].index or -1,
                              str(spec[1]), spec[2]),
        )
        for device, dtype, output_width in ordered_specs:
            spec = (device, dtype, output_width)
            output = self._allocate(representatives[spec], capacity,
                                    output_width, dtype)
            if output is not None:
                buffers[spec] = output

        # Keep allocation calls rank-symmetric and enable the optimization only
        # when every signature received NCCL-window storage.
        if len(buffers) != len(representatives):
            return

        for output in buffers.values():
            self._preallocate_spare(output)

        for module, like, output_width, dtype in self._registrations:
            module._nccl_window_output = buffers[(like.device, dtype,
                                                  output_width)]

        self._buffers = list(buffers.values())
        self.capacity = capacity

        # The retained GEMM buffer and one cached all-reduce result buffer
        # replace AllReduceRunner's much larger model-maximum preallocation.
        self._preallocation_keys = self._mark_group_preallocated()
        if self._log_stats:
            logger.info(
                "NCCL window Tensor pool reserved: "
                f"capacity={self.capacity}, "
                f"registrations={len(self._registrations)}, "
                f"buffers={len(self._buffers)}")

    def _allocate(self, like: torch.Tensor, capacity: int, output_width: int,
                  dtype: torch.dtype) -> torch.Tensor | None:
        output, actual_kind = torch.ops.trtllm.allocate_output(
            like,
            int(BufferKind.NCCL_WINDOW),
            self.mapping.tp_group,
            [capacity, output_width],
            dtype,
        )
        if int(actual_kind) != int(BufferKind.NCCL_WINDOW):
            return None
        return output

    def _preallocate_spare(self, output: torch.Tensor) -> None:
        torch.ops.trtllm.preallocate_nccl_window_buffer(
            output, self.mapping.tp_group, 1)

    def _mark_group_preallocated(self) -> Tuple[_PreallocationKey, ...]:
        from ..custom_ops.torch_custom_ops import AllReduceRunner

        return AllReduceRunner.mark_nccl_window_preallocated(
            self.mapping.tp_group, self._buffers)

    def _unmark_group_preallocated(
            self, keys: Tuple[_PreallocationKey, ...]) -> None:
        from ..custom_ops.torch_custom_ops import AllReduceRunner

        AllReduceRunner.unmark_nccl_window_preallocated(keys)

    def clear(self) -> None:
        """Drop module references before releasing retained Tensor wrappers."""
        if self._log_stats and self.enabled:
            logger.info(
                "NCCL window Tensor pool cleared: "
                f"capacity={self.capacity}, "
                f"registrations={len(self._registrations)}, "
                f"buffers={len(self._buffers)}")
        keys = self._preallocation_keys
        self._preallocation_keys = ()
        if keys:
            self._unmark_group_preallocated(keys)
        for module, _, _, _ in self._registrations:
            module._nccl_window_output = None
        self._buffers.clear()
        self._registrations.clear()
        self.capacity = 0
