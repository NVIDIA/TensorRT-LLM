# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading
from typing import Optional

import torch


class ContextProgress:
    """Tracks per-layer completion of context phase for overlapping compute
    and KV cache transfer in Chunked Pipeline Parallelism.

    Mirrors the C++ ContextProgress class in
    cpp/include/tensorrt_llm/batch_manager/contextProgress.h.

    Each decoder layer records a CUDA event after its forward pass completes.
    The KV cache transceiver can then wait on individual layer events to begin
    transferring that layer's KV cache before the entire forward pass finishes.
    """

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}")
        self._num_layers = num_layers
        self._events = [
            torch.cuda.Event(enable_timing=False)
            for _ in range(num_layers)
        ]
        self._recorded = [False] * num_layers
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def record_event(self,
                     layer_idx: int,
                     stream: Optional[torch.cuda.Stream] = None):
        """Record completion of layer_idx on the given CUDA stream.

        Args:
            layer_idx: Global layer index (0-based).
            stream: CUDA stream to record on. If None, uses current stream.

        Raises:
            IndexError: If layer_idx is out of range.
            RuntimeError: If the previous layer has not been recorded yet,
                or if this layer has already been recorded.
        """
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers})")
        if layer_idx > 0 and not self._recorded[layer_idx - 1]:
            raise RuntimeError(
                f"Layer {layer_idx - 1} has not been recorded yet")
        if self._recorded[layer_idx]:
            raise RuntimeError(
                f"Layer {layer_idx} has already been recorded")

        if stream is not None:
            self._events[layer_idx].record(stream)
        else:
            self._events[layer_idx].record()

        with self._condition:
            self._recorded[layer_idx] = True
            self._condition.notify_all()

    def wait(self, layer_idx: int,
             stream: Optional[torch.cuda.Stream] = None):
        """Block until layer_idx computation has completed.

        If stream is provided, makes that stream wait on the layer's event
        (non-blocking on CPU). If stream is None, synchronizes the event
        on the CPU thread.

        Args:
            layer_idx: Global layer index to wait for.
            stream: Optional stream that should wait for the event.
        """
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers})")

        with self._condition:
            while not self._recorded[layer_idx]:
                self._condition.wait(timeout=0.01)

        if stream is not None:
            stream.wait_event(self._events[layer_idx])
        else:
            self._events[layer_idx].synchronize()

    def is_recorded(self, layer_idx: int) -> bool:
        """Check if the event for layer_idx has been recorded."""
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers})")
        return self._recorded[layer_idx]

    def reset(self):
        """Reset all recorded flags for reuse with a new request."""
        with self._condition:
            self._recorded = [False] * self._num_layers
