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
"""
uGPU Runtime: stream, mempool, event, and synchronization management.

Wraps the low-level ugpu_utils.py functions with a cleaner interface.
Module layer code should use UgpuRuntime instead of calling ugpu_utils directly.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from tensorrt_llm._torch.ugpu.policy import PartitionPlan
from tensorrt_llm._torch.ugpu_utils import (
    end_for_all_ugpu,
    get_ugpu_compute_sm_counts,
    get_ugpu_mempool,
    get_ugpu_stream,
    initialize_ugpu_resources,
    optional_ugpu_mem_pool,
    start_for_all_ugpu,
    ugpu_device,
)


class UgpuRuntime:
    """Manages uGPU execution resources.

    Provides a clean boundary between module-level code and low-level
    uGPU resource management. Thread-local ugpu context is only set
    within partition_context(), never by module code directly.
    """

    def __init__(self, num_partitions: int = 2):
        self.num_partitions = num_partitions

    def partition_stream(self, partition_id: int) -> torch.cuda.Stream:
        """Get the CUDA stream for a specific partition."""
        return get_ugpu_stream(partition_id)

    def partition_mempool(self, partition_id: int) -> torch.cuda.MemPool:
        """Get the memory pool for a specific partition."""
        return get_ugpu_mempool(partition_id)

    def topology_identity(self) -> tuple[tuple[int, int], ...]:
        """Return a stable identity for the actual public CUDA compute split."""
        initialize_ugpu_resources()
        return tuple(
            get_ugpu_compute_sm_counts(partition_id) or (0, 0)
            for partition_id in range(self.num_partitions)
        )

    @contextmanager
    def partition_context(self, partition_id: int):
        """Set thread-local uGPU context and stream for kernel dispatch.

        This is the ONLY place where thread-local ugpu_id is set during
        forward execution. Kernel runners read it via get_current_ugpu().
        """
        with ugpu_device(partition_id):
            with torch.cuda.stream(self.partition_stream(partition_id)):
                yield

    @contextmanager
    def partition_weight_context(self, partition_id: int):
        """Set thread-local uGPU context and memory pool for weight operations.

        Used during create_weights and load_weights to allocate on the
        correct uGPU partition's memory.
        """
        with ugpu_device(partition_id):
            with optional_ugpu_mem_pool():
                yield

    def fork(self):
        """Record event on current stream, then wait on all partition streams.

        Call before launching partition work.
        """
        start_for_all_ugpu()

    def join(self):
        """Record events on partition streams, then wait on current stream.

        Call after all partition work is launched.
        """
        end_for_all_ugpu()

    def prepare_for_capture(self, plan: PartitionPlan):
        """Pre-initialize all resources before CUDA Graph capture.

        Must be called before any graph capture to ensure streams,
        mempools, and allocators are ready.
        """
        initialize_ugpu_resources()
