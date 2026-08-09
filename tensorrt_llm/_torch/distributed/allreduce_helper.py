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

import os
from typing import List, Tuple

import torch

from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
from tensorrt_llm.bindings.internal.runtime import (
    lamport_initialize,
    max_workspace_size_lowprecision,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


def force_all_reduce_deterministic():
    return (
        os.getenv("FORCE_DETERMINISTIC", "0") == "1"
        or os.getenv("FORCE_ALL_REDUCE_DETERMINISTIC", "0") == "1"
    )


class CustomAllReduceHelper:
    """
    Helper for custom all-reduce workspace/IPC buffer allocation and sizing,
    used by the PyTorch backend.
    """

    POINTERS_PER_RANK = 7
    POINTERS_OF_COUNTER = 3

    @staticmethod
    def max_workspace_size_auto(tp_size: int, support_deterministic=True) -> int:
        """Calculate workspace size for allreduce fusion kernel.

        The workspace is used for lamport buffers in the fusion kernel.
        Required size calculation:
        - Each GPU needs 3 sub-buffers (for triple buffering)
        - Each sub-buffer stores: max_num_tokens * hidden_size * dtype_size (bf16=2)
        - The lamport allocation multiplies by tp_size, so:
          lamport_size = 3 * size * tp_size (per GPU)

        Example: Llama 8B (hidden=4096), max_tokens=8192, bf16, TP=4
        - Data per sub-buffer: 8192 * 4096 * 2 = 64 MiB
        - Total lamport: 3 * 64MB * 4 = 768 MiB per GPU
        - Required 'size' parameter: 64 MiB (gets multiplied by tp_size in allocation)

        Default (67,108,864 = 64 MiB) supports:
        - Models up to hidden_size=4096 with max_num_tokens=8192
        - Or hidden_size=8192 with max_num_tokens=4096

        Override with TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE env var if needed for larger models.
        """
        if force_all_reduce_deterministic() and support_deterministic:
            workspace_size = os.getenv("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE", "1000000000")
            return int(workspace_size)

        # Allow override via environment variable for edge cases
        workspace_size_env = os.getenv("TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE")
        if workspace_size_env:
            size = int(workspace_size_env)
            logger.info(
                f"Using custom allreduce fusion workspace size: {size} bytes ({size / (1024**2):.1f} MiB)"
            )
            return size

        # Default: 64 MiB - supports most common model configurations
        # Increase via env var if you see CUDA illegal memory access errors with large models
        default_size = 67_108_864  # Exactly 64 MiB
        return default_size

    @staticmethod
    def max_workspace_size_lowprecision(tp_size: int) -> int:
        return max_workspace_size_lowprecision(tp_size)

    @staticmethod
    def initialize_lowprecision_buffers(workspace: "torch.tensor", tp_size: int) -> None:
        return torch.ops.trtllm.initialize_static_lowprecision_buffers(workspace, tp_size)

    @staticmethod
    def allocate_lowprecision_workspace(
        mapping: Mapping, size: int
    ) -> Tuple[List[IpcMemory], "torch.tensor"]:
        # Force pull mode and disable lamport when force deterministic is enabled, for reducing device memory usage.
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size
        ipc_buffers_ping = IpcMemory(mapping, ipc_buffers_size, is_p2p_supported)
        ipc_buffers_pong = IpcMemory(mapping, ipc_buffers_size, is_p2p_supported)
        ipc_barriers_in = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2, is_p2p_supported
        )
        ipc_barriers_out = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2, is_p2p_supported
        )
        buffers = [ipc_buffers_ping, ipc_buffers_pong, ipc_barriers_in, ipc_barriers_out]

        return buffers, torch.tensor(
            ipc_buffers_ping.serialize()
            + ipc_buffers_pong.serialize()
            + ipc_barriers_in.serialize()
            + ipc_barriers_out.serialize()
            + [0]
            + [0],
            dtype=torch.int64,
            device="cpu",
        )

    @staticmethod
    def allocate_allreduce_fusion_workspace(
        mapping: Mapping, size: int
    ) -> Tuple[List[IpcMemory], "torch.tensor"]:
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size * mapping.tp_size
        ipc_buffers = IpcMemory(mapping, ipc_buffers_size, is_p2p_supported)
        ipc_barriers = IpcMemory(mapping, 256 * mapping.tp_size, is_p2p_supported)
        lamport_buffers_size = size * mapping.tp_size
        lamport_buffers = IpcMemory(mapping, 3 * lamport_buffers_size, is_p2p_supported)
        if is_p2p_supported:
            lamport_initialize(
                lamport_buffers.local_ptr,
                3 * lamport_buffers_size,
            )
        # flag_buffer[0], atomic flag read counter
        # flag_buffer[1], non-lamport flag
        # flag_buffer[2], lamport flag
        flag_buffer = torch.tensor([0, 0, 0], dtype=torch.int, device="cuda")
        # layout_buffer[0], clear size for next lamport kernel
        # layout_buffer[1], triple buffer offset for lamport kernel
        layout_buffer = torch.tensor([0, lamport_buffers_size], dtype=torch.int64, device="cuda")

        buffers = [ipc_buffers, ipc_barriers, lamport_buffers, flag_buffer, layout_buffer]

        return buffers, torch.tensor(
            ipc_buffers.serialize()
            + ipc_barriers.serialize()
            + lamport_buffers.serialize()
            + [flag_buffer.data_ptr()]
            + [layout_buffer.data_ptr()],
            dtype=torch.int64,
            device="cuda",
        )
