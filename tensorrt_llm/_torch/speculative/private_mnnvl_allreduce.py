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
"""An MNNVL all-reduce with its own Lamport workspace.

``MNNVLAllReduce`` instances of one mapping share a single workspace (buffers
plus the flags that sequence them), so two of its all-reduces must not be in
flight at once -- they would interleave in the same Lamport buffers. A
reduction that runs on a side stream concurrently with the main stream's
all-reduces needs its own workspace; this class allocates one the same way
``get_or_scale_allreduce_mnnvl_workspace`` does and calls the same fused
kernel, so its results are those of ``MNNVLAllReduce`` for the same inputs.
"""

import math
import os

import torch

from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.mapping import Mapping

from ..distributed.ops import AllReduceFusionOp, MNNVLAllReduce

_NUM_LAMPORT_BUFFERS = 3


class PrivateMNNVLAllReduce:
    """``MNNVLAllReduce`` over a private workspace (for a side stream).

    Construction is collective over the mapping's TP group.
    """

    def __init__(self, mapping: Mapping, dtype: torch.dtype, max_tokens: int, hidden: int):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        self.mapping = mapping
        self.dtype = dtype
        elem = torch.tensor([], dtype=dtype).element_size()
        need = MNNVLAllReduce.get_required_workspace_size(
            max_tokens, hidden, mapping.tp_size, dtype
        )
        self.buffer_size_bytes = math.ceil(max(need, 1) / (8 * 1024 * 1024)) * (8 * 1024 * 1024)
        self.max_tokens = max_tokens
        self.hidden = hidden
        self._elem = elem
        workspace_bytes = _NUM_LAMPORT_BUFFERS * self.buffer_size_bytes
        comm = mpi_comm().Split(
            int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank), mapping.tp_rank
        )
        force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
        self._handle = McastGPUBuffer(
            workspace_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            mapping.local_rank,
            force_mn or mapping.is_multi_node(),
            comm.py2f(),
        )
        buffer = self._handle.get_uc_buffer(
            mapping.tp_rank, (workspace_bytes // torch.float32.itemsize,), torch.float32, 0
        )
        buffer.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._comm = comm
        self._buffer_base = buffer.view(dtype).view(_NUM_LAMPORT_BUFFERS, -1)
        # [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count]
        self._buffer_flags = torch.tensor(
            [0, 2, self.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

    def fits(self, num_tokens: int, hidden: int) -> bool:
        need = MNNVLAllReduce.get_required_workspace_size(
            num_tokens, hidden, self.mapping.tp_size, self.dtype
        )
        return need <= self.buffer_size_bytes

    def rms_norm(self, partial: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        """``RMSNorm(all_reduce(partial))``, as ``MNNVLAllReduce`` with ``RMS_NORM``."""
        assert partial.dtype == self.dtype and self.fits(
            partial.numel() // partial.shape[-1], partial.shape[-1]
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            partial,
            gamma,
            None,
            eps,
            self._buffer_base,
            self._buffer_flags,
            True,
            None,
            int(AllReduceFusionOp.RMS_NORM),
        )
        return outputs[0]
