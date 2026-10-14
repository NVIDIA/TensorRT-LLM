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
"""Exchange sharded draft-argmax pairs through a private MNNVL workspace."""

from __future__ import annotations

import functools
import os
import threading
from typing import ClassVar

import torch

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    MNNVLAllReduce,
    _get_mnnvl_workspace_comm,
    _initialize_allreduce_mnnvl_protocol,
    _make_mnnvl_mcast_buffer,
    _mnnvl_device_index,
    _mnnvl_workspace_all_succeeded,
    _MnnvlWorkspace,
    allgather,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


@functools.lru_cache(maxsize=1)
def _mailbox_enabled() -> bool:
    return os.environ.get("TRTLLM_MTP_TAIL_TRIM", "1").strip() != "0"


if triton is not None:

    @triton.jit
    def _pack_mailbox_kernel(pairs, mailbox, slot, WIDTH: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, WIDTH)
        values = tl.load(
            pairs + row * 2 + cols - slot, mask=(cols >= slot) & (cols < slot + 2), other=0.0
        )
        tl.store(mailbox + row * WIDTH + cols, values)


class _DraftArgmaxMailbox:
    """Fixed-size Lamport workspace retained for eager and captured draft steps."""

    WIDTH = 128
    _instances: ClassVar[dict[Mapping, _DraftArgmaxMailbox | None]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, mapping: Mapping) -> None:
        self.mapping = mapping
        self.max_rows = _MNNVL_ONE_SHOT_THRESHOLD_BYTES // (
            self.WIDTH * torch.float32.itemsize * mapping.tp_size
        )
        buffer_size_bytes = self.max_rows * self.WIDTH * torch.float32.itemsize * mapping.tp_size
        comm = _get_mnnvl_workspace_comm(mapping)
        workspace: _MnnvlWorkspace | None = None
        local_error: Exception | None = None
        try:
            use_fabric_handle = (
                os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
            )
            handle = _make_mnnvl_mcast_buffer(
                comm, 3 * buffer_size_bytes, mapping, use_fabric_handle
            )
            workspace = {
                "handle": handle,
                "uc_buffer": handle.get_uc_buffer(
                    mapping.tp_rank,
                    (3 * buffer_size_bytes // torch.float32.itemsize,),
                    torch.float32,
                    0,
                ),
                "buffer_flags": torch.zeros(
                    9, dtype=torch.uint32, device=torch.device("cuda", _mnnvl_device_index(mapping))
                ),
                "buffer_size_bytes": buffer_size_bytes,
                "comm": comm,
                "captured": False,
            }
        except Exception as error:  # noqa: BLE001 - converge allocation errors across the TP group
            local_error = error
        if not _mnnvl_workspace_all_succeeded(comm, local_error is None):
            if not mpi_disabled():
                comm.Free()
            raise RuntimeError(
                "Draft argmax mailbox allocation failed on at least one rank"
            ) from local_error
        assert workspace is not None
        try:
            _initialize_allreduce_mnnvl_protocol(workspace)
        except RuntimeError:
            if not mpi_disabled():
                comm.Free()
            raise
        # Keep the handle, communicator and flags alive for every graph that uses them.
        self._workspace = workspace
        self._buffer_base = workspace["uc_buffer"].view(3, -1)

    @classmethod
    def get(cls, mapping: Mapping) -> _DraftArgmaxMailbox | None:
        if mapping in cls._instances:
            return cls._instances[mapping]
        if torch.cuda.is_current_stream_capturing():
            return None
        with cls._lock:
            if mapping in cls._instances:
                return cls._instances[mapping]
            mailbox = None
            if (
                1 < mapping.tp_size <= cls.WIDTH // 2
                and mapping.world_size == mapping.tp_size
                and not mapping.has_cp()
                and not mapping.enable_attention_dp
                and MNNVLAllReduce.is_mnnvl(mapping, torch.float32)
            ):
                try:
                    mailbox = cls(mapping)
                except RuntimeError as error:
                    logger.warning(
                        "Draft argmax mailbox unavailable; using NCCL allgather: %s", error
                    )
            cls._instances[mapping] = mailbox
            return mailbox

    def exchange(self, combined: torch.Tensor) -> torch.Tensor:
        rows = combined.shape[0]
        mailbox = torch.empty((rows, self.WIDTH), dtype=torch.float32, device=combined.device)
        _pack_mailbox_kernel[(rows,)](combined, mailbox, 2 * self.mapping.tp_rank, self.WIDTH)
        # Each rank writes only its pair; SUM therefore preserves all finite values
        # exactly. NaN payloads and signed zero do not affect the subsequent argmax.
        result = torch.ops.trtllm.mnnvl_fusion_allreduce(
            mailbox,
            None,
            None,
            1e-6,
            self._buffer_base,
            self._workspace["buffer_flags"],
            False,
            None,
            int(AllReduceFusionOp.NONE),
        )[0]
        return result[:, : 2 * self.mapping.tp_size]


def gather_draft_argmax_pairs(
    combined: torch.Tensor, mapping: Mapping, *, allow_mnnvl: bool = True
) -> torch.Tensor:
    """Gather fp32 ``[rows, 2]`` pairs to rank-major ``[rows, 2 * tp_size]``.

    A small pure-TP multi-node MNNVL exchange uses a dedicated one-shot workspace.
    Other mappings, oversized batches and first calls during capture use NCCL.
    Sleep-enabled engines set ``allow_mnnvl=False`` to keep the NCCL lifecycle.
    """
    if (
        allow_mnnvl
        and triton is not None
        and _mailbox_enabled()
        and combined.is_cuda
        and combined.dtype == torch.float32
        and combined.ndim == 2
        and combined.shape[1] == 2
        and combined.shape[0] > 0
        and combined.is_contiguous()
        and combined.device.index == _mnnvl_device_index(mapping)
        and combined.shape[0] * _DraftArgmaxMailbox.WIDTH * torch.float32.itemsize * mapping.tp_size
        <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    ):
        mailbox = _DraftArgmaxMailbox.get(mapping)
        if mailbox is not None:
            return mailbox.exchange(combined)
    return allgather(combined, mapping, dim=-1)
