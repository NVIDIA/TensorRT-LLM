# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL Lamport workspace the two draft-step ops exchange through, and their shape constants.

``glm_draft_sample_glue`` and ``glm_draft_ehproj_table`` publish each rank's (index, value) packet (and, for the
table op, a token-table row and the rank's hidden-half outputs) through the multicast address of a dedicated
workspace and poll the local copy until every rank's slot has landed. The workspace has three Lamport buffers and
the nine 32-bit flags of the MNNVL all-reduce layout (current index, dirty index, bytes per buffer, dirty stages,
four byte counts to clear, an access counter); an unwritten word holds 0xFFFFFFFF rather than the all-reduce's -0.0.

Each op empties the buffer of its previous call with the threads of its current launch, so a workspace serves one
op and nothing else.
"""

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm, _mnnvl_device_index
from tensorrt_llm._utils import get_env_flag, torch_pybind11_abi
from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer
from tensorrt_llm.mapping import Mapping

#: Rows (requests) one draft-step launch serves (kDraftSampleGlueMaxRows).
MAX_ROWS = 8
#: Bytes of one exchanged (index, value) packet (kDraftSampleGluePacketBytes).
PACKET_BYTES = 16
#: Hidden size the draft-step kernels are built for (kDraftSampleGlueHidden).
HIDDEN_SIZE = 6144
#: Largest TP size of glm_draft_ehproj_table (kDraftEhprojTableMaxRanks).
EHPROJ_TABLE_MAX_RANKS = 16


def ehproj_table_supports_tp(tp_size: int) -> bool:
    """Whether glm_draft_ehproj_table serves ``tp_size`` ranks (draftEhprojTableSupportsTpSize).

    2 to 16 ranks whose shares of the hidden size are whole 8-column vectors.
    """
    return 2 <= tp_size <= EHPROJ_TABLE_MAX_RANKS and HIDDEN_SIZE % (8 * tp_size) == 0


def sample_glue_slot_bytes(tp_size: int) -> int:
    """Bytes one rank publishes per row in glm_draft_sample_glue: its (index, value) packet."""
    del tp_size
    return PACKET_BYTES


def ehproj_table_slot_bytes(tp_size: int) -> int:
    """Bytes one rank publishes per row in glm_draft_ehproj_table (draftEhprojTableSlotBytes).

    The packet, the BF16 token-table row of its candidate and its FP32 share of the hidden-half outputs.
    """
    return PACKET_BYTES + 2 * HIDDEN_SIZE + 4 * HIDDEN_SIZE // tp_size


def _all_succeeded(comm: Any, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank of ``comm`` (an MPI communicator or a ProcessGroup with a CPU backend)."""
    if isinstance(comm, torch.distributed.ProcessGroup):
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        work = comm._get_backend(torch.device("cpu")).allreduce([flag])
        work.wait()
        return int(flag.item()) == comm.size()
    return comm.allreduce(int(ok)) == comm.Get_size()


class DraftExchangeWorkspace:
    """One draft-step op's MNNVL workspace over a TP group.

    ``buffer`` is the local unicast view of the three Lamport buffers (int32, every word 0xFFFFFFFF) and ``flags``
    the nine uint32 Lamport flags. Construction is collective over the group and either succeeds on every rank or
    raises on every rank. Build it outside CUDA-graph capture, once per op; the captured graphs keep its addresses.
    """

    NUM_LAMPORT_BUFFERS = 3

    def __init__(
        self,
        buffer_size_bytes: int,
        group_size: int,
        group_rank: int,
        device_index: int,
        use_fabric_handle: bool,
        comm: Any,
    ):
        """Allocate ``3 * buffer_size_bytes`` of multicast memory over the group and reset it.

        ``comm`` is the group's MPI communicator or a ``torch.distributed.ProcessGroup`` with a CPU backend, ordered
        by group rank.
        """
        assert buffer_size_bytes > 0 and buffer_size_bytes % 16 == 0, buffer_size_bytes
        self.buffer_size_bytes = buffer_size_bytes
        self.group_size = group_size
        workspace_size_bytes = self.NUM_LAMPORT_BUFFERS * buffer_size_bytes
        error = None
        state = None
        try:
            if isinstance(comm, torch.distributed.ProcessGroup):
                handle = McastGPUBuffer(
                    workspace_size_bytes,
                    group_size,
                    group_rank,
                    device_index,
                    use_fabric_handle,
                    process_group=comm,
                    pybind11_abi=torch_pybind11_abi(),
                )
            else:
                handle = McastGPUBuffer(
                    workspace_size_bytes,
                    group_size,
                    group_rank,
                    device_index,
                    use_fabric_handle,
                    comm.py2f(),
                )
            buffer = handle.get_uc_buffer(group_rank, (workspace_size_bytes // 4,), torch.int32, 0)
            with torch.inference_mode():
                buffer.fill_(-1)
                flags = torch.tensor(
                    [0, 2, buffer_size_bytes, 0, 0, 0, 0, 0, 0],
                    dtype=torch.uint32,
                    device=torch.device("cuda", device_index),
                )
            torch.cuda.synchronize()
            state = (handle, buffer, flags)
        except Exception as e:
            # Every ordinary setup failure must reach the group vote.
            error = e
        if not _all_succeeded(comm, error is None):
            raise RuntimeError(
                "draft-step MNNVL workspace construction failed on at least one rank"
            ) from error
        # The handle keeps the multicast allocation alive; the communicator outlives it.
        self._handle, self.buffer, self.flags = state
        self._comm = comm

    @classmethod
    def for_mapping(cls, mapping: Mapping, buffer_size_bytes: int) -> "DraftExchangeWorkspace":
        """The workspace over ``mapping``'s TP group, with the MNNVL all-reduce workspace's communicator.

        Fabric handles across nodes or under ``TRTLLM_FORCE_MNNVL_AR``, NVLS handles within a node, as the MNNVL
        all-reduce workspace takes them.
        """
        use_fabric_handle = get_env_flag("TRTLLM_FORCE_MNNVL_AR", False) or mapping.is_multi_node()
        return cls(
            buffer_size_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            _mnnvl_device_index(mapping),
            use_fabric_handle,
            _get_mnnvl_workspace_comm(mapping),
        )

    @classmethod
    def for_sample_glue(cls, mapping: Mapping) -> "DraftExchangeWorkspace":
        """The workspace of glm_draft_sample_glue for up to MAX_ROWS rows."""
        slot_bytes = sample_glue_slot_bytes(mapping.tp_size)
        return cls.for_mapping(mapping, MAX_ROWS * mapping.tp_size * slot_bytes)

    @classmethod
    def for_ehproj_table(cls, mapping: Mapping) -> "DraftExchangeWorkspace":
        """The workspace of glm_draft_ehproj_table for up to MAX_ROWS rows."""
        slot_bytes = ehproj_table_slot_bytes(mapping.tp_size)
        return cls.for_mapping(mapping, MAX_ROWS * mapping.tp_size * slot_bytes)
