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
"""Exchange workspace of the cross_rank_topk_global_emit context-partitioned top-k (with the bridge outputs).

cross_rank_topk_global_emit owns one multicast workspace per tensor-parallel group, allocated with
the native ``McastGPUBuffer`` and used by nothing else (cross_rank_topk's and
cross_rank_topk_cluster_merge's workspaces are separate ones). Its layout and Lamport protocol are
cross_rank_topk_cluster_merge's, defined by the ``glm_moe_dsa_cross_rank_topk_global_emit`` kernel
(``crossRankTopkGlobalEmit.h``): every 32-bit word starts as ``0xFFFFFFFF`` and the kernel rotates
three buffers with per-CTA epoch words held in a separate per-rank state tensor.

Creation is collective over the group and only happens outside CUDA-graph
capture, exactly as cross_rank_topk's (``indexer_topk/cross_rank_workspace.py``): each rank runs the
caller's host-side preparation, the ranks agree on the capability (compiled
op, supported world size, MPI, successful preparation, no rank-local reason,
and enough co-resident clusters for the largest row count), allocate, and
agree on success; if any step fails on any rank, every rank records the
failure, so the collective count and order stay identical.
"""

import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

# Indices of torch.ops.trtllm.glm_moe_dsa_cross_rank_topk_global_emit_workspace_info().
INFO_WORKSPACE_WORDS = 0
INFO_TOPK = 1
INFO_MAX_RANKS = 2
INFO_MAX_ROWS = 3
INFO_PAGE_TOKENS = 4
INFO_MAX_LOCAL_CAPACITY = 7
INFO_STATE_WORDS = 8
INFO_TIMEOUT_MS = 10
INFO_MAX_ACTIVE_CLUSTERS = 11
INFO_MAX_GROUP_SIZE = 12
# First epoch word of the state (state[EPOCH_BASE + row * 8 + cta]).
EPOCH_BASE = 4


@dataclass
class CpWorkspace:
    """One rank's view of the cross_rank_topk_global_emit exchange workspace.

    Attributes:
        handle: The native ``McastGPUBuffer`` (keeps the mapping alive).
        mc: Multicast int32 view of the whole workspace.
        uc: This rank's unicast int32 view of the whole workspace.
        state: int32 device state: [2] second rounds taken, [3] error word
            written before a timeout abort, then one epoch word (exchange
            launches) per (row, CTA) from ``EPOCH_BASE``.
        rank: Rank in the group.
        world: Group size.
        comm: The group communicator the handle was created with (kept alive).
    """

    handle: object
    mc: torch.Tensor
    uc: torch.Tensor
    state: torch.Tensor
    rank: int
    world: int
    comm: object = None

    @property
    def mc_ptr(self) -> int:
        return self.mc.data_ptr()

    @property
    def uc_ptr(self) -> int:
        return self.uc.data_ptr()


_LOCK = threading.Lock()
_WORKSPACES: Dict[Tuple, CpWorkspace] = {}
_FAILED: Dict[Tuple, str] = {}


_OP_INFO: Dict[Tuple[int, int], list] = {}


def op_info(world_size: int = 8) -> Optional[list]:
    """Constants of the compiled kernel for a group size, or None when the op is not built.

    Queried once per group size and current device (the region calls it on
    every eager forward); a missing op is not cached, so an op loaded later
    is seen.
    """
    key = (world_size, torch.cuda.current_device() if torch.cuda.is_available() else -1)
    info = _OP_INFO.get(key)
    if info is None:
        try:
            info = list(
                torch.ops.trtllm.glm_moe_dsa_cross_rank_topk_global_emit_workspace_info(world_size)
            )
        except (AttributeError, RuntimeError):
            return None
        _OP_INFO[key] = info
    return list(info)


def cluster_reason(info: Optional[list]) -> str:
    """Why this device cannot keep every row's cluster resident ("" when it can)."""
    if info is None or len(info) <= INFO_MAX_ACTIVE_CLUSTERS:
        return "glm_moe_dsa_cross_rank_topk_global_emit op is not built"
    clusters, rows = info[INFO_MAX_ACTIVE_CLUSTERS], info[INFO_MAX_ROWS]
    if clusters < rows:
        return f"{clusters} co-resident 8-CTA clusters < {rows} rows"
    return ""


def _key(mapping) -> Tuple:
    return (mapping.world_size, mapping.rank, mapping.tp_size, mapping.pp_size, mapping.cp_size)


def failure_reason(mapping) -> Optional[str]:
    """Reason recorded when this group's workspace could not be created."""
    return _FAILED.get(_key(mapping))


def get_workspace(
    mapping, prepare: Optional[Callable[[], None]] = None, local_reason: str = ""
) -> Optional[CpWorkspace]:
    """Return the group's workspace, creating it collectively on first use.

    Must be reached by every rank of the tensor-parallel group in the same
    order. Returns None when the workspace is unavailable on any rank, and
    during CUDA-graph capture when it was not created before.

    Args:
        mapping: The rank's ``Mapping``.
        prepare: Optional host-side preparation run on every rank before the
            capability agreement of the creation; an exception counts as a
            local failure that every rank agrees on.
        local_reason: Why this rank's build or device cannot run cross_rank_topk_global_emit (""
        when
            it can); part of the creation's capability agreement.
    """
    key = _key(mapping)
    ws = _WORKSPACES.get(key)
    if ws is not None or key in _FAILED:
        return ws
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        if key in _WORKSPACES or key in _FAILED:
            return _WORKSPACES.get(key)
        ws, reason = _create(mapping, prepare, local_reason)
        if ws is None:
            _FAILED[key] = reason
            logger.info(
                f"[glm_moe_dsa] cross_rank_topk_global_emit/workspace: not created ({reason})"
            )
        else:
            _WORKSPACES[key] = ws
            info = op_info(ws.world) or []
            timeout = info[INFO_TIMEOUT_MS] if len(info) > INFO_TIMEOUT_MS else "n/a"
            clusters = (
                info[INFO_MAX_ACTIVE_CLUSTERS] if len(info) > INFO_MAX_ACTIVE_CLUSTERS else "n/a"
            )
            logger.info(
                f"[glm_moe_dsa] cross_rank_topk_global_emit/workspace: created ({ws.uc.numel() * 4} bytes, "
                f"rank {ws.rank}/{ws.world}, exchange wait timeout {timeout} ms, "
                f"co-resident clusters {clusters})"
            )
        return ws


def _local_capability_reason(mapping, info, prepare) -> str:
    """This rank's op, group size, cluster residency and preparation check ("" when capable)."""
    if info is None:
        return "glm_moe_dsa_cross_rank_topk_global_emit op is not built"
    if mapping.tp_size > info[INFO_MAX_RANKS]:
        return f"tp_size {mapping.tp_size} > {info[INFO_MAX_RANKS]}"
    reason = cluster_reason(info)
    if reason:
        return reason
    if prepare is not None:
        try:
            prepare()
        except Exception as e:  # noqa: BLE001 - any local failure must be agreed before use
            return f"preparation failed ({type(e).__name__}: {e})"
    return ""


def _create(mapping, prepare=None, local_reason="") -> Tuple[Optional[CpWorkspace], str]:
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _mnnvl_device_index,
        _mnnvl_workspace_all_succeeded,
    )
    from tensorrt_llm._utils import mpi_disabled

    if mpi_disabled():
        # Without MPI every rank takes this branch before any collective.
        return None, "MPI is disabled"
    comm = _get_mnnvl_workspace_comm(mapping)
    info = op_info(mapping.tp_size)
    if not local_reason:
        local_reason = _local_capability_reason(mapping, info, prepare)
    if not _mnnvl_workspace_all_succeeded(comm, not local_reason):
        return None, local_reason or "capability not agreed by every rank"

    words = info[INFO_WORKSPACE_WORDS]
    error: Optional[Exception] = None
    ws: Optional[CpWorkspace] = None
    try:
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        use_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
        device = _mnnvl_device_index(mapping)
        handle = McastGPUBuffer(
            words * 4, mapping.tp_size, mapping.tp_rank, device, use_fabric, comm.py2f()
        )
        uc = handle.get_uc_buffer(mapping.tp_rank, [words], torch.int32, 0)
        mc = handle.get_mc_buffer([words], torch.int32, 0)
        with torch.inference_mode():
            uc.fill_(-1)
            state = torch.zeros(
                info[INFO_STATE_WORDS], dtype=torch.int32, device=torch.device("cuda", device)
            )
        torch.cuda.synchronize()
        ws = CpWorkspace(handle, mc, uc, state, mapping.tp_rank, mapping.tp_size, comm)
    except Exception as e:  # noqa: BLE001 - any local failure must be agreed before use
        error = e
    # Also fences the initialization: no rank sends before every rank cleared
    # its buffer.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        return None, f"allocation failed on a rank ({type(error).__name__ if error else 'peer'})"
    return ws, ""
