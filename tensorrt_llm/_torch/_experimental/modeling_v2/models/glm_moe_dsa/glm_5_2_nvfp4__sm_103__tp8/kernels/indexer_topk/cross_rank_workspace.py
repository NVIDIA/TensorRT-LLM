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
"""Exchange workspace of the cross_rank_topk context-partitioned indexer top-k.

cross_rank_topk owns one multicast workspace per tensor-parallel group, allocated with the
native ``McastGPUBuffer`` (the class behind the MNNVL all-reduce workspaces)
and used by nothing else. Its layout and Lamport protocol are defined by the
``glm_moe_dsa_cross_rank_topk`` kernel (``crossRankTopk.h``): every 32-bit word starts as
``0xFFFFFFFF`` and the kernel rotates three buffers with a device epoch held
in a separate per-rank state tensor.

Creation is collective over the group and only happens outside CUDA-graph
capture. Each rank first runs the caller's host-side preparation (cross_rank_topk
compiles its logits kernel there, so compile-time skew between ranks is
absorbed by the first agreement rather than by the first exchange), then the
ranks agree on the capability (every rank must have the compiled op, a
supported world size, MPI, a successful preparation and none of the rank-local
reasons of ``cross_rank_policy.rank_local_reason``: SM version, CuTe DSL runtime and
the kernel's local width limit), allocate, and agree
on success; if any step fails on any rank, every rank records the failure and
keeps the native path, so the collective count and order stay identical.
"""

import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

# Indices of torch.ops.trtllm.glm_moe_dsa_cross_rank_topk_workspace_info().
_INFO_WORKSPACE_WORDS = 0
_INFO_TOPK = 1
_INFO_MAX_RANKS = 2
_INFO_MAX_ROWS = 3
_INFO_PAGE_TOKENS = 4
_INFO_MAX_LOCAL_CAPACITY = 7
_INFO_STATE_WORDS = 8
_INFO_TIMEOUT_MS = 10


@dataclass
class CpWorkspace:
    """One rank's view of the cross_rank_topk exchange workspace.

    Attributes:
        handle: The native ``McastGPUBuffer`` (keeps the mapping alive).
        mc: Multicast int32 view of the whole workspace.
        uc: This rank's unicast int32 view of the whole workspace.
        state: ``[4]`` int32 device state (epoch = exchange launches, arrivals,
            second rounds taken, error word written before a timeout abort).
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


def op_info(world_size: int = 8) -> Optional[list]:
    """Constants of the compiled kernel for a group size, or None when the op is not built."""
    try:
        return list(torch.ops.trtllm.glm_moe_dsa_cross_rank_topk_workspace_info(world_size))
    except (AttributeError, RuntimeError):
        return None


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
    order (it is called from the same layer of the same forward on every
    rank). Returns None when the workspace is unavailable on any rank, and
    during CUDA-graph capture when it was not created before.

    Args:
        mapping: The rank's ``Mapping``.
        prepare: Optional host-side preparation run on every rank before the
            capability agreement of the creation (not called when the
            workspace already exists); an exception counts as a local
            failure that every rank agrees on.
        local_reason: Why this rank's build or device cannot run cross_rank_topk
            (``cross_rank_policy.rank_local_reason``; "" when it can). Part of the
            creation's capability agreement, so it keeps every rank native.
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
            logger.info(f"[glm_moe_dsa] cross_rank_topk/workspace: native fallback ({reason})")
        else:
            _WORKSPACES[key] = ws
            info = op_info(ws.world) or []
            timeout = info[_INFO_TIMEOUT_MS] if len(info) > _INFO_TIMEOUT_MS else "n/a"
            logger.info(
                f"[glm_moe_dsa] cross_rank_topk/workspace: created ({ws.uc.numel() * 4} bytes, "
                f"rank {ws.rank}/{ws.world}, exchange wait timeout {timeout} ms)"
            )
        return ws


def _local_capability_reason(mapping, info, prepare) -> str:
    """This rank's op, group size and preparation check ("" when capable)."""
    if info is None:
        return "glm_moe_dsa_cross_rank_topk op is not built"
    if mapping.tp_size > info[_INFO_MAX_RANKS]:
        return f"tp_size {mapping.tp_size} > {info[_INFO_MAX_RANKS]}"
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

    words = info[_INFO_WORKSPACE_WORDS]
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
                info[_INFO_STATE_WORDS], dtype=torch.int32, device=torch.device("cuda", device)
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
