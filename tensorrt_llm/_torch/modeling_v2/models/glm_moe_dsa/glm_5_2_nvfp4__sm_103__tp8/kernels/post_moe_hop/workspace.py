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
"""post_moe_hop-owned MNNVL workspace and the rank-agreed capability decision.

The workspace is separate from the native MNNVL all-reduce workspace: post_moe_hop keeps
its own multicast buffer, its own Lamport epoch and its own error word, so its
call sequence never interleaves with another all-reduce's protocol state. One
workspace serves every post_moe_hop of a TP group; hops run in the same order on
every rank, which keeps the per-rank epochs in lockstep.
"""

import threading
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping

from .kernels import MAX_CHUNKS, STATE_CALLS, STATE_ERROR_WORDS, STATE_TIMEOUT_POLLS, STATE_WORDS

_BF16_NEG_ZERO_BITS = -32768


def _workspace_comm(mapping: Mapping):
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

    return _get_mnnvl_workspace_comm(mapping)


def _all_ranks_agree(comm, local_ok: bool) -> bool:
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return _mnnvl_workspace_all_succeeded(comm, local_ok)


@dataclass(frozen=True)
class _WorkspaceKey:
    mapping: Mapping
    hidden_size: int
    max_rows: int


class PostMoeHopWorkspace:
    """Multicast Lamport buffers plus local partials/state for one TP group.

    Attributes:
        mc: bf16 ``[3 * max_rows * world * hidden]`` multicast view.
        uc: bf16 view of the same region through this rank's unicast address.
        partials: fp32 ``[3 * max_rows * MAX_CHUNKS]`` local row-sum exchange.
        state: int32 ``[STATE_WORDS]`` epoch / previous rows / arrivals / errors /
            calls / poll count at the latest timeout.
        layout_repairs: host-side count of post_moe_hop calls whose raw tuple needed an
            aligned copy (see ``PostMoeHop``); a CUDA graph capture counts
            once, its replays do not.
    """

    _instances: ClassVar[Dict[_WorkspaceKey, "PostMoeHopWorkspace"]] = {}
    _comms: ClassVar[Dict[Mapping, object]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, mapping: Mapping, hidden_size: int, max_rows: int, comm):
        from tensorrt_llm._torch.distributed.ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
        )

        self.mapping = mapping
        self.hidden_size = hidden_size
        self.max_rows = max_rows
        self.world = mapping.tp_size
        self.comm = comm
        self.lamport_elems = 3 * max_rows * self.world * hidden_size
        nbytes = self.lamport_elems * torch.bfloat16.itemsize
        self.handle = _make_mnnvl_mcast_buffer(comm, nbytes, mapping, mapping.is_multi_node())
        self.uc = self.handle.get_uc_buffer(
            mapping.tp_rank, (self.lamport_elems,), torch.bfloat16, 0
        )
        self.mc = self.handle.get_mc_buffer((self.lamport_elems,), torch.bfloat16, 0)
        device = torch.device("cuda", _mnnvl_device_index(mapping))
        self.partials = torch.full(
            (3 * max_rows * MAX_CHUNKS,), -0.0, dtype=torch.float32, device=device
        )
        self.state = torch.zeros((STATE_WORDS,), dtype=torch.int32, device=device)
        # The kernels access every buffer with 16-byte vectors.
        for name in ("uc", "mc", "partials", "state"):
            if getattr(self, name).data_ptr() % 16:
                raise RuntimeError(f"post_moe_hop workspace buffer {name} is not 16-byte aligned")
        self.layout_repairs = 0
        self._reset_local()

    @classmethod
    def comm_for(cls, mapping: Mapping):
        """Return the cached TP-group communicator (collective on first use)."""
        with cls._lock:
            comm = cls._comms.get(mapping)
            if comm is None:
                comm = _workspace_comm(mapping)
                cls._comms[mapping] = comm
            return comm

    @classmethod
    def instances(cls) -> List["PostMoeHopWorkspace"]:
        """Every workspace created in this process (one per TP group and shape)."""
        with cls._lock:
            return [ws for ws in cls._instances.values() if ws is not None]

    @classmethod
    def get_or_create(
        cls, mapping: Mapping, hidden_size: int, max_rows: int
    ) -> Tuple[Optional["PostMoeHopWorkspace"], str]:
        """Collectively allocate (once) and return the workspace, or ``None``.

        Every rank of the TP group must call this in the same order. The
        allocation either succeeds on all ranks or is discarded on all ranks.
        """
        key = _WorkspaceKey(mapping, hidden_size, max_rows)
        with cls._lock:
            existing = cls._instances.get(key)
        if existing is not None:
            return existing, ""
        comm = cls.comm_for(mapping)
        workspace = None
        error: Optional[Exception] = None
        try:
            workspace = cls(mapping, hidden_size, max_rows, comm)
        except Exception as exc:  # noqa: BLE001 - converged across ranks below
            error = exc
        if not _all_ranks_agree(comm, error is None):
            return None, f"workspace allocation failed on some rank ({error!r})"
        with cls._lock:
            cls._instances[key] = workspace
        return workspace, ""

    def _reset_local(self) -> None:
        with torch.inference_mode():
            self.uc.view(torch.int16).fill_(_BF16_NEG_ZERO_BITS)
            self.partials.fill_(-0.0)
            self.state.zero_()
        torch.cuda.synchronize()

    def reset(self) -> bool:
        """Collectively return every rank to epoch 0 with empty buffers.

        Needed only after an error bit was raised, because a timed-out call
        leaves the ranks' epochs out of step. Returns True when every rank reset.
        """
        ok = True
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            ok = False
        # Every rank's in-flight multicast stores have landed once all ranks
        # drained their streams; only then may a rank refill its buffers.
        if not _all_ranks_agree(self.comm, ok):
            return False
        try:
            self._reset_local()
        except RuntimeError:
            ok = False
        return _all_ranks_agree(self.comm, ok)

    def error_bits(self) -> int:
        """Return this rank's sticky error bits (synchronizes the device)."""
        return self.counters()["error_bits"]

    def calls(self) -> int:
        """Return how many hop launches completed on this rank (synchronizes)."""
        return int(self.state[STATE_CALLS].item())

    def counters(self) -> Dict[str, int]:
        """Return the state words of this rank in one copy (synchronizes the device).

        Keys: ``calls`` (completed hop launches since the last reset),
        ``error_bits`` (sticky; bit 1 Lamport timeout, 2 row-sum timeout,
        4 permutation index out of range), ``timeout_polls`` (largest poll
        count at a timeout) and the host-side ``layout_repairs``.
        """
        torch.cuda.synchronize(self.state.device)
        words = self.state.cpu().tolist()
        return {
            "calls": words[STATE_CALLS],
            "error_bits": words[STATE_ERROR_WORDS[0]]
            | words[STATE_ERROR_WORDS[1]]
            | words[STATE_ERROR_WORDS[2]],
            "timeout_polls": words[STATE_TIMEOUT_POLLS],
            "layout_repairs": self.layout_repairs,
        }
