# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot logits gather of the target head straight into the final row layout (``distributed_head_sampler``).

Part of the ``distributed_head_sampler`` fast path (built by ``split_sampler.target_head.TargetHead``
on top of ``split_sampler``). On a step of ``ROWS`` rows the head's
``[ROWS, shard]`` BF16 shards are gathered by one kernel (``pv_logits_gather.cu``) into a
fresh ``[ROWS, world * shard]`` tensor, rank ``p``'s shard in columns ``p * shard ..`` --
what the built-in ``Linear`` all-gather followed by its ``cat`` along the last dimension
(``_allgather``; ``target_head.MultimemGather``'s MULTIMEM gather + ``cat`` the same) returns, as a copy of the same
bytes, in one launch, without the gather's two barriers or the ``cat`` launch.

Transport and protocol: LL pairs over the NVLink multicast fabric, no reset, two areas
alternating by epoch parity (the CUDA source's header has the ownership and progress
argument). The receive area is written only
through the multicast address by the publishing ranks' kernels and read only by this rank's
kernel; no kernel stores sentinels into fabric memory.

Construction is collective over the TP group (the library load, the fabric allocation, the
zero fill and a dry launch, each agreed) and either succeeds on every rank or raises on every
rank. Whether a call takes the gather depends only on rank-uniform facts (the row count and
the LM-head shard width), so every rank runs the same sequence of gathers.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch

from tensorrt_llm.mapping import Mapping

#: Rows the gather runs on: the K = 7 verify block of one request.
ROWS = 8
#: CTAs of the kernel (``kCtas`` in the source); one epoch counter each.
CTAS = 128
#: A poll waiting longer than this traps (a peer that never published).
POLL_TIMEOUT_S = 600
#: The kernel's poll schedule (``pollMode`` in the source): 0, every stale vector re-polled each round
#: (1 spins on the first stale vector, then re-polls; 2 is 1 with a short sleep between those spins); every
#: mode reads the same words.
POLL_MODE = 0
LIBRARY_NAME = "musespark_open_pv_logits"
SOURCE = Path(__file__).resolve().parent / "pv_logits_gather.cu"
NVCC_FLAGS = ("-O3", "-lineinfo")

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 3):
        return "103a"
    if capability == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the pv logits gather is built for sm_103a / sm_107a, not sm_{capability[0]}{capability[1]}"
    )


def load_library() -> str:
    """Build (if needed) and load the library once per process; its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            from torch.utils.cpp_extension import load

            target = arch_target()
            path = load(
                name=f"{LIBRARY_NAME}_sm{target}",
                sources=[str(SOURCE)],
                extra_cuda_cflags=list(NVCC_FLAGS)
                + [f"-gencode=arch=compute_{target},code=sm_{target}"],
                is_python_module=False,
                verbose=False,
            )
            _LOADED = str(path) if path else LIBRARY_NAME
        return _LOADED


def area_bytes(world: int, shard: int, rows: int = ROWS) -> int:
    """Bytes of the receive areas: two (alternating by epoch parity), each one 64-bit pair per
    32-bit data word of the gathered rows."""
    return 2 * rows * world * (shard // 2) * 8


class PvLogitsGather:
    """This rank's one-shot gather of the LM head's ``[ROWS, shard]`` BF16 shard."""

    @staticmethod
    def reason(mapping, lm_head) -> Optional[str]:
        """Why the gather cannot stand in for the head's all-gather here, or None (rank-local)."""
        if getattr(lm_head, "lora", None) is not None:
            return "the LM head has LoRA"
        shard = int(lm_head.weight.shape[0])
        if shard % 4:
            return f"a shard of {shard} is not a whole number of 4-element vectors"
        if not 1 <= int(mapping.tp_size) <= 8:
            return f"the gather is written for up to 8 ranks, not {mapping.tp_size}"
        if ROWS * int(mapping.tp_size) * (shard // 4) > CTAS * 512 * 8:
            return "the gathered rows exceed the kernel's grid"
        if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
            return "the gather runs on SM103 or SM107 only"
        return None

    def __init__(
        self, mapping: Mapping, shard: int, device: torch.device, poll_mode: Optional[int] = None
    ) -> None:
        """Collective over the TP group."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        self.world = int(mapping.tp_size)
        self.rank = int(mapping.tp_rank)
        self.shard = int(shard)
        self.device = device
        self.pdl = True
        self.poll_mode = POLL_MODE if poll_mode is None else int(poll_mode)
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            load_library()
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] pv logits gather: the library failed to load on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        size = area_bytes(self.world, self.shard)
        try:
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, size, mapping, mapping.is_multi_node()
            )
            self._uc = self._handle.get_uc_buffer(self.rank, (size // 4,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((size // 4,), torch.int32, 0)
            with torch.cuda.device(device):
                self._uc.zero_()
                self.counters = torch.zeros(CTAS, dtype=torch.int32, device=device)
                self._op(self._dummy(), self._dummy_out(), dry=True)
                torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] pv logits gather: the workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self.calls = 0

    def _dummy(self) -> torch.Tensor:
        return torch.empty(ROWS, self.shard, dtype=torch.bfloat16, device=self.device)

    def _dummy_out(self) -> torch.Tensor:
        return torch.empty(ROWS, self.world * self.shard, dtype=torch.bfloat16, device=self.device)

    def _op(self, local: torch.Tensor, out: torch.Tensor, dry: bool = False) -> None:
        torch.ops.musespark_open_pv_logits.gather(
            local,
            out,
            self.counters,
            int(self._mc.data_ptr()),
            int(self._uc.data_ptr()),
            self.rank,
            self.world,
            ROWS,
            self.shard,
            self.pdl,
            int(POLL_TIMEOUT_S * 1e9),
            self.poll_mode,
            dry,
        )

    def takes(self, hidden_states: torch.Tensor, lm_head) -> bool:
        return (
            hidden_states.dim() == 2
            and hidden_states.shape[0] == ROWS
            and int(lm_head.weight.shape[0]) == self.shard
        )

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        """The gathered ``[ROWS, world * shard]`` row of this call's ``[ROWS, shard]`` shard."""
        local = local.contiguous()
        out = torch.empty(
            local.shape[0], self.world * self.shard, dtype=local.dtype, device=local.device
        )
        self._op(local, out)
        self.calls += 1
        return out

    def __call__(self, lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
        """``Linear.forward`` of the column-parallel gathered head, with this gather.

        ``apply_linear`` as the built-in forward calls it (so a fork point bound on the
        instance fires), then the gather in place of ``_allgather``'s gather and ``cat``.
        """
        local = lm_head.apply_linear(hidden_states, lm_head.bias, None, None)
        return self.gather(local.view(hidden_states.shape[0], self.shard))
