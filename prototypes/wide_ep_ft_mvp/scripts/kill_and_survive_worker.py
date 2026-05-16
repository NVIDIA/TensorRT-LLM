#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP FT MVP prototype — MPI worker.

One instance per MPI rank. Wires together every stub in ``stubs/`` plus the
real cherry-picked ``EPGroupHealth``, then runs a tight pseudo-AlltoAll loop
that drives the watchdog seam.

This is intentionally **not** a real model engine. It is a minimal harness
that:

  * Allocates a fake "completion-flag table" in shared host memory (one
    counter per rank, advanced once per loop iteration).
  * Runs the iteration-boundary hook at the top of each iteration.
  * Lets the test driver (``kill_and_survive_driver.py``) SIGKILL one rank
    mid-loop and observe whether survivors detect, reconfigure, and continue.

The pseudo-AlltoAll loop replaces the real MNNVL kernel because:

  1. Without 1a.2/1a.3 (kernel mask) cherry-picked or stubbed inline, the
     real kernel will hang on the dead peer indefinitely. See
     ``kernel/README.md`` for the two integration paths.
  2. The point of the prototype is to exercise the **integration seams** —
     watchdog → ``mark_failed`` → broadcast → iteration hook → reconfigure —
     not the kernel itself.

Per-rank events are streamed to stdout as JSON lines so the driver can
aggregate them into a single timeline.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

# Stubs live in the same package as this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from prototypes.wide_ep_ft_mvp.stubs.alltoall_watchdog import (  # noqa: E402
    AlltoAllWatchdog,
    WatchdogConfig,
)
from prototypes.wide_ep_ft_mvp.stubs.eplb_slot_remap import (  # noqa: E402
    EplbSlotRemapStub,
    LayerPlacement,
)
from prototypes.wide_ep_ft_mvp.stubs.iteration_boundary_hook import (  # noqa: E402
    IterationBoundaryHook,
    IterationHookEvent,
)
from prototypes.wide_ep_ft_mvp.stubs.mpi_ft_subcomm import MpiFtSubcommStub  # noqa: E402
from prototypes.wide_ep_ft_mvp.stubs.nccl_async_error_monitor import (  # noqa: E402
    configure_nccl_async_error_handling,
)
from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth  # noqa: E402

# -------------------------------------------------------------------------
# Pseudo completion-flag table
# -------------------------------------------------------------------------
#
# In production this is MNNVL fabric memory shared across all ranks. The
# prototype substitutes a per-rank monotonic counter advanced once per
# iteration; the watchdog reads its peers' counters via MPI Allgather. A
# rank that's been SIGKILL'd stops advancing its counter; the watchdog sees
# no progress for ``timeout_sec`` and presumes it dead.


class _PseudoCompletionFlagTable:
    """A per-rank monotonic counter, exchanged via MPI Allgather."""

    def __init__(self, ep_size: int, local_rank: int, comm: Any) -> None:
        self._ep_size = ep_size
        self._local_rank = local_rank
        self._comm = comm
        self._local_value = 0

    def advance(self) -> None:
        self._local_value += 1

    def snapshot(self) -> list[int]:
        from mpi4py import MPI  # noqa: F401  (sanity check that mpi4py is loaded)

        gathered = self._comm.allgather(self._local_value)
        return [int(v) for v in gathered]


# -------------------------------------------------------------------------
# Event log (one JSON line per event)
# -------------------------------------------------------------------------


def _emit(local_rank: int, event_type: str, **fields: Any) -> None:
    payload = {
        "rank": local_rank,
        "event": event_type,
        "wall_time_sec": time.monotonic(),
        **fields,
    }
    print(json.dumps(payload, default=str), flush=True)


def _install_quiet_atexit_for_clean_exit_code() -> None:
    """Install a SIGTERM handler that bypasses Python finalizers.

    If the test driver SIGKILLs us, we want survivors to log a clean exit
    rather than have Python's atexit / MPI cleanup fire at the wrong time.
    """

    def _handler(signum: int, _frame: Any) -> None:  # noqa: ARG001
        # The 1d.0 handler in mpiUtils.cpp covers SIGABRT/SIGSEGV; this Python
        # handler is for SIGTERM from the driver in case we're asked to wind
        # down a survivor cleanly at end of test.
        os._exit(0)

    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--iter-sleep-sec",
        type=float,
        default=0.05,
        help="Time per pseudo-iteration; tight loop drives the watchdog seam",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers to register with the EPLB stub (1-2 per spec)",
    )
    args = parser.parse_args()

    configure_nccl_async_error_handling()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    ep_size = comm.Get_size()
    local_rank = comm.Get_rank()
    pid = os.getpid()

    _install_quiet_atexit_for_clean_exit_code()
    _emit(local_rank, "worker_start", pid=pid, ep_size=ep_size)

    health = EPGroupHealth(ep_size)
    flags = _PseudoCompletionFlagTable(ep_size, local_rank, comm)
    eplb = EplbSlotRemapStub(ep_size)
    for layer_idx in range(args.num_layers):
        slot_to_rank = list(range(ep_size))  # one slot per rank, identity mapping
        eplb.register_layer_placement(
            LayerPlacement(layer_index=layer_idx, slot_count=ep_size, slot_to_rank=slot_to_rank)
        )

    def _on_iter_event(ev: IterationHookEvent) -> None:
        if ev.triggered_reconfigure and ev.reconfigure_result is not None:
            _emit(
                local_rank,
                "reconfigure_done",
                iteration=ev.iteration,
                cached_generation=ev.cached_generation,
                observed_generation=ev.observed_generation,
                duration_sec=ev.reconfigure_result.duration_sec,
                layers_touched=ev.reconfigure_result.layers_touched,
            )

    iter_hook = IterationBoundaryHook(health, eplb, on_event=_on_iter_event)

    def _on_peer_death(peer: int, when: float) -> None:
        _emit(local_rank, "watchdog_marked_failed", peer=peer, mark_time_sec=when)
        broadcast.broadcast_failure(peer)

    def _on_failure_received(peer: int, src: int, when: float) -> None:
        _emit(
            local_rank,
            "broadcast_received",
            peer=peer,
            from_rank=src,
            recv_time_sec=when,
        )

    watchdog = AlltoAllWatchdog(
        ep_size=ep_size,
        local_rank=local_rank,
        health=health,
        completion_flag_view=flags.snapshot,
        config=WatchdogConfig(poll_interval_sec=0.1, timeout_sec=5.0),
        on_peer_death=_on_peer_death,
    )
    broadcast = MpiFtSubcommStub(
        ep_size=ep_size,
        local_rank=local_rank,
        health=health,
        on_failure_received=_on_failure_received,
    )

    watchdog.start()
    broadcast.start()

    comm.Barrier()
    _emit(local_rank, "loop_start")

    try:
        for i in range(args.iterations):
            iter_hook.at_iteration_boundary()
            flags.advance()
            time.sleep(args.iter_sleep_sec)
            if i % 20 == 0:
                _emit(
                    local_rank,
                    "heartbeat",
                    iteration=i,
                    active_count=health.get_active_count(),
                    failed_ranks=sorted(health.get_failed_ranks()),
                    eplb_reconfigure_count=eplb.reconfigure_count(),
                )
    finally:
        broadcast.stop()
        watchdog.stop()

    _emit(local_rank, "loop_end", final_active_count=health.get_active_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
