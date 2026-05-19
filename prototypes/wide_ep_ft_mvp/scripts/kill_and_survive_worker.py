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
import importlib.util
import json
import os
import signal
import sys
import time
import types
from typing import Any

# Stubs live in the same package as this script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)


def _load_ep_group_health() -> Any:
    """Load EPGroupHealth without triggering the broken tensorrt_llm package init.

    The full ``tensorrt_llm`` package init pulls in ``transformers`` symbols
    that don't exist in this dev environment (see audit notes); a clean
    ``from tensorrt_llm... import EPGroupHealth`` blows up before we ever
    reach EPGroupHealth. Workaround: load the single module file by absolute
    path and stub the package chain.
    """
    src = os.path.join(_REPO_ROOT, "tensorrt_llm/_torch/modules/fused_moe/ep_group_health.py")
    spec = importlib.util.spec_from_file_location("_loaded_ep_group_health", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name in (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.modules",
        "tensorrt_llm._torch.modules.fused_moe",
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = pkg
    sys.modules["tensorrt_llm._torch.modules.fused_moe.ep_group_health"] = module
    return module


_egh = _load_ep_group_health()
EPGroupHealth = _egh.EPGroupHealth

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
from prototypes.wide_ep_ft_mvp.stubs.shm_completion_flags import (  # noqa: E402
    ShmCompletionFlagTable,
)

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
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Unique per-run identifier for the shared-memory completion-flag dir",
    )
    parser.add_argument(
        "--watchdog-timeout-sec",
        type=float,
        default=5.0,
        help="Watchdog presumes a peer dead if its completion-flag has not advanced for this long",
    )
    parser.add_argument(
        "--watchdog-poll-interval-sec",
        type=float,
        default=0.1,
        help="How often the watchdog snapshots the completion-flag table",
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
    flags = ShmCompletionFlagTable(ep_size, local_rank, run_id=args.run_id)
    # Barrier so every rank has created its counter file before any watchdog
    # opens its peers' files (otherwise a fast rank could mmap a not-yet-
    # created peer file and get an empty/garbage view).
    comm.Barrier()
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
        config=WatchdogConfig(
            poll_interval_sec=args.watchdog_poll_interval_sec,
            timeout_sec=args.watchdog_timeout_sec,
        ),
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
        flags.close()

    _emit(local_rank, "loop_end", final_active_count=health.get_active_count())
    # Bypass mpi4py's atexit (which would call MPI_Finalize — a collective
    # that hangs forever because the dead victim cannot participate). This is
    # exactly the audit-1a Day 2 F4 finding ("survivors hang in their next
    # collective") manifesting at process-shutdown time. Production: PR 1d.0
    # + 1c.3 will install a coordinated shutdown path that detects the
    # poisoned world and skips MPI_Finalize.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
