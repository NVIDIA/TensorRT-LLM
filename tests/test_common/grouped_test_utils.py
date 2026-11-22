# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side helpers for tests that reuse MPI workers.

Used by the automatic session-reuse layer (``test_common/session_reuse.py``).
This is test infrastructure only and must not be used on production inference
paths. Heavy imports are done lazily so importing this module never triggers a
circular import during package initialization.
"""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def preserve_worker_env(n_workers: int, name: str) -> Iterator[None]:
    """Restore a variable on launcher-owned ranks after the executor shuts down.

    Callers must use ``private_mpi_session`` for locally spawned workers.
    """
    from tensorrt_llm.executor.utils import create_mpi_comm_session, get_spawn_proxy_process_env

    session = create_mpi_comm_session(n_workers) if get_spawn_proxy_process_env() else None
    previous = dict(session.submit_sync(get_worker_env, name)) if session else {}
    try:
        yield
    finally:
        if session:
            session.submit_sync(restore_worker_env, name, previous)


def get_worker_env(name: str) -> tuple[int, str | None]:
    """Read an environment variable with its rank for order-independent restoration."""
    import os

    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank(), os.environ.get(name)


def restore_worker_env(name: str, previous: dict[int, str | None]) -> None:
    """Restore this rank's environment variable, including its original absence."""
    import os

    from mpi4py import MPI

    value = previous[MPI.COMM_WORLD.Get_rank()]
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _ranked_run(fn):
    """Runs inside a worker; module-level so it is picklable."""
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank(), fn()


def submit_sync_per_worker(mpi_session, fn, timeout: float = 60.0) -> list:
    """Run ``fn`` exactly ONCE on every worker; return results ordered by rank.

    ``MpiPoolSession.submit`` runs ``fn`` once per worker behind a barrier
    across the worker world, but returns the results in arbitrary order; each
    result is tagged with its worker's rank to order them.

    This doubles as the pool health probe: completing the barrier proves every
    worker is alive AND that the worker world's collectives still function —
    the thing tests actually depend on. The wait is bounded by ``timeout``: a
    wedged worker (e.g. stuck in a collective after the previous test's
    executor died mid-shutdown) never completes the barrier, so an unbounded
    ``result()`` would hang the whole session. A timeout is reported as probe
    failure, making the caller retire the pool and spawn a fresh one.
    """
    import concurrent.futures

    futures = mpi_session.submit(_ranked_run, fn)
    done, not_done = concurrent.futures.wait(futures, timeout=timeout)
    if not_done:
        raise RuntimeError(
            f"health probe timed out after {timeout}s: "
            f"{len(not_done)}/{len(futures)} worker tasks never returned"
        )
    by_rank = dict(f.result() for f in done)  # .result() re-raises worker errors
    missing = set(range(mpi_session.n_workers)) - set(by_rank)
    if missing:  # unreachable given submit's one-per-worker guarantee
        raise RuntimeError(f"no task ran on worker ranks {sorted(missing)}")
    return [result for _, result in sorted(by_rank.items())]


def reset_worker_torch_compile_state() -> None:
    """Reset per-worker torch.compile / Dynamo state (runs inside each worker).

    Known torch.compile test items use private sessions because other
    process-global state, including the Userbuffers Manager, cannot be reset
    safely between Engines. Dynamo's process-global, per-code-object recompile
    counter can still accumulate for compile activity outside those items,
    trip ``recompile_limit``, and hard-fail under ``fullgraph=True`` with
    ``FailOnRecompileLimitHit``, aborting the MPI job. Keep this reset as a
    defensive health-probe action and run it on every worker via
    ``submit_sync_per_worker``.
    """
    import torch

    torch._dynamo.reset()
