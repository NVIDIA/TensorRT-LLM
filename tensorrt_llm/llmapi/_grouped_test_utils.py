# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for grouped multi-GPU tests that reuse one ``MpiPoolSession``
across many ``LLM`` instances.

This lives in-package (rather than under ``tests/``) because the unittest and
integration test roots have separate ``sys.path`` bases and cannot import a
single shared module otherwise. It is test-infrastructure only and must not be
used on production inference paths. Heavy imports are done lazily so importing
this module never triggers a circular import during package initialization.
"""

import os
from contextlib import contextmanager
from unittest import mock


@contextmanager
def hf_weight_cache_env(max_entries: str = "1"):
    """Enable the HF weight cache via env vars, restoring prior values on exit.

    IMPORTANT (load-bearing ordering): enter this BEFORE spawning the shared
    ``MpiPoolSession``. ``MpiPoolSession`` snapshots ``TRTLLM*``/``TLLM*`` env at
    spawn time and passes that copy to the workers (which are the processes that
    actually load weights); enabling the cache AFTER the pool spawns leaves it
    silently disabled in the workers. Back a module-scoped ``hf_weight_cache``
    fixture with this and have the session fixture depend on that fixture so the
    env is exported first.
    """
    with mock.patch.dict(
        os.environ,
        {
            "TRTLLM_HF_WEIGHT_CACHE": "1",
            "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES": max_entries,
        },
    ):
        yield


def shared_mpi_session(n_workers: int):
    """Generator yielding a shared ``MpiPoolSession`` (or ``None`` when MPI is
    disabled), shutting it down on teardown. Intended to back a module-scoped
    pytest fixture so a pool of workers is reused across many LLMs."""
    from tensorrt_llm._utils import mpi_disabled

    if mpi_disabled():
        yield None
        return

    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    mpi_session = MpiPoolSession(n_workers=n_workers)
    try:
        yield mpi_session
    finally:
        mpi_session.shutdown()


def mpi_session_kwargs(mpi_session) -> dict:
    """LLM kwargs that inject a shared MPI session, or ``{}`` when there is
    none (e.g. MPI disabled). Lets harness-based tests forward the session."""
    return {"_mpi_session": mpi_session} if mpi_session is not None else {}


def _run_on_worker_reporting_rank(fn) -> int:
    fn()
    from tensorrt_llm._utils import mpi_rank

    return mpi_rank()


def submit_sync_per_worker(mpi_session, fn, max_rounds: int = 8) -> None:
    """Run ``fn`` at least once on EVERY worker of the pool.

    ``MpiPoolSession.submit_sync`` enqueues ``n_workers`` tasks, but
    ``MPIPoolExecutor`` gives no one-task-per-worker guarantee: a fast task can
    be drained twice by one idle worker, leaving another untouched. Verify
    coverage by collecting the worker ranks that actually ran ``fn`` and
    resubmitting until all workers are covered.
    """
    expected = mpi_session.n_workers
    seen: set = set()
    for _ in range(max_rounds):
        seen.update(mpi_session.submit_sync(_run_on_worker_reporting_rank, fn))
        if len(seen) >= expected:
            return
    raise RuntimeError(
        f"task only reached worker ranks {sorted(seen)} after {max_rounds} "
        f"rounds; expected {expected} distinct workers"
    )


def reset_worker_torch_compile_state() -> None:
    """Reset per-worker torch.compile / Dynamo state (runs inside each worker).

    Dynamo's recompile counter is process-global and per-code-object. When
    worker processes are reused across LLMs (shared ``MpiPoolSession``), each
    ``torch_compile`` case recompiles the same ``model.forward`` code object
    under new guards; the count accumulates and eventually trips
    ``recompile_limit`` (16), which is a HARD failure under ``fullgraph=True``
    (``FailOnRecompileLimitHit``) and aborts the whole MPI job. Resetting
    between cases makes each LLM start from a clean compile cache, like a fresh
    process. Submit this to each worker via ``mpi_session.submit_sync(...)``.
    """
    import torch

    torch._dynamo.reset()


def reset_shared_session_torch_compile_state(make_llm) -> None:
    """Reset per-worker torch.compile state on the shared session behind a
    ``make_shared_llm`` factory (no-op if there is no shared session).

    Call from a grouped test's per-case teardown so recompile counts don't
    accumulate across cases on reused workers (see
    ``reset_worker_torch_compile_state`` for the failure it prevents).
    """
    mpi_session = getattr(make_llm, "mpi_session", None)
    if mpi_session is not None:
        submit_sync_per_worker(mpi_session, reset_worker_torch_compile_state)


def clear_worker_weight_cache() -> None:
    """Drop the per-worker HF raw-weight cache (runs inside each worker).

    The cache is a process-global keyed by checkpoint file fingerprints, so it
    otherwise lives until the worker process exits. Submit this to each worker
    via ``mpi_session.submit_sync(...)`` on group teardown to invalidate it
    explicitly instead of relying on process death.
    """
    from tensorrt_llm._torch.models.checkpoints import HfWeightLoader

    HfWeightLoader._clear_weight_cache()


def make_shared_llm(mpi_session):
    """Return an ``LLM`` factory that transparently injects a shared MPI session.

    Tests build the LLM by calling the factory exactly like ``LLM(...)``; the
    shared session (if any) is passed through without the test having to know it
    exists. Falls back to a private per-LLM session when ``mpi_session`` is None.
    The factory exposes ``.mpi_session`` so callers can reset per-worker compile
    state between cases without threading the session separately.
    """
    from tensorrt_llm import LLM

    def shared_llm(*args, **kwargs):
        return LLM(*args, **kwargs, **mpi_session_kwargs(mpi_session))

    shared_llm.mpi_session = mpi_session
    return shared_llm
