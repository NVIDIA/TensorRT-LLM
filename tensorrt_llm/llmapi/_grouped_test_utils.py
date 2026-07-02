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
from typing import Optional


def restore_env_var(name: str, value: Optional[str]) -> None:
    """Restore an env var to a previously captured value (or remove it)."""
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


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
    prev = os.environ.get("TRTLLM_HF_WEIGHT_CACHE")
    prev_entries = os.environ.get("TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES")
    os.environ["TRTLLM_HF_WEIGHT_CACHE"] = "1"
    os.environ["TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"] = max_entries
    try:
        yield
    finally:
        restore_env_var("TRTLLM_HF_WEIGHT_CACHE", prev)
        restore_env_var("TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES", prev_entries)


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
