# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process attachment shared by HTTP and OpenEngine serving."""

import json
import os
import selectors

# Re-exec the serving CLI with sys.executable and an argument list, without a shell.
import subprocess  # nosec B404
import sys
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from tensorrt_llm.llmapi.mpi_session import split_mpi_env
from tensorrt_llm.logger import logger


class FrontendStartupCancelled(RuntimeError):
    """Frontend startup was interrupted; the caller determines the exit status."""


class MultiFrontendMode(NamedTuple):
    """This process's role under multi-frontend serving (prototype)."""

    num_frontends: int
    is_attached_frontend: bool

    @property
    def is_launcher(self) -> bool:
        """Owns the engine and spawns/cleans the attached frontends."""
        return self.num_frontends > 1 and not self.is_attached_frontend


def _init_multi_frontend_mode(llm_args: dict, enabled: bool) -> MultiFrontendMode:
    """Resolve this process's multi-frontend serving role.

    num_serve_frontends=K runs K serving frontend processes against ONE
    executor: the launcher (frontend 0) owns the engine and spawns K-1
    attached frontends (classic IPC executor path only). enabled=False
    entry points (e.g. disaggregated MPI workers) never honor the knob.
    """
    if not enabled:
        if llm_args.pop("num_serve_frontends", 1) > 1:
            logger.warning(
                "num_serve_frontends is only supported on plain "
                "trtllm-serve; ignored on this entry point."
            )
        return MultiFrontendMode(1, False)

    mode = MultiFrontendMode(
        llm_args.get("num_serve_frontends", 1), os.getenv("TLLM_EXECUTOR_ATTACH_INFO") is not None
    )
    if mode.is_launcher and llm_args.get("orchestrator_type") is not None:
        raise ValueError(
            "num_serve_frontends > 1 requires the default (classic IPC) "
            "executor path, not orchestrator_type="
            f"{llm_args.get('orchestrator_type')!r}"
        )
    return mode


def _spawn_attached_frontends(
    llm,
    num_frontends: int,
    *,
    extra_env: dict[str, str] | None = None,
    cancelled: threading.Event | None = None,
    report_failure: Callable[[int, str, str], None] | None = None,
) -> list:
    """Spawn num_frontends - 1 attached serving frontend processes.

    Each child re-execs this trtllm-serve command line with env vars
    carrying the launcher executor's attach endpoints; its executor
    attaches to the already-running worker instead of launching one (see
    GenerationExecutor.create / GenerationExecutorFrontendProxy).

    Blocks until every child signals READY over its inherited pipe: a
    successful Popen only proves the process exists, while the frontend
    can still fail during executor attach or server setup. Any child
    failure (or a missed deadline) fails the whole group, terminating
    the children already started, so num_serve_frontends=K never
    silently degrades to fewer frontends.
    """
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy

    executor = getattr(llm, "_executor", None)
    if (
        not isinstance(executor, GenerationExecutorProxy)
        or (attach_info := executor.multi_frontend_attach_info()) is None
    ):
        raise ValueError(
            "num_serve_frontends > 1 requires the classic IPC executor "
            f"proxy in multi-frontend mode, got {type(executor).__name__}"
        )
    # Carries the executor HMAC keys; the child deletes it from its env
    # once consumed (GenerationExecutor.create).
    attach_env = json.dumps(attach_info)

    children, ready_fds = [], []
    try:
        for frontend_id in range(1, num_frontends):
            # Strip MPI/SLURM identity vars: an inherited rank identity would
            # make the child's mpi4py try to (re-)join the launcher's job.
            env, _ = split_mpi_env()
            env.update(extra_env or {})
            env["TLLM_EXECUTOR_ATTACH_INFO"] = attach_env
            env["TLLM_EXECUTOR_FRONTEND_ID"] = str(frontend_id)
            env["TLLM_DISABLE_MPI"] = "1"
            read_fd, write_fd = os.pipe()
            ready_fds.append(read_fd)
            env["TLLM_FRONTEND_READY_FD"] = str(write_fd)
            try:
                child = subprocess.Popen([sys.executable] + sys.argv, env=env, pass_fds=(write_fd,))  # nosec B603
            finally:
                # The child now holds the only write end; its exit before
                # READY surfaces as EOF on read_fd.
                os.close(write_fd)
            children.append(child)
            logger.info(f"Launched attached serving frontend {frontend_id} (pid {child.pid})")
        _wait_attached_frontends_ready(children, ready_fds, cancelled, report_failure)
    except BaseException:
        _terminate_attached_frontends(children)
        raise
    finally:
        for fd in ready_fds:
            os.close(fd)
    return children


def _wait_attached_frontends_ready(
    children: list,
    ready_fds: list,
    cancelled: threading.Event | None = None,
    report_failure: Callable[[int, str, str], None] | None = None,
) -> None:
    """Block until every attached frontend writes its READY byte."""
    timeout = float(os.getenv("TLLM_FRONTEND_READY_TIMEOUT", "300"))
    deadline = time.monotonic() + timeout
    pending = dict(zip(ready_fds, children))
    with selectors.DefaultSelector() as selector:
        for fd in ready_fds:
            selector.register(fd, selectors.EVENT_READ)
        while pending:
            if cancelled is not None and cancelled.is_set():
                raise FrontendStartupCancelled("Frontend startup cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"{len(pending)} attached frontend(s) not ready within "
                    f"{timeout:.0f}s (TLLM_FRONTEND_READY_TIMEOUT)"
                )
            for key, _ in selector.select(min(remaining, 1.0)):
                fd = key.fd
                selector.unregister(fd)
                child = pending.pop(fd)
                if os.read(fd, 1) != b"R":  # EOF: pipe closed without READY
                    try:
                        return_code = child.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        return_code = None
                    if return_code is not None and return_code != 0 and report_failure is not None:
                        report_failure(return_code, "server", "model_initialization")
                    raise RuntimeError(
                        f"Attached frontend (pid {child.pid}) exited before signaling READY"
                    )
                logger.info(f"Attached frontend (pid {child.pid}) is ready")
            # READY is only a startup milestone: keep checking those children while
            # siblings initialize, including the iteration that consumes the last READY.
            for child in children:
                if child.poll() is not None:
                    if child.returncode != 0 and report_failure is not None:
                        report_failure(child.returncode, "server", "model_initialization")
                    raise RuntimeError(
                        f"Attached frontend (pid {child.pid}) exited with code "
                        f"{child.returncode} during frontend startup"
                    )


def _signal_frontend_ready(multi_frontend: MultiFrontendMode) -> None:
    """Report READY to the launcher over the inherited pipe.

    Called once everything fallible in an attached frontend's startup
    (port bind, executor attach, LLM and OpenAIServer construction,
    middleware registration) has succeeded; the launcher blocks group
    startup on this byte (see _wait_attached_frontends_ready).
    """
    ready_fd = os.environ.pop("TLLM_FRONTEND_READY_FD", None)
    if not (multi_frontend.is_attached_frontend and ready_fd):
        return
    fd = int(ready_fd)
    os.write(fd, b"R")
    os.close(fd)


def _terminate_attached_frontends(children: list) -> None:
    """Stop and reap the group with one shared grace deadline."""
    for child in children:
        child.terminate()
    deadline = time.monotonic() + 10
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
    for child in children:
        child.wait()
