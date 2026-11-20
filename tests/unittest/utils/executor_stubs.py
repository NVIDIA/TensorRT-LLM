# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for testing the executor proxy without workers or an MPI session."""

import queue
import threading
from types import SimpleNamespace

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker_process_monitor import WorkerProcessMonitor


def bare_proxy(cls=GenerationExecutorProxy, **attributes):
    """Return a ``cls`` instance built without ``__init__``.

    It has no MPI session, sockets or threads, and carries the state that the
    health, fatal-error, cancellation and shutdown paths read. Its workers
    never started, so ``__del__`` -> ``shutdown()`` does nothing; a test that
    sets ``workers_started`` must reset it. ``attributes`` override or add
    state.
    """
    proxy = cls.__new__(cls)
    proxy.workers_started = False
    proxy.doing_shutdown = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._results = {}
    proxy._error_queue = queue.Queue()
    proxy._request_sender = None
    proxy._multi_frontend_ipc_dir = None
    proxy._owns_mpi_session = False
    proxy.mpi_session = None
    proxy.mpi_futures = []
    proxy._shutdown_event = threading.Event()
    proxy._worker_process_monitor = WorkerProcessMonitor()
    for name, value in attributes.items():
        setattr(proxy, name, value)
    return proxy


def stub_command_socket(on_use=lambda: None):
    """Return the ZMQ socket surface that ``_RequestSender`` uses, backed by nothing.

    ``on_use`` runs on every socket and monitor call that libzmq would require
    to come from the socket's owning thread.
    """

    def use(*_args, **_kwargs):
        on_use()

    def get_monitor_socket(**_kwargs):
        return SimpleNamespace(poll=lambda timeout: False, getsockopt=lambda option: 0, close=use)

    return SimpleNamespace(
        setsockopt=use, get_monitor_socket=get_monitor_socket, disable_monitor=use
    )
