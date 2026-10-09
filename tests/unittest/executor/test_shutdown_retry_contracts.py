# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle contracts for retryable Ray and RPC shutdown."""

import ast
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[3]


def _extract_function(source, function_name, namespace):
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[function_name]


def _extract_class(source, class_name, methods, namespace, bases=()):
    tree = ast.parse(source.read_text())
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    body = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods
    ]
    derived = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id=base, ctx=ast.Load()) for base in bases],
        keywords=[],
        body=body,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


class _Timeout(Exception):
    pass


class _Ref:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def get(self):
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _RemoteShutdown:
    def __init__(self, refs):
        self.refs = list(refs)
        self.calls = 0

    def remote(self):
        ref = self.refs[min(self.calls, len(self.refs) - 1)]
        self.calls += 1
        return ref


class _RayWorker:
    def __init__(self, refs):
        self.shutdown = _RemoteShutdown(refs)


class _FakeRay:
    def __init__(self):
        self.exceptions = SimpleNamespace(GetTimeoutError=_Timeout)
        self.util = SimpleNamespace(remove_placement_group=lambda group: None)
        self.killed = []

    @staticmethod
    def get(ref, timeout=None):
        return ref.get()

    def kill(self, worker, no_restart=True):
        self.killed.append(worker)

    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def shutdown():
        return None


def _ray_executor(ray):
    cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/ray/executor.py",
        "RayExecutor",
        {"shutdown"},
        {
            "ray": ray,
            "time": __import__("time"),
            "logger": SimpleNamespace(
                warning=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None
            ),
            "logger_debug": lambda *args, **kwargs: None,
        },
    )
    executor = cls()
    executor._shutdown_event = threading.Event()
    executor.main_loop = None
    executor.main_loop_task_obj = None
    executor.rpc_client = SimpleNamespace(close=lambda: None)
    executor.placement_group = None
    executor.bundle_indices = None
    executor.has_start_local_cluser = False
    executor._wait_for_cluster_resource_release = lambda timeout: None
    return executor


def test_ray_shutdown_failure_preserves_actors_for_retry():
    ray = _FakeRay()
    worker = _RayWorker([_Ref([RuntimeError("cleanup failed")]), _Ref([None])])
    executor = _ray_executor(ray)
    executor.workers = [worker]

    with pytest.raises(RuntimeError, match="cleanup failed"):
        executor.shutdown()

    assert executor.workers == [worker]
    assert not executor._shutdown_event.is_set()
    assert ray.killed == []

    executor.shutdown()
    assert worker.shutdown.calls == 2
    assert ray.killed == [worker]
    assert executor.workers is None


def test_ray_shutdown_timeout_reuses_pending_ref():
    ray = _FakeRay()
    pending = _Ref([_Timeout("still running"), None])
    worker = _RayWorker([pending])
    executor = _ray_executor(ray)
    executor.workers = [worker]

    with pytest.raises(_Timeout, match="still running"):
        executor.shutdown()
    assert executor._pending_shutdown_refs == [pending]

    executor.shutdown()
    assert worker.shutdown.calls == 1
    assert executor.workers is None


class _BaseWorker:
    def shutdown(self):
        outcome = self.base_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome


class _Comm:
    def __init__(self, gathered):
        self.gathered = list(gathered)
        self.broadcasts = []

    def allgather(self, value):
        return self.gathered.pop(0)

    def bcast(self, value, root):
        self.broadcasts.append(value)
        return value


def _rpc_worker(comm):
    cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/rpc_worker.py",
        "RpcWorker",
        {"shutdown", "commit_shutdown", "_complete_shutdown_commit"},
        {
            "Base": _BaseWorker,
            "mpi_rank": lambda: 0,
            "mpi_comm": lambda: comm,
            "logger_debug": lambda *args, **kwargs: None,
            "_RPC_SHUTDOWN_RETRY": "retry",
            "_RPC_SHUTDOWN_COMMIT": "commit",
        },
        bases=("Base",),
    )
    worker = cls()
    worker._rpc_collective_shutdown = True
    worker._rpc_shutdown_attempted = False
    worker._rpc_shutdown_ready = False
    worker._rpc_shutdown_committed = False
    worker.shutdown_event = threading.Event()
    return worker


def test_rpc_cleanup_retry_and_commit_are_rank_aligned_and_idempotent():
    comm = _Comm([["RuntimeError: local failure", None], [None, None]])
    worker = _rpc_worker(comm)
    worker.base_outcomes = [RuntimeError("local failure"), None]

    with pytest.raises(RuntimeError, match="local failure"):
        worker.shutdown()
    assert not worker._rpc_shutdown_ready
    assert comm.broadcasts == []

    worker.shutdown()
    assert worker._rpc_shutdown_ready
    assert comm.broadcasts == ["retry"]

    worker.commit_shutdown()
    worker.commit_shutdown()
    assert comm.broadcasts == ["retry", "commit"]
    assert not worker.shutdown_event.is_set()

    worker._complete_shutdown_commit()
    assert worker.shutdown_event.is_set()


def test_rpc_peer_failure_is_propagated_to_rank_zero():
    comm = _Comm([[None, "RuntimeError: peer failure"]])
    worker = _rpc_worker(comm)
    worker.base_outcomes = [None]

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError: peer failure"):
        worker.shutdown()
    assert not worker._rpc_shutdown_ready


class _RemoteCall:
    def __init__(self, name, calls, error=None):
        self.name = name
        self.calls = calls
        self.error = error

    def remote(self, *, need_response):
        self.calls.append((self.name, need_response))
        if self.error is not None:
            raise self.error


class _RpcClient:
    def __init__(self, calls, shutdown_error=None):
        self.calls = calls
        self.shutdown_error = shutdown_error

    def shutdown(self):
        return _RemoteCall("cleanup", self.calls, self.shutdown_error)

    def commit_shutdown(self):
        return _RemoteCall("commit", self.calls)

    def close(self):
        self.calls.append(("close", True))


def _rpc_proxy():
    return _extract_class(
        _ROOT / "tensorrt_llm/executor/rpc_proxy.py",
        "GenerationExecutorRpcProxy",
        {"shutdown_remote", "shutdown"},
        {
            "logger_debug": lambda *args, **kwargs: None,
            "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
            "threading": threading,
        },
    )


def test_rpc_proxy_requires_cleanup_and_commit_responses():
    calls = []
    proxy = _rpc_proxy()()
    proxy.rpc_client = _RpcClient(calls)

    proxy.shutdown_remote()

    assert calls == [("cleanup", True), ("commit", True)]


def test_rpc_proxy_cleanup_failure_keeps_retry_gate_and_resources_live():
    calls = []
    proxy = _rpc_proxy()()
    proxy._shutdown_event = threading.Event()
    proxy.rpc_client = _RpcClient(calls, RuntimeError("remote cleanup failed"))
    proxy.main_loop = None
    proxy.main_loop_task_obj = None
    proxy.main_loop_thread = None
    proxy.mpi_session = object()

    with pytest.raises(RuntimeError, match="remote cleanup failed"):
        proxy.shutdown()

    assert not proxy._shutdown_event.is_set()
    assert proxy.mpi_session is not None
    assert calls == [("cleanup", True)]


def test_rpc_server_completes_commit_only_after_response_send():
    source = _ROOT / "tensorrt_llm/executor/rpc/rpc_server.py"
    tree = ast.parse(source.read_text())
    server = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RPCServer"
    )
    process = next(
        node
        for node in server.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_requests"
    )
    send_line = min(
        node.lineno
        for node in ast.walk(process)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_send_response"
    )
    complete_line = min(
        node.lineno
        for node in ast.walk(process)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "complete_commit"
    )
    method_source = ast.get_source_segment(source.read_text(), process)

    assert send_line < complete_line
    assert 'req.method_name == "commit_shutdown"' in method_source
    assert "response.error is None" in method_source


def test_symm_buffer_release_retains_failures_for_retry():
    cache = {
        "failed": ("bad", object()),
        "released": ("good", object()),
    }
    attempts = {"bad": 0}

    def free(buffer):
        if buffer == "bad" and attempts["bad"] == 0:
            attempts["bad"] += 1
            raise RuntimeError("destroy failed")
        return 16

    release = _extract_function(
        _ROOT / "tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_deepgemm.py",
        "release_symm_buffer_cache",
        {
            "_MEGA_MOE_SYMM_BUFFER_CACHE": cache,
            "_free_symm_buffer": free,
            "logger": SimpleNamespace(
                error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None
            ),
        },
    )

    with pytest.raises(RuntimeError, match="failed to release 1"):
        release()
    assert list(cache) == ["failed"]

    release()
    assert cache == {}


def test_mpi_worker_retries_symm_release_before_destroying_process_group():
    events = []
    module_name = "test_retryable_megamoe_symm_release"
    release_attempts = 0

    def release_symm_buffer_cache():
        nonlocal release_attempts
        release_attempts += 1
        events.append("release_symm")
        if release_attempts == 1:
            raise RuntimeError("symm release failed")

    mega_moe = types.ModuleType(module_name)
    mega_moe.release_symm_buffer_cache = release_symm_buffer_cache
    fake_distributed = types.ModuleType("torch.distributed")
    fake_distributed.is_initialized = lambda: True
    fake_distributed.destroy_process_group = lambda: events.append("destroy_pg")
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_distributed
    fake_torch.cuda = SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))

    previous = {name: sys.modules.get(name) for name in (module_name, "torch", "torch.distributed")}
    sys.modules[module_name] = mega_moe
    sys.modules["torch"] = fake_torch
    sys.modules["torch.distributed"] = fake_distributed
    try:
        worker_cls = _extract_class(
            _ROOT / "tensorrt_llm/executor/worker.py",
            "GenerationExecutorWorker",
            {"shutdown"},
            {
                "logger_debug": lambda *args, **kwargs: None,
                "mpi_rank": lambda: 0,
                "mpi_comm": lambda: SimpleNamespace(allgather=lambda value: [value]),
                "sys": sys,
                "_MEGA_MOE_DEEPGEMM_MODULE": module_name,
                "gc": SimpleNamespace(collect=lambda: events.append("gc")),
            },
        )
        engine = SimpleNamespace(
            shutdown_all_ranks=True,
            can_enqueue_requests=lambda: False,
            shutdown=lambda: events.append("engine_shutdown"),
            terminal_cleanup=lambda *, local_safe: events.append(("terminal_cleanup", local_safe)),
        )
        worker = worker_cls()
        worker.doing_shutdown = False
        worker.engine = engine
        worker.llm_args = SimpleNamespace(backend="pytorch")
        worker._handle_background_error = lambda: events.append("background")

        with pytest.raises(RuntimeError, match="symm release failed"):
            worker.shutdown()
        assert events == [
            "engine_shutdown",
            ("terminal_cleanup", True),
            "release_symm",
        ]
        assert worker.engine is engine
        assert not worker.doing_shutdown

        worker.shutdown()
        assert events == [
            "engine_shutdown",
            ("terminal_cleanup", True),
            "release_symm",
            ("terminal_cleanup", True),
            "release_symm",
            "destroy_pg",
            "gc",
            "empty_cache",
            "background",
        ]
        assert worker.engine is None
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_mpi_worker_peer_cleanup_failure_blocks_later_collectives():
    events = []
    outcomes = [
        [None, "RuntimeError: peer cleanup failed"],
        [None, None],
        [None, None],
    ]

    class Comm:
        def allgather(self, value):
            events.append(("allgather", value))
            return outcomes.pop(0)

    fake_distributed = types.ModuleType("torch.distributed")
    fake_distributed.is_initialized = lambda: True
    fake_distributed.destroy_process_group = lambda: events.append("destroy_pg")
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_distributed
    fake_torch.cuda = SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))

    previous = {name: sys.modules.get(name) for name in ("torch", "torch.distributed")}
    sys.modules["torch"] = fake_torch
    sys.modules["torch.distributed"] = fake_distributed
    try:
        worker_cls = _extract_class(
            _ROOT / "tensorrt_llm/executor/worker.py",
            "GenerationExecutorWorker",
            {"shutdown"},
            {
                "logger_debug": lambda *args, **kwargs: None,
                "mpi_rank": lambda: 0,
                "mpi_comm": lambda: Comm(),
                "sys": sys,
                "_MEGA_MOE_DEEPGEMM_MODULE": "missing_peer_failure_megamoe",
                "gc": SimpleNamespace(collect=lambda: events.append("gc")),
            },
        )
        engine = SimpleNamespace(
            shutdown_all_ranks=True,
            can_enqueue_requests=lambda: False,
            shutdown=lambda: events.append("engine_shutdown"),
            terminal_cleanup=lambda *, local_safe: events.append(("terminal_cleanup", local_safe)),
        )
        worker = worker_cls()
        worker.doing_shutdown = False
        worker.engine = engine
        worker.llm_args = SimpleNamespace(backend="pytorch")
        worker._handle_background_error = lambda: events.append("background")

        with pytest.raises(RuntimeError, match="rank 1.*peer cleanup failed"):
            worker.shutdown()
        assert events == [
            "engine_shutdown",
            ("terminal_cleanup", True),
            ("allgather", None),
        ]
        assert worker.engine is engine
        assert not worker.doing_shutdown

        worker.shutdown()
        assert events == [
            "engine_shutdown",
            ("terminal_cleanup", True),
            ("allgather", None),
            ("terminal_cleanup", True),
            ("allgather", None),
            ("allgather", None),
            "destroy_pg",
            "gc",
            "empty_cache",
            "background",
        ]
        assert outcomes == []
        assert worker.engine is None
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
