# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify launcher-owned shutdown and replacement fencing with a real engine."""

import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import grpc
import psutil
import pytest
import torch

from tensorrt_llm.grpc.openengine.bindings import (
    generation_pb2,
    lifecycle_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

pytestmark = pytest.mark.threadleak(enabled=False)
_OWNER_ENV = "OPENENGINE_SHUTDOWN_TEST_OWNER"

_LAUNCHER = """
import os
import sys
from tensorrt_llm.grpc.openengine import server

original = server.OpenEngineControlServicer.GetServerInfo
async def info(self, request, context):
    result = await original(self, request, context)
    result.extra["test_frontend_id"] = int(os.environ.get("TLLM_EXECUTOR_FRONTEND_ID", "0"))
    return result
server.OpenEngineControlServicer.GetServerInfo = info
if __name__ == "__main__":
    server.launch_server("127.0.0.1", int(sys.argv[2]), {
        "model": sys.argv[1], "backend": "pytorch", "num_serve_frontends": int(sys.argv[3]),
        "max_batch_size": 2, "max_seq_len": 2048, "max_num_tokens": 2048,
        "kv_cache_config": {"free_gpu_memory_fraction": 0.1}, "cuda_graph_config": None,
    })
"""


def _owned_workers(owner):
    """Find this launch's workers even after detachment or launcher exit."""
    workers = []
    for process in psutil.process_iter():
        try:
            if process.uids().real == os.getuid() and process.environ().get(_OWNER_ENV) == owner:
                workers.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return workers


@contextmanager
def _engine(tmp_path, model, port, frontends, incarnation):
    script = tmp_path / "launch.py"
    script.write_text(_LAUNCHER)
    log_path = tmp_path / f"engine-{incarnation}.log"
    owner = uuid.uuid4().hex
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), str(model), str(port), str(frontends)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, _OWNER_ENV: owner},
        )
    workers = {}

    def collect_workers():
        # Keep captured descendants even after reparenting, and discover workers
        # created or detached since the previous snapshot through the marker.
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in [*descendants, *_owned_workers(owner)]:
            if child.pid != process.pid:
                workers[child.pid] = child
        return list(workers.values())

    channel = None
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert process.poll() is None, log_path.read_text()
            channel = grpc.insecure_channel(
                f"127.0.0.1:{port}", options=[("grpc.use_local_subchannel_pool", 1)]
            )
            control = openengine_pb2_grpc.ControlStub(channel)
            try:
                info = control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=1)
                ready = control.Health(lifecycle_pb2.HealthRequest(), timeout=1)
                if (
                    int(info.extra["test_frontend_id"]) == frontends - 1
                    and ready.state == lifecycle_pb2.HEALTH_STATE_READY
                ):
                    break
            except grpc.RpcError:
                pass
            channel.close()
            channel = None
            time.sleep(0.1)
        else:
            pytest.fail(f"Engine did not become ready: {log_path.read_text()}")
        collect_workers()
        yield process, channel, info
        assert process.wait(timeout=60) == 0, log_path.read_text()
        # MPI helpers can observe launcher exit just after wait() returns.
        # Require every owned worker/helper to exit within a bounded window.
        deadline = time.monotonic() + 10
        while True:
            alive = [
                child
                for child in collect_workers()
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE
            ]
            if not alive:
                break
            assert time.monotonic() < deadline, f"Workers survived shutdown: {alive}"
            time.sleep(0.05)
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
    except BaseException:
        print(log_path.read_text())
        raise
    finally:
        if channel is not None:
            channel.close()
        collect_workers()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for child in collect_workers():
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(list(workers.values()), timeout=3)


@pytest.mark.parametrize("frontends", [1, 2])
def test_shutdown_active_engine_and_fence_replacement(tmp_path, frontends):
    """Shutdown via an attached frontend must reap workers and fence old retries."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU and a matching TensorRT-LLM runtime")
    model = Path(os.environ["LLM_MODELS_ROOT"]) / "Qwen3/Qwen3-0.6B"
    assert model.is_dir(), f"Missing qualification model: {model}"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    previous_id = None
    for incarnation in range(2):
        with _engine(tmp_path, model, port, frontends, incarnation) as (_, channel, info):
            control = openengine_pb2_grpc.ControlStub(channel)
            if previous_id is not None:
                assert previous_id != info.instance_id
                with pytest.raises(grpc.RpcError) as stale:
                    control.Shutdown(
                        lifecycle_pb2.ShutdownRequest(expected_instance_id=previous_id), timeout=5
                    )
                assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                assert (
                    control.Health(lifecycle_pb2.HealthRequest(), timeout=5).state
                    == lifecycle_pb2.HEALTH_STATE_READY
                )
            stream = openengine_pb2_grpc.InferenceStub(channel).Generate(
                generation_pb2.GenerateRequest(
                    request_id="active",
                    model="test",
                    prompt="Count upwards from one:",
                    stopping=generation_pb2.StoppingOptions(max_tokens=1900, ignore_eos=True),
                ),
                timeout=60,
            )
            for response in stream:
                assert not response.HasField("error"), response
                if response.HasField("token"):
                    break
            else:
                pytest.fail("Engine produced no token before shutdown")
            assert control.GetLoad(server_pb2.GetLoadRequest(), timeout=5).running_requests > 0
            response = control.Shutdown(
                lifecycle_pb2.ShutdownRequest(expected_instance_id=info.instance_id), timeout=5
            )
            assert response.status == lifecycle_pb2.SHUTDOWN_STATUS_ACCEPTED
            assert response.instance_id == info.instance_id
            stream.cancel()
            previous_id = info.instance_id


@pytest.mark.cpu_only
def test_engine_fixture_reaps_detached_worker_on_startup_failure(tmp_path, monkeypatch):
    """A failed model startup must not leave a separately sessioned worker alive."""
    monkeypatch.setitem(
        globals(),
        "_LAUNCHER",
        """
import subprocess
import sys
from pathlib import Path
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
)
Path(sys.argv[1]).write_text(str(child.pid))
raise SystemExit(17)
""",
    )
    marker = tmp_path / "worker-pid"
    try:
        with pytest.raises(AssertionError):
            with _engine(tmp_path, marker, 0, 1, "failed-startup"):
                pytest.fail("A failed launcher cannot become ready")
        try:
            worker = psutil.Process(int(marker.read_text()))
        except psutil.NoSuchProcess:
            return
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
    finally:
        if marker.exists():
            try:
                worker = psutil.Process(int(marker.read_text()))
                if worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE:
                    worker.kill()
                    worker.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass


@pytest.mark.cpu_only
def test_engine_fixture_rejects_worker_created_after_readiness(tmp_path, monkeypatch):
    """Qualification must fail before emergency cleanup kills a leaked worker."""
    monkeypatch.setitem(
        globals(),
        "_LAUNCHER",
        """
import subprocess
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
while not (root / "start-worker").exists():
    time.sleep(0.01)
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
)
(root / "worker-pid").write_text(str(child.pid))
""",
    )
    monkeypatch.setattr(
        openengine_pb2_grpc,
        "ControlStub",
        lambda channel: SimpleNamespace(
            GetServerInfo=lambda *args, **kwargs: SimpleNamespace(extra={"test_frontend_id": 0}),
            Health=lambda *args, **kwargs: SimpleNamespace(state=lifecycle_pb2.HEALTH_STATE_READY),
        ),
    )
    with pytest.raises(AssertionError, match="Workers survived shutdown"):
        with _engine(tmp_path, tmp_path, 0, 1, "late-worker"):
            (tmp_path / "start-worker").touch()
    try:
        worker = psutil.Process(int((tmp_path / "worker-pid").read_text()))
    except psutil.NoSuchProcess:
        return
    assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
