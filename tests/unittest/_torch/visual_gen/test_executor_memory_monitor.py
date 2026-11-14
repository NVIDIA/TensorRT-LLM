# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A VisualGen worker registers with the GPU memory monitor while it loads,
reports the out-of-memory errors it catches, and unregisters when it stops."""

import contextlib
import queue
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import (
    claims,
    providers,
    settings,
    startup,
    tagging,
    triggers,
)
from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm._torch.visual_gen.memory_monitor import WorkerMemoryMonitor

_TORCH_OOM = (
    "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.10 GiB"
)


def _reset() -> None:
    providers.clear_registered_providers()
    settings.configure(None)
    startup.clear()
    claims.clear_claims()
    if tagging.tagging_window_active():
        tagging.stop_tagging_window()
    tagging.set_frozen_map(tagging.TagSnapshot(supported=False))
    triggers.uninstall_oom_reporter()


@pytest.fixture(autouse=True)
def _monitor_state_is_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(settings.ENV_VAR, raising=False)
    monkeypatch.delenv(settings.RANK_ENV_VAR, raising=False)
    monkeypatch.delenv(settings.PERIODIC_INTERVAL_VAR, raising=False)
    _reset()
    yield
    _reset()


def _executor(monitor: object) -> executor_module.DiffusionExecutor:
    """A rank-0 executor with a pipeline whose inference raises a device OOM."""

    def run_inference(req):
        raise torch.OutOfMemoryError(_TORCH_OOM)

    executor = object.__new__(executor_module.DiffusionExecutor)
    executor.rank = 0
    executor.device_id = 0
    executor.response_queue = queue.Queue()
    executor.memory_monitor = monitor
    executor.pipeline = SimpleNamespace(
        default_generation_params={},
        extra_param_specs={},
        prepare_request=lambda req: None,
        request_warmup_cache_key=lambda req: (None,),
        _warmed_up_shapes=set(),
        run_inference=run_inference,
        classify_request_failure=lambda exc: "server",
    )
    return executor


@pytest.mark.cpu_only
def test_a_failed_request_is_reported_after_its_error_response(monkeypatch):
    """The OOM a request dies of reaches the monitor once its error response is queued."""
    executor = _executor(WorkerMemoryMonitor(device_id=0, rank=0, world_size=1))
    seen = []
    monkeypatch.setattr(
        triggers,
        "notify_error",
        lambda message=None, *, error=None, **kw: seen.append(
            (type(error), executor.response_queue.qsize())
        ),
    )
    request = SimpleNamespace(request_id=7, params=SimpleNamespace(extra_params=None))

    executor.process_request(request)

    response = executor.response_queue.get_nowait()
    assert response.request_id == 7 and "CUDA out of memory" in response.error_msg
    assert seen == [(torch.OutOfMemoryError, 1)]


@pytest.mark.cpu_only
def test_loading_registers_the_pipeline_then_arms(monkeypatch):
    """The load runs inside the monitor's startup, names the weights, then arms."""
    events = []

    class _Monitor:
        @contextlib.contextmanager
        def startup(self):
            events.append("startup")
            yield
            events.append("loaded")

        def claim_weights(self, pipeline):
            events.append(("claim", pipeline))

        def arm(self):
            events.append("arm")

    pipeline = torch.nn.Linear(2, 2)

    class _Loader:
        def __init__(self, args, device):
            pass

        def load(self, skip_warmup):
            events.append("load")
            return pipeline

    monkeypatch.setattr(executor_module, "PipelineLoader", _Loader)
    monkeypatch.setattr(executor_module.dist, "barrier", lambda: None)
    executor = object.__new__(executor_module.DiffusionExecutor)
    executor.rank = 1
    executor.device_id = 0
    executor.memory_monitor = _Monitor()
    executor.visual_gen_args = SimpleNamespace(compilation_config=SimpleNamespace(skip_warmup=True))

    executor._load_pipeline()

    assert events == ["startup", "load", ("claim", pipeline), "loaded", "arm"]


@pytest.mark.cpu_only
def test_the_worker_is_one_monitored_executor_from_load_to_shutdown():
    """Loading registers the worker, arming follows the load, and release undoes both."""
    monitor = WorkerMemoryMonitor(device_id=None, rank=0, world_size=1)
    with monitor.startup():
        assert providers.current_executor() is not None
    assert monitor.monitored and providers.monitor_held()
    monitor.arm()
    assert triggers._armed is not None, "the default mode arms OOM reporting once loaded"

    monitor.release()
    monitor.release()
    assert not providers.monitor_held()
    assert triggers._armed is None


@pytest.mark.cpu_only
def test_a_worker_started_next_to_a_monitored_executor_runs_without_the_monitor():
    """The monitor follows one executor per process; a second one is not armed."""
    held = providers.begin_executor()
    try:
        monitor = WorkerMemoryMonitor(device_id=None, rank=0, world_size=1)
        with monitor.startup():
            pass
        assert not monitor.monitored
        monitor.arm()
        assert triggers._armed is None
        monitor.release()
        assert providers.monitor_held(), "the first executor still holds the monitor"
    finally:
        providers.end_executor(held)


@pytest.mark.cpu_only
def test_a_failed_load_leaves_no_registered_worker():
    monitor = WorkerMemoryMonitor(device_id=None, rank=0, world_size=1)
    with pytest.raises(RuntimeError, match="checkpoint"):
        with monitor.startup():
            raise RuntimeError("missing checkpoint key")
    assert not providers.monitor_held()
    monitor.arm()
    assert triggers._armed is None, "a worker that failed to load is not armed"


@pytest.mark.cpu_only
def test_a_stopping_worker_releases_the_monitor_even_when_serving_raised(monkeypatch):
    """``run_diffusion_worker`` releases the monitor whichever way serving ends."""
    released = []

    class _Executor:
        def __init__(self, **kwargs):
            self.pipeline = None
            self.memory_monitor = SimpleNamespace(release=lambda: released.append(True))

        def serve_forever(self):
            raise RuntimeError("serving failed")

    for var in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.setenv(var, "0")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(executor_module, "init_torch_distributed", lambda **kwargs: None)
    monkeypatch.setattr(executor_module, "DiffusionExecutor", _Executor)

    with pytest.raises(RuntimeError, match="serving failed"):
        executor_module.run_diffusion_worker(
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=0,
            request_queue_addr=None,
            response_queue_addr=None,
            visual_gen_args=None,
        )
    assert released == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_full_mode_records_the_load_and_names_the_pipeline_weights(monkeypatch):
    """In full mode the load is a startup step, and the pipeline's weights are named."""
    monkeypatch.setenv(settings.ENV_VAR, "1")
    device = torch.cuda.current_device()
    monitor = WorkerMemoryMonitor(device_id=device, rank=0, world_size=1)
    with monitor.startup():
        pipeline = torch.nn.Linear(4096, 4096, device="cuda")
        monitor.claim_weights(pipeline)
        # A buffer the pipeline keeps outside its parameters, e.g. a workspace.
        workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    try:
        monitor.arm()
        report = triggers.report_now(tag="test", rank=0, save=False)
        assert report is not None
        assert "Model weights" in report
        by_label = tagging.live_bytes_by_tag(tagging.get_frozen_map())
        assert by_label.get("WEIGHTS", 0) >= pipeline.weight.nbytes
        assert by_label.get("model", 0) >= workspace.nbytes, "the load is the 'model' step"
    finally:
        monitor.release()
    assert not providers.monitor_held()
