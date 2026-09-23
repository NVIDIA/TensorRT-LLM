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
import os
import select
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from prometheus_client import CollectorRegistry, multiprocess

from tensorrt_llm.metrics import batch_metrics

pytestmark = pytest.mark.cpu_only


def _batch(context=0, generation=0, dummy_context=0, dummy_generation=0):
    return SimpleNamespace(
        context_requests=[SimpleNamespace(is_dummy=False) for _ in range(context)]
        + [SimpleNamespace(is_dummy=True) for _ in range(dummy_context)],
        generation_requests=[SimpleNamespace(is_dummy=False) for _ in range(generation)]
        + [SimpleNamespace(is_dummy=True) for _ in range(dummy_generation)],
        batch_size=context + generation + dummy_context + dummy_generation,
    )


@pytest.mark.parametrize("filter_dummies", [False, True])
def test_batch_changes_and_idle(filter_dummies):
    metrics = batch_metrics.BatchMetrics("test_model", "pytorch", rank=0)
    assert metrics._batch_size._value.get() == 0
    for context, generation in [(4, 0), (2, 7), (0, 8), (0, 1), (0, 0)]:
        metrics.update(_batch(context, generation), filter_dummies=filter_dummies)
        assert metrics._batch_size._value.get() == context + generation
    metrics.update(_batch(generation=3), filter_dummies=filter_dummies)
    metrics.reset()
    assert metrics._batch_size._value.get() == 0


def test_attention_dp_dummies_are_not_user_work():
    metrics = batch_metrics.BatchMetrics("test_model", "pytorch", rank=3)
    metrics.update(
        _batch(context=2, generation=5, dummy_context=1, dummy_generation=3),
        filter_dummies=True,
    )
    assert metrics._batch_size._value.get() == 7
    metrics.update(_batch(dummy_generation=1), filter_dummies=True)
    assert metrics._batch_size._value.get() == 0


def _reap_workers(workers):
    """Reap every child, including a child that ignores stdin closure."""
    failures = []
    for worker in workers:
        try:
            _, stderr = worker.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            _, stderr = worker.communicate()
            failures.append(f"worker {worker.pid} timed out: {stderr}")
        else:
            if worker.returncode != 0:
                failures.append(f"worker {worker.pid} exited {worker.returncode}: {stderr}")
    assert not failures, "\n".join(failures)


def test_worker_timeout_still_reaps_remaining_children():
    stuck, healthy = Mock(pid=1), Mock(pid=2, returncode=0)
    stuck.communicate.side_effect = [subprocess.TimeoutExpired("worker", 10), ("", "stuck")]
    healthy.communicate.return_value = ("", "")
    with pytest.raises(AssertionError, match="timed out"):
        _reap_workers([stuck, healthy])
    stuck.kill.assert_called_once_with()
    assert stuck.communicate.call_count == 2
    healthy.communicate.assert_called_once_with(timeout=10)


def test_multiprocess_scrape_keeps_workers_and_ranks_separate(tmp_path):
    # Use fresh interpreters: the Prometheus client chooses mmap storage at
    # import time. Load only the module under test, avoiding CUDA initialization
    # in these CPU-only metric publishers.
    script = """
import importlib.util
import sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("batch_metrics", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metrics = module.BatchMetrics("model", "pytorch", int(sys.argv[2]))
for command in sys.stdin:
    value = int(command)
    if value == 0:
        metrics.reset()
    else:
        metrics.update(SimpleNamespace(batch_size=value), filter_dummies=False)
    print("updated", flush=True)
"""
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    workers = []
    try:
        for rank in (0, 1, 0):
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-u", "-c", script, batch_metrics.__file__, str(rank)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
    except BaseException:
        _reap_workers(workers)
        raise
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(tmp_path))

    def update_and_scrape(values):
        for worker, value in zip(workers, values):
            worker.stdin.write(f"{value}\n")
            worker.stdin.flush()
            assert select.select([worker.stdout], [], [], 10)[0], "worker did not reply"
            assert worker.stdout.readline().strip() == "updated"
        samples = [
            sample
            for metric in registry.collect()
            for sample in metric.samples
            if sample.name == "trtllm_scheduled_batch_size"
        ]
        assert len(samples) == len(workers)
        actual = {(s.labels["pid"], s.labels["rank"]): s.value for s in samples}
        expected = {
            (str(worker.pid), str(rank)): value
            for worker, rank, value in zip(workers, (0, 1, 0), values)
        }
        assert actual == expected

    try:
        update_and_scrape((2, 7, 11))
        update_and_scrape((6, 3, 4))
        update_and_scrape((0, 0, 0))
    finally:
        _reap_workers(workers)


@pytest.mark.parametrize("enable_iter_perf_stats", [False, True])
def test_executor_publishes_without_iteration_stats_and_clears_before_idle(
    enable_iter_perf_stats,
):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    executor._batch_metrics = batch_metrics.BatchMetrics("model", "pytorch", rank=0)
    executor.enable_attention_dp = False
    executor.enable_iter_perf_stats = enable_iter_perf_stats
    executor._update_batch_metrics(_batch(context=2, generation=5))
    assert executor._batch_metrics._batch_size._value.get() == 7

    executor.control_requests = []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0)
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    executor.executor_request_queue = Mock()
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.return_value = ([], {})
    executor._handle_special_queue_items = Mock(return_value=[])
    waiting_queue = MagicMock()
    waiting_queue.__len__.return_value = 0

    def block_until_new_work(timeout):
        assert timeout is None
        assert executor._batch_metrics._batch_size._value.get() == 0
        return []

    executor.executor_request_queue.get_from_request_queue.side_effect = block_until_new_work
    executor._fetch_and_enqueue_requests(waiting_queue, 0)
    executor.executor_request_queue.get_from_request_queue.assert_called_once_with(None)


def test_repeated_metrics_directory_setup_preserves_executor_files(monkeypatch):
    from pathlib import Path

    from tensorrt_llm import _utils

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setattr(_utils, "prometheus_multiproc_dir", None, raising=False)
    _utils.set_prometheus_multiproc_dir()
    owner = _utils.prometheus_multiproc_dir
    try:
        metric_file = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"]) / "executor.db"
        metric_file.write_text("preserved")
        _utils.set_prometheus_multiproc_dir()
        assert _utils.prometheus_multiproc_dir is owner
        assert metric_file.read_text() == "preserved"
    finally:
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        owner.cleanup()


@pytest.mark.parametrize("configured_model", ["local", "org/hub-model"])
@pytest.mark.parametrize("served_model", [None, "public-alias"])
def test_scrape_uses_server_labels(tmp_path, monkeypatch, configured_model, served_model):
    model = str(tmp_path / "model") if configured_model == "local" else configured_model
    if configured_model == "local":
        (tmp_path / "model").mkdir()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
    script = """
import importlib.util
import sys
from types import SimpleNamespace
from prometheus_client import Gauge
spec = importlib.util.spec_from_file_location("batch_metrics", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for model, backend, size in [(sys.argv[2], "pytorch", 5), ("other", "pytorch", 9), (sys.argv[2], "unknown", 3)]:
    metric = module.BatchMetrics(model, backend, rank=2)
    metric.update(SimpleNamespace(batch_size=size), filter_dummies=False)
Gauge("existing_metric", "Existing gauge", ["model_name"], multiprocess_mode="all").labels(model_name="original").set(5)
"""
    subprocess.run(
        [sys.executable, "-c", script, batch_metrics.__file__, model],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from prometheus_client.parser import text_string_to_metric_families

    from tensorrt_llm.serve.openai_server import OpenAIServer

    model_label = served_model or ("model" if configured_model == "local" else model)
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = SimpleNamespace(args=SimpleNamespace(model=model))
    server.metrics_collector = SimpleNamespace(
        labels={"model_name": model_label, "engine_type": "pytorch"}
    )
    server.mount_metrics()
    with TestClient(server.app) as client:
        response = client.get("/prometheus/metrics")
    assert response.status_code == 200
    samples = [
        sample
        for metric in text_string_to_metric_families(response.text)
        for sample in metric.samples
    ]
    published = {
        sample.value: sample for sample in samples if sample.name == "trtllm_scheduled_batch_size"
    }
    assert len(published) == 3
    assert published[5].labels["model_name"] == model_label
    assert published[5].labels["engine_type"] == "pytorch"
    assert published[5].labels["rank"] == "2"
    assert "pid" in published[5].labels
    assert published[9].labels["model_name"] == "other"
    assert published[3].labels["engine_type"] == "unknown"
    assert published[3].labels["model_name"] == ("model" if configured_model == "local" else model)
    # The unrelated gauge deliberately shares a value with the batch metric.
    existing = [sample for sample in samples if sample.name == "existing_metric"]
    assert [(sample.labels["model_name"], sample.value) for sample in existing] == [("original", 5)]


@pytest.mark.parametrize("invalid_path", ["missing", "file", ""])
def test_supplied_metrics_directory_must_exist(tmp_path, monkeypatch, invalid_path):
    from tensorrt_llm import _utils

    path = tmp_path / invalid_path if invalid_path else ""
    if invalid_path == "file":
        path.write_text("not a directory")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(path))
    with pytest.raises(ValueError, match="PROMETHEUS_MULTIPROC_DIR"):
        _utils.set_prometheus_multiproc_dir()


def test_supplied_metrics_directory_is_logged(tmp_path, monkeypatch):
    from tensorrt_llm import _utils

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    info = Mock()
    monkeypatch.setattr(_utils.logger, "info", info)
    _utils.set_prometheus_multiproc_dir()
    assert any(str(tmp_path) in str(call) for call in info.call_args_list)


class _ReachedForward(Exception):
    """Stop a real scheduler loop at the mocked GPU boundary."""


@pytest.mark.parametrize(
    "loop_name", ["_executor_loop", "_executor_loop_overlap", "_executor_loop_pp"]
)
@pytest.mark.parametrize("can_queue", [False, True])
@pytest.mark.parametrize("enable_iter_perf_stats", [False, True])
def test_scheduler_publishes_prepared_batch(
    loop_name, can_queue, enable_iter_perf_stats, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor import py_executor

    executor = object.__new__(py_executor.PyExecutor)
    executor._batch_metrics = batch_metrics.BatchMetrics("model", "pytorch", rank=0)
    executor._batch_metrics.update(_batch(generation=11), filter_dummies=False)
    executor.device_id = 0
    executor.iter_counter = 0
    executor.enable_iter_perf_stats = enable_iter_perf_stats
    executor.enable_attention_dp = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.is_benchmark_disagg = False
    executor._pp_rebalance_drain_iters = None
    executor.is_shutdown = False
    executor.active_requests = [Mock()]
    executor.dist = SimpleNamespace(rank=0, pp_rank=0, is_last_pp_rank=False)
    executor.pp_async_broadcast_sample_state = True
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.previous_batch = None
    executor.has_previous_draft_tokens = False
    executor.hang_detector = MagicMock()
    executor._profiler = MagicMock()
    executor._disagg_coordinator = Mock()
    executor.perf_manager = Mock()
    executor.resource_manager = Mock()
    executor._can_pause_for_rebalance = Mock(return_value=False)
    executor._can_queue = Mock(return_value=(can_queue, can_queue))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    for name in (
        "_handle_control_request",
        "_pad_attention_dp_dummy_request",
        "_terminate_requests",
        "_pause_requests",
        "_wait_for_model_engine_input_copy",
        "_add_inflight_ids",
        "_prepare_disagg_gen_transmission_complete",
        "_handle_dynamic_draft_len",
        "_kv_connector_start_batch",
        "_finalize_adp_dummy_allocation",
        "_commit_kv_cache_stats",
        "_get_init_iter_stats",
        "_get_new_active_requests_queue_latency",
        "_collect_scheduled_batch_stats",
    ):
        setattr(executor, name, Mock())

    def request():
        return SimpleNamespace(is_dummy=False, is_attention_dp_dummy=False, py_batch_idx=None)

    batch = py_executor.ScheduledRequests()
    batch.context_requests_last_chunk = [request() for _ in range(2)]
    batch.generation_requests = [request() for _ in range(5)]
    executor._prepare_and_schedule_batch = Mock(return_value=(batch, None))
    executor._pp_schedule_and_propagate = Mock(return_value=(batch, [], None, None))
    # An outstanding PP microbatch does not make this a total-in-flight gauge.
    outstanding = object()
    executor.micro_batches = [None, outstanding]

    def prepare_resources(scheduled):
        assert scheduled is batch
        assert executor._batch_metrics._batch_size._value.get() == 11
        # Resource preparation may change the batch that actually runs.
        scheduled.context_requests_last_chunk = scheduled.context_requests_last_chunk[:1]
        scheduled.generation_requests = scheduled.generation_requests[:1]

    executor.resource_manager.prepare_resources.side_effect = prepare_resources
    executor._revert_gen_alloc = Mock(side_effect=_ReachedForward)
    executor.perf_manager.create_timing_events.side_effect = _ReachedForward
    executor.perf_manager.borrow_forward_timing_events.side_effect = _ReachedForward
    executor._step_scope = Mock(side_effect=_ReachedForward)
    monkeypatch.setattr(py_executor.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(py_executor.cudart, "cudaSetDevice", Mock())
    monkeypatch.setattr(py_executor.torch.cuda.nvtx, "range", MagicMock())
    monkeypatch.setattr(py_executor, "CUASSERT", Mock())
    with pytest.raises(_ReachedForward):
        getattr(executor, loop_name)()
    assert executor._batch_metrics._batch_size._value.get() == (2 if can_queue else 0)
    assert executor.resource_manager.prepare_resources.call_count == int(can_queue)
    assert executor.micro_batches[1] is outstanding


@pytest.mark.parametrize("error", [None, RuntimeError("forward failed"), KeyboardInterrupt()])
def test_event_loop_cleanup_clears_published_batch(monkeypatch, error):
    from tensorrt_llm._torch.pyexecutor import py_executor

    executor = object.__new__(py_executor.PyExecutor)
    executor._batch_metrics = batch_metrics.BatchMetrics("model", "pytorch", rank=0)
    executor._batch_metrics.update(_batch(generation=7), filter_dummies=False)
    executor.garbage_collection_gen0_threshold = None
    executor.dist = SimpleNamespace(world_size=1)
    executor._event_loop_error_delivered = None
    executor.event_loop = Mock(side_effect=error)
    monkeypatch.setattr(py_executor, "start_rank_crash_kill_watchdog", Mock(return_value=None))
    monkeypatch.setattr(py_executor, "hard_kill_on_rank_crash", Mock())

    def cleanup():
        assert executor._batch_metrics._batch_size._value.get() == 0

    executor._executor_loop_cleanup = Mock(side_effect=cleanup)
    if error is None:
        executor._event_loop_wrapper()
    else:
        with pytest.raises(type(error)):
            executor._event_loop_wrapper()
    executor._executor_loop_cleanup.assert_called_once_with()


@pytest.mark.parametrize(
    "enabled,directory_exists,multiprocess_mode",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_executor_initializes_metrics_only_with_multiprocess_storage(
    tmp_path, monkeypatch, enabled, directory_exists, multiprocess_mode
):
    from prometheus_client import values

    from tensorrt_llm._torch.pyexecutor import py_executor

    metrics_dir = tmp_path if directory_exists else tmp_path / "missing"
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
    monkeypatch.setattr(
        values, "ValueClass", values.MultiProcessValue() if multiprocess_mode else values.MutexValue
    )
    monkeypatch.setattr(py_executor.torch.cuda, "current_device", Mock(return_value=0))
    monkeypatch.setattr(py_executor.torch.cuda, "Stream", Mock())
    # Stop after the metric initialization in the real constructor; remaining
    # executor setup requires a model, resource managers, and worker threads.
    monkeypatch.setattr(py_executor, "PerfMetricsManager", Mock(side_effect=_ReachedForward))
    warning = Mock()
    monkeypatch.setattr(py_executor.logger, "warning", warning)
    executor = object.__new__(py_executor.PyExecutor)
    args = SimpleNamespace(
        max_stats_len=10,
        max_num_tokens=10,
        print_iter_log=False,
        enable_iter_perf_stats=False,
        enable_iter_req_stats=False,
        return_perf_metrics=enabled,
        stream_interval=1,
        model="org/model",
        backend=None,
    )
    engine = SimpleNamespace(llm_args=args, enable_attention_dp=False)
    with pytest.raises(_ReachedForward):
        executor.__init__(
            Mock(),
            Mock(),
            engine,
            Mock(),
            SimpleNamespace(rank=2),
            max_num_sequences=8,
            start_worker=False,
        )
    expected = enabled and directory_exists and multiprocess_mode
    assert (executor._batch_metrics is not None) == expected
    if expected:
        executor._batch_metrics.update(_batch(generation=3), filter_dummies=False)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=str(tmp_path))
        sample = next(sample for metric in registry.collect() for sample in metric.samples)
        assert sample.name == "trtllm_scheduled_batch_size"
        assert sample.labels["engine_type"] == "unknown"
        assert sample.labels["rank"] == "2"
        assert sample.value == 3
    elif enabled:
        assert warning.call_count == 1
        reason = "PROMETHEUS_MULTIPROC_DIR" if not directory_exists else "imported before"
        assert reason in warning.call_args.args[0]
    else:
        warning.assert_not_called()


def test_llm_constructor_initializes_multiprocess_storage_before_model_build(tmp_path):
    script = """
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
assert "PROMETHEUS_MULTIPROC_DIR" not in os.environ
from tensorrt_llm import LLM
from tensorrt_llm.metrics.batch_metrics import BatchMetrics

class ModelBuildReached(Exception):
    pass

def build_model(llm):
    # This runs at the real model-build boundary, after public argument parsing.
    from prometheus_client import CollectorRegistry, multiprocess, values
    assert values.ValueClass._multiprocess
    metric = BatchMetrics(str(llm.args.model), llm.args.backend, rank=0)
    metric.update(SimpleNamespace(batch_size=4), filter_dummies=False)
    directory = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])
    assert list(directory.glob("gauge_all_*.db"))
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    sample = next(s for m in registry.collect() for s in m.samples)
    assert sample.name == "trtllm_scheduled_batch_size"
    assert sample.value == 4
    print("constructor metric scrape verified", flush=True)
    raise ModelBuildReached()

with patch.object(LLM, "_build_model", build_model):
    try:
        LLM(model=sys.argv[1], skip_tokenizer_init=True, return_perf_metrics=True)
    except ModelBuildReached:
        pass
    else:
        raise AssertionError("model-build boundary was not reached")
"""
    env = {key: value for key, value in os.environ.items() if key != "PROMETHEUS_MULTIPROC_DIR"}
    env["TLLM_TELEMETRY_OPT_OUT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "constructor metric scrape verified" in result.stdout
