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


def test_multiprocess_scrape_keeps_workers_and_ranks_separate(tmp_path):
    # Use fresh interpreters: the Prometheus client chooses mmap storage at
    # import time. Load only the module under test, avoiding CUDA initialization
    # in these CPU-only metric publishers.
    script = """
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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
    workers = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", script, batch_metrics.__file__, str(rank)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1, 0)
    ]
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(tmp_path))

    def update_and_scrape(values):
        for worker, value in zip(workers, values):
            worker.stdin.write(f"{value}\n")
            worker.stdin.flush()
            assert worker.stdout.readline().strip() == "updated"
        samples = [
            sample
            for metric in registry.collect()
            for sample in metric.samples
            if sample.name == "trtllm_inflight_batch_size"
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
        for worker in workers:
            worker.communicate(timeout=10)
        for worker in workers:
            assert worker.returncode == 0


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
        owner.cleanup()
