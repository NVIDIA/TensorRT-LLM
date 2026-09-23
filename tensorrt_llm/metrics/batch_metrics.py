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
"""Scheduler metrics that do not require iteration statistics."""

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry
    from prometheus_client.core import Metric

    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


SCHEDULED_BATCH_SIZE = "trtllm_scheduled_batch_size"


def _model_label(model: str) -> str:
    """Use the serving convention for local directories and Hugging Face IDs."""
    path = Path(model)
    return path.name if path.is_dir() else model


class BatchMetrics:
    """Publish a rank's latest scheduled batch through the serving metrics directory.

    The caller must initialize PROMETHEUS_MULTIPROC_DIR before importing the
    Prometheus client. The multiprocess collector adds a pid label, keeping
    independent executors (including prefill and decode workers) separate.
    """

    def __init__(self, model_name: str, engine_type: str, rank: int):
        """Create a scheduled-request gauge for one executor rank."""
        # Keep this lazy: prometheus_client chooses its storage at import time.
        from prometheus_client import Gauge

        # Retain per-process series. liveall alone cannot detect dead workers;
        # process-wide reaping must belong to the worker supervisor.
        self._batch_size = Gauge(
            SCHEDULED_BATCH_SIZE,
            "Number of real context and generation requests in the latest scheduled "
            "batch on this executor rank; zero when no batch is submitted. "
            "Under pipeline parallelism, other microbatches may still be running.",
            labelnames=("model_name", "engine_type", "rank"),
            registry=None,
            multiprocess_mode="all",
        ).labels(model_name=_model_label(model_name), engine_type=engine_type, rank=str(rank))
        self.reset()

    def update(self, scheduled_batch: "ScheduledRequests", *, filter_dummies: bool) -> None:
        """Record the batch before forward can mutate its request lists."""
        if filter_dummies:
            batch_size = sum(
                not request.is_dummy
                for requests in (
                    scheduled_batch.context_requests,
                    scheduled_batch.generation_requests,
                )
                for request in requests
            )
        else:
            batch_size = scheduled_batch.batch_size
        self._batch_size.set(batch_size)

    def reset(self) -> None:
        """Clear the last batch when the executor is idle or exits."""
        self._batch_size.set(0)


class BatchMetricsCollector:
    """Collect multiprocess metrics using the server's batch-metric labels.

    Served-model aliases belong to the server, which can be created after the
    executor. Relabel only its model/backend's scheduled-batch samples, keeping
    other models and all existing metrics unchanged.
    """

    def __init__(
        self, registry: "CollectorRegistry", *, model_name: str, labels: dict[str, str]
    ) -> None:
        from prometheus_client import multiprocess

        self._collector = multiprocess.MultiProcessCollector(None)
        self._source_labels = {
            "model_name": _model_label(model_name),
            "engine_type": labels["engine_type"],
        }
        self._labels = dict(labels)
        registry.register(self)

    def collect(self) -> Iterator["Metric"]:
        """Apply the serving identity while retaining worker PID and rank."""
        for metric in self._collector.collect():
            if metric.name == SCHEDULED_BATCH_SIZE:
                for sample in metric.samples:
                    if all(
                        sample.labels.get(key) == value
                        for key, value in self._source_labels.items()
                    ):
                        sample.labels.update(self._labels)
            yield metric
