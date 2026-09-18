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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


class BatchMetrics:
    """Publish a rank's latest scheduled batch through the serving metrics directory.

    The caller must initialize PROMETHEUS_MULTIPROC_DIR before importing the
    Prometheus client. The multiprocess collector adds a pid label, keeping
    independent executors (including prefill and decode workers) separate.
    """

    def __init__(self, model_name: str, engine_type: str, rank: int):
        # Keep this lazy: prometheus_client chooses its storage at import time.
        from prometheus_client import Gauge

        self._batch_size = Gauge(
            "trtllm_inflight_batch_size",
            "Number of real context and generation requests in the latest scheduled "
            "batch on this executor rank; zero while idle.",
            labelnames=("model_name", "engine_type", "rank"),
            registry=None,
            multiprocess_mode="all",
        ).labels(model_name=model_name, engine_type=engine_type, rank=str(rank))
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
