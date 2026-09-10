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

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker


class _Args:
    sleep_config = object()


def _make_worker(state="running", parked_tags=None):
    worker = object.__new__(RayGPUWorker)
    worker.doing_shutdown = True
    worker.llm_args = _Args()
    worker.engine = MagicMock()
    worker.engine.get_memory_status.return_value = {
        "state": state,
        "parked_tags": parked_tags or [],
    }
    return worker


def _run_unwrapped(method, worker, tags):
    getattr(RayGPUWorker, method).__wrapped__(worker, tags)


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sleep", "release_with_tag"),
        ("wakeup", "materialize_with_tag"),
    ],
)
def test_ray_worker_publishes_persistent_admission_transition(method, operation):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize"),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
    ):
        _run_unwrapped(method, worker, ["model"])

    getattr(worker.engine, f"begin_{method}_transition").assert_called_once()
    getattr(worker.engine, f"complete_{method}_transition").assert_called_once_with()


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sleep", "release_with_tag"),
        ("wakeup", "materialize_with_tag"),
    ],
)
def test_ray_worker_fails_closed_after_mutation(method, operation):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}"),
        patch(
            "tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize",
            side_effect=[None, RuntimeError("post-mutation failure")],
        ),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="post-mutation failure"),
    ):
        _run_unwrapped(method, worker, ["model"])

    worker.engine.fail_sleep_wakeup_transition.assert_called_once_with()
    getattr(worker.engine, f"abort_{method}_transition").assert_called_once()
