# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from collections import defaultdict
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.models.modeling_utils import KvCacheConnectorConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

KvConnectorScheduler = MagicMock()
KvConnectorWorker = MagicMock()


def init_connector_classes():
    KvConnectorScheduler.reset_mock()
    KvConnectorWorker.reset_mock()

    scheduler = MagicMock()
    worker = MagicMock()

    KvConnectorScheduler.return_value = scheduler
    KvConnectorWorker.return_value = worker

    return scheduler, worker


# Makes sure everything is called in the right order.
class CallTimeMonitor:

    def __init__(self):
        self.call_times = defaultdict(list)
        self.counter = 0

    def monitor_fn(self, mock_fn, name):

        def wrapper(*args, **kwargs):
            self.call_times[name].append(self.counter)
            self.counter += 1

            return DEFAULT

        mock_fn.side_effect = wrapper

    def __getitem__(self, name):
        return self.call_times[name]


@pytest.fixture
def connector():
    with patch("tensorrt_llm._torch.pyexecutor.py_executor_creator.importlib"
               ) as importlib_mock:
        mock_scheduler = MagicMock()
        mock_worker = MagicMock()

        importlib_mock.import_module.return_value.KvConnectorScheduler.return_value = mock_scheduler
        importlib_mock.import_module.return_value.KvConnectorWorker.return_value = mock_worker

        connector_config = KvCacheConnectorConfig(
            connector_module="",
            connector_scheduler_class="KvConnectorScheduler",
            connector_worker_class="KvConnectorWorker",
        )

        call_time_monitor = CallTimeMonitor()

        call_time_monitor.monitor_fn(mock_scheduler.build_connector_meta,
                                     "build_connector_meta")
        call_time_monitor.monitor_fn(mock_scheduler.get_num_new_matched_tokens,
                                     "get_num_new_matched_tokens")
        call_time_monitor.monitor_fn(mock_scheduler.request_finished,
                                     "request_finished")

        call_time_monitor.monitor_fn(mock_worker.start_load_kv, "start_load_kv")
        call_time_monitor.monitor_fn(mock_worker.wait_for_layer_load,
                                     "wait_for_layer_load")
        call_time_monitor.monitor_fn(mock_worker.save_kv_layer, "save_kv_layer")
        call_time_monitor.monitor_fn(mock_worker.wait_for_save, "wait_for_save")
        call_time_monitor.monitor_fn(mock_worker.get_finished, "get_finished")

        yield connector_config, mock_scheduler, mock_worker, call_time_monitor


@pytest.mark.threadleak(enabled=False)
def test_llm_api_connector_simple(connector):
    connector_config, scheduler, worker, call_time_monitor = connector

    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"

    NUM_TOKENS = 8

    model = LLM(model="Qwen/Qwen2-0.5B",
                backend="pytorch",
                disable_overlap_scheduler=True,
                connector_config=connector_config,
                cuda_graph_config=None)

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    model.generate(["Hello, world"], sampling_params)

    assert scheduler.build_connector_meta.call_count == NUM_TOKENS

    for i, call in enumerate(scheduler.build_connector_meta.call_args_list):
        scheduler_output = call[0][0]
        assert len(scheduler_output.requests) == 1
        if i != 0:
            assert len(scheduler_output.requests[0].new_tokens) == 1

    assert worker.start_load_kv.call_count == NUM_TOKENS

    # We should have always built our metadata before loading kv.
    for load_kv_call_time, build_metadata_call_time in zip(
            call_time_monitor["start_load_kv"],
            call_time_monitor["build_connector_meta"]):
        assert build_metadata_call_time < load_kv_call_time

    assert scheduler.get_num_new_matched_tokens.call_count == 1

    num_layers = max(call.args[0]
                     for call in worker.wait_for_layer_load.call_args_list) + 1

    assert worker.wait_for_layer_load.call_count == num_layers * NUM_TOKENS
    assert worker.save_kv_layer.call_count == num_layers * NUM_TOKENS

    for i, call in enumerate(worker.wait_for_layer_load.call_args_list):
        assert call.args[0] == i % num_layers

    for i, call in enumerate(worker.save_kv_layer.call_args_list):
        assert call.args[0] == i % num_layers

    assert worker.wait_for_save.call_count == NUM_TOKENS

    assert scheduler.request_finished.call_count == 1
    assert worker.get_finished.call_count == NUM_TOKENS
