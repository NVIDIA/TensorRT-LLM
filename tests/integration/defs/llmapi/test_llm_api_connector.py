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
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
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

        yield connector_config, mock_scheduler, mock_worker


# Needed because MagicMocks don't work across processes.
# TODO(jthomson04): This limits us to testing only TP1 for now.
os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"


@pytest.mark.threadleak(enabled=False)
def test_llm_api_connector_simple(connector):
    connector_config, scheduler, worker = connector

    NUM_TOKENS = 8

    model = LLM(model="Qwen/Qwen2-0.5B",
                backend="pytorch",
                disable_overlap_scheduler=True,
                connector_config=connector_config,
                cuda_graph_config=None,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1))

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    model.generate(["Hello, world"], sampling_params)

    assert scheduler.build_connector_meta.call_count == NUM_TOKENS

    # We should have a single `SchedulerOutput` per forward pass.
    for i, call in enumerate(scheduler.build_connector_meta.call_args_list):
        scheduler_output = call[0][0]
        assert len(scheduler_output.requests) == 1

        # If this is not prefill, we should always be adding a single token.
        if i != 0:
            assert len(scheduler_output.requests[0].new_tokens) == 1

    # We call `start_load_kv` once at the beginning of each forward pass.
    assert worker.start_load_kv.call_count == NUM_TOKENS

    # Only called once when the request is received.
    assert scheduler.get_num_new_matched_tokens.call_count == 1

    num_layers = max(call.args[0]
                     for call in worker.wait_for_layer_load.call_args_list) + 1

    # Called num_layers * num_forward_passes times.
    assert worker.wait_for_layer_load.call_count == num_layers * NUM_TOKENS
    assert worker.save_kv_layer.call_count == num_layers * NUM_TOKENS

    for i, call in enumerate(worker.wait_for_layer_load.call_args_list):
        assert call.args[0] == i % num_layers

    for i, call in enumerate(worker.save_kv_layer.call_args_list):
        assert call.args[0] == i % num_layers

    assert worker.wait_for_save.call_count == NUM_TOKENS

    assert scheduler.request_finished.call_count == 1
    assert worker.get_finished.call_count == NUM_TOKENS


@pytest.mark.threadleak(enabled=False)
def test_llm_api_connector_async_onboard(connector):
    connector_config, scheduler, worker = connector

    NUM_TOKENS = 8

    model = LLM(model="Qwen/Qwen2-0.5B",
                backend="pytorch",
                disable_overlap_scheduler=True,
                connector_config=connector_config,
                cuda_graph_config=None,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1))

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 16, True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    model.generate([
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    ], SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True))

    # Once for the initial poll, then once for each token.
    assert worker.get_finished.call_count == NUM_TOKENS + 1

    # In the first iteration, there should be a single request id provided.
    assert len(worker.get_finished.call_args_list[0].args[1]) == 1


@pytest.mark.threadleak(enabled=False)
def test_llm_api_connector_async_save(connector):
    connector_config, scheduler, worker = connector

    NUM_TOKENS = 8

    model = LLM(model="Qwen/Qwen2-0.5B",
                backend="pytorch",
                disable_overlap_scheduler=True,
                connector_config=connector_config,
                cuda_graph_config=None,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1))

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    scheduler.request_finished.return_value = True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    model.generate(["Hello, world"], sampling_params)

    assert scheduler.request_finished.call_count == 1

    # On the last call to get_finished, we should be providing the async saving request.
    assert worker.get_finished.call_count == NUM_TOKENS

    for i in range(NUM_TOKENS):
        args = worker.get_finished.call_args_list[i].args
        if i != NUM_TOKENS - 1:
            assert args == ([], [])
        else:
            assert len(args[0]) == 1
            assert args[0][0] == scheduler.request_finished.call_args.args[
                0].request_id
