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

import math
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm import LLM, DisaggregatedParams, SamplingParams
from tensorrt_llm.llmapi.llm_args import (CacheTransceiverConfig, KvCacheConfig,
                                          KvCacheConnectorConfig)

from ..conftest import llm_models_root


@pytest.fixture(scope="function")
def model_with_connector():
    with patch("tensorrt_llm._torch.pyexecutor.py_executor_creator.importlib"
               ) as importlib_mock:
        mock_scheduler = MagicMock()
        mock_worker = MagicMock()

        importlib_mock.import_module.return_value.KvConnectorScheduler.return_value = mock_scheduler
        importlib_mock.import_module.return_value.KvConnectorWorker.return_value = mock_worker

        kv_connector_config = KvCacheConnectorConfig(
            connector_module="",
            connector_scheduler_class="KvConnectorScheduler",
            connector_worker_class="KvConnectorWorker",
        )

        def model_fn(*args, **kwargs):
            return LLM(
                *args,
                **kwargs,
                model=f"{llm_models_root()}/Qwen2-0.5B",
                backend="pytorch",
                kv_connector_config=kv_connector_config,
                cuda_graph_config=None,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
            )

        yield model_fn, mock_scheduler, mock_worker


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    yield


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
def test_connector_simple(enforce_single_worker, model_with_connector,
                          use_overlap_scheduler):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    model.generate(["Hello, world"], sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1

    # Allocate 1 block.
    assert len(scheduler.update_state_after_alloc.call_args.args[1]) == 1

    # With the overlap scheduler, we generate one extra token.
    assert scheduler.build_connector_meta.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)

    # We should have a single `SchedulerOutput` per forward pass.
    for i, call in enumerate(scheduler.build_connector_meta.call_args_list):
        scheduler_output = call[0][0]
        if i == 0:
            assert len(scheduler_output.new_requests) == 1
            assert len(scheduler_output.cached_requests) == 0
        elif i == 1 and use_overlap_scheduler:
            assert len(scheduler_output.new_requests) == 0
            assert len(scheduler_output.cached_requests) == 1

            assert len(scheduler_output.cached_requests[0].new_tokens) == 0
        else:
            assert len(scheduler_output.new_requests) == 0
            assert len(scheduler_output.cached_requests) == 1

            assert len(scheduler_output.cached_requests[0].new_tokens) == 1

    # We call `start_load_kv` once at the beginning of each forward pass.
    assert worker.start_load_kv.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)

    # Only called once when the request is received.
    assert scheduler.get_num_new_matched_tokens.call_count == 1

    num_layers = max(call.args[0]
                     for call in worker.wait_for_layer_load.call_args_list) + 1

    # Called num_layers * num_forward_passes times.
    assert worker.wait_for_layer_load.call_count == num_layers * (
        NUM_TOKENS + int(use_overlap_scheduler))
    assert worker.save_kv_layer.call_count == num_layers * (
        NUM_TOKENS + int(use_overlap_scheduler))

    for i, call in enumerate(worker.wait_for_layer_load.call_args_list):
        assert call.args[0] == i % num_layers

    for i, call in enumerate(worker.save_kv_layer.call_args_list):
        assert call.args[0] == i % num_layers

    assert worker.wait_for_save.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)

    assert scheduler.request_finished.call_count == 1

    assert len(scheduler.request_finished.call_args.args[1]) == 1

    assert worker.get_finished.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
def test_connector_async_onboard(enforce_single_worker, model_with_connector,
                                 use_overlap_scheduler):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 16, True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    model.generate([
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    ], SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True))

    # Once for the initial poll, then once for each token. One extra token when using the overlap scheduler.
    assert worker.get_finished.call_count == NUM_TOKENS + 1 + int(
        use_overlap_scheduler)

    # In the first iteration, there should be a single request id provided.
    assert len(worker.get_finished.call_args_list[0].args[1]) == 1


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
def test_connector_async_save(enforce_single_worker, model_with_connector,
                              use_overlap_scheduler):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    scheduler.request_finished.return_value = True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    model.generate(["Hello, world"], sampling_params)

    assert scheduler.request_finished.call_count == 1

    assert len(scheduler.request_finished.call_args.args[1]) == 1

    # On the last call to get_finished, we should be providing the async saving request. One extra token when using the overlap scheduler.
    assert worker.get_finished.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)

    for i, call in enumerate(worker.get_finished.call_args_list):
        args = call.args
        if i != len(worker.get_finished.call_args_list) - 1:
            assert args == ([], [])
        else:
            assert len(args[0]) == 1
            assert args[0][0] == scheduler.request_finished.call_args.args[
                0].request_id


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
def test_connector_scheduler_output(enforce_single_worker, model_with_connector,
                                    use_overlap_scheduler):
    NUM_INPUT_TOKENS = 48
    NUM_TOKENS = 32
    BLOCK_SIZE = 32

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

    model.generate([0] * NUM_INPUT_TOKENS, sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1
    assert len(
        scheduler.update_state_after_alloc.call_args.args[1]) == math.ceil(
            NUM_INPUT_TOKENS / BLOCK_SIZE)

    # Additional token when using the overlap scheduler.
    assert scheduler.build_connector_meta.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)

    for i, call in enumerate(scheduler.build_connector_meta.call_args_list):
        sched_output = call.args[0]

        if i == 0:
            assert len(sched_output.new_requests) == 1
            assert len(sched_output.cached_requests) == 0
            request = sched_output.new_requests[0]

            assert len(request.new_tokens) == NUM_INPUT_TOKENS
            assert len(request.new_block_ids) == math.ceil(NUM_INPUT_TOKENS /
                                                           BLOCK_SIZE)
            assert request.computed_position == 0
            assert request.num_scheduled_tokens == NUM_INPUT_TOKENS
        elif i == 1 and use_overlap_scheduler:
            assert len(sched_output.new_requests) == 0
            assert len(sched_output.cached_requests) == 1

            assert len(sched_output.cached_requests[0].new_tokens) == 0
            assert sched_output.cached_requests[0].num_scheduled_tokens == 1
        else:
            assert len(sched_output.cached_requests) == 1
            assert len(sched_output.new_requests) == 0
            request = sched_output.cached_requests[0]

            assert len(request.new_tokens) == 1

            if (request.computed_position +
                    int(use_overlap_scheduler)) % BLOCK_SIZE == 0:
                assert len(request.new_block_ids) == 1
            else:
                assert request.new_block_ids == []

            assert request.num_scheduled_tokens == 1

    scheduler.build_connector_meta.reset_mock()

    scheduler.get_num_new_matched_tokens.return_value = 8, False

    assert len(scheduler.request_finished.call_args.args[1]) == math.ceil(
        (NUM_INPUT_TOKENS + NUM_TOKENS) / BLOCK_SIZE)

    model.generate([1] * NUM_INPUT_TOKENS, sampling_params)

    # The initial computed position should be 0, since we haven't yet onboarded any blocks.
    assert scheduler.build_connector_meta.call_args_list[0].args[
        0].new_requests[0].computed_position == 0


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
def test_connector_scheduler_output_chunked_context(enforce_single_worker,
                                                    model_with_connector,
                                                    use_overlap_scheduler):
    model_fn, scheduler, worker = model_with_connector

    CHUNK_SIZE = 128
    BLOCK_SIZE = 32

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler,
                     enable_chunked_prefill=True,
                     max_num_tokens=CHUNK_SIZE)

    assert worker.register_kv_caches.call_count == 1

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=BLOCK_SIZE, ignore_eos=True)

    model.generate([0] * (CHUNK_SIZE * 2), sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1

    assert len(
        scheduler.update_state_after_alloc.call_args.args[1]) == math.ceil(
            CHUNK_SIZE * 2 / BLOCK_SIZE)

    for i, call in enumerate(scheduler.build_connector_meta.call_args_list):
        sched_output = call.args[0]

        if i == 0:
            assert len(sched_output.new_requests) == 1
            assert len(sched_output.cached_requests) == 0
            req = sched_output.new_requests[0]
        else:
            assert len(sched_output.cached_requests) == 1
            assert len(sched_output.new_requests) == 0
            req = sched_output.cached_requests[0]

        if i == 0:
            # The first prefill chunk.
            # All of the prefill tokens and all the blocks should be provided upfront.
            assert req.computed_position == 0
            assert len(req.new_tokens) == CHUNK_SIZE * 2
            assert len(req.new_block_ids) == math.ceil(CHUNK_SIZE * 2 /
                                                       BLOCK_SIZE)
            assert req.num_scheduled_tokens == CHUNK_SIZE
        elif i == 1:
            # The second prefill chunk.
            assert req.computed_position == CHUNK_SIZE
            assert len(req.new_tokens) == 0
            assert len(req.new_block_ids) == 0
            assert req.num_scheduled_tokens == CHUNK_SIZE
        elif i == 2 and use_overlap_scheduler:
            assert len(req.new_tokens) == 0
            assert req.num_scheduled_tokens == 1
        else:
            assert len(req.new_tokens) == 1
            assert req.num_scheduled_tokens == 1
    assert len(scheduler.request_finished.call_args.args[1]) == math.ceil(
        (CHUNK_SIZE * 2 + BLOCK_SIZE) / BLOCK_SIZE)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("save_async", [False, True])
def test_connector_disagg_prefill(enforce_single_worker, model_with_connector,
                                  save_async):
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(
        disable_overlap_scheduler=True,
        cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"))

    sampling_params = SamplingParams(ignore_eos=True)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    if save_async:
        scheduler.request_finished.return_value = True

        worker.get_finished.side_effect = lambda finished_gen, load_async: (
            finished_gen, load_async)
    else:
        scheduler.request_finished.return_value = False
        worker.get_finished.return_value = [], []

    model.generate([0] * 48,
                   sampling_params=sampling_params,
                   disaggregated_params=disaggregated_params)

    assert scheduler.build_connector_meta.call_count == 1

    scheduler_output = scheduler.build_connector_meta.call_args.args[0]

    assert len(scheduler_output.new_requests) == 1
    assert len(scheduler_output.cached_requests) == 0

    req = scheduler_output.new_requests[0]

    assert req.computed_position == 0
    assert req.num_scheduled_tokens == 48
    assert len(req.new_tokens) == 48

    assert scheduler.request_finished.call_count == 1
