# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import math
import os
import shutil
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm import LLM, DisaggregatedParams, SamplingParams
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import (CacheTransceiverConfig, KvCacheConfig,
                                          KvCacheConnectorConfig)
from tensorrt_llm.llmapi.llm_utils import KvCacheRetentionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from ..conftest import llm_models_root

# Name of the TensorRT-LLM logger. It sets `propagate = False`
# (tensorrt_llm/logger.py:186-187), so pytest's `caplog` only sees its records
# once `caplog.handler` is attached to it directly.
TRTLLM_LOGGER_NAME = "TRT-LLM"

# Emitted by `_fallback_if_unsupported_kv_cache_manager_v2`
# (tensorrt_llm/_torch/pyexecutor/_util.py:629-631) when a connector run is
# downgraded from V2 to V1.
FALLBACK_WARNING_FRAGMENT = "Falling back to KVCacheManager"

# V1 `KVCacheManager` methods reached on the KV connector path proper. Each is
# defined in tensorrt_llm/_torch/pyexecutor/resource_manager.py and wraps a
# nanobind method on the C++ manager. `KVCacheManagerV2` implements none of
# them and is not meant to: each assumes a single flat block-id space over one
# primary pool, which cannot describe memory whose page indices are scoped to a
# layer group. V2's connector contract is `register_kv_cache_layout` plus
# `get_page_indices_by_layer_group` / `get_connector_page_indices` instead.
CONNECTOR_V1_ONLY_KV_CACHE_MANAGER_METHODS = (
    # Connector bring-up: hands the worker the single primary pool tensor.
    # `PyExecutor._maybe_init_kv_connector_manager`.
    "get_unique_primary_pool",
    # `KvCacheConnectorSchedulerOutputRequest.update_and_build_data` and
    # `PyExecutor.kv_connector_request_finished`.
    "get_cache_indices",
    # `update_and_build_data`, for `RequestData.block_hashes`.
    "commit_and_get_block_hashes",
    # `update_and_build_data`, for `RequestData.priorities`.
    "get_priority_by_block_id",
)

# Reached only when the connector coexists with disaggregated serving
# (`test_connector_disagg_prefill`), via AsyncTransferManager and the V1 cache
# reuse adapter - not from `KvCacheConnectorManager` itself. None of them is a
# gap for V2, because V2 does not traverse those paths:
# `enable_partial_reuse_for_disagg` excludes V2, so AsyncTransferManager never
# reaches the pin/unpin pair, and the Python transceiver (the only one V2 can
# be driven by) resolves block ids through `_CacheReuseAdapterV2`.
DISAGG_PATH_KV_CACHE_MANAGER_METHODS = (
    "store_blocks_for_reuse",
    "unpin_blocks_by_id",
    "get_memory_pool_block_indices",
    "pin_blocks",  # no Python caller today
)


@pytest.fixture(scope="function")
def use_kv_cache_manager_v2(request):
    """Run each connector test under both KV cache managers.

    Parametrized by an explicit `@pytest.mark.parametrize(..., indirect=True)`
    on each test, applied as the innermost decorator so the manager lands first
    in the generated test id. It is spelled out per test rather than set as a
    fixture `params=` because the test-list validator
    (scripts/check_test_list.py) resolves ids from parametrize decorators via
    AST and cannot see fixture-level parametrization.

    Selecting V2 must actually reach V2: `_fallback_if_unsupported_kv_cache_manager_v2`
    silently substitutes the V1 manager for combinations it cannot serve, and a
    connector test that ran on V1 while claiming to test V2 would pass while
    exercising nothing. `test_connector_runs_on_kv_cache_manager_v2` guards that.
    """
    return request.param


@pytest.fixture(scope="function")
def model_with_connector(use_kv_cache_manager_v2):
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

            default_kwargs = {
                "model": f"{llm_models_root()}/Qwen3/Qwen3-0.6B",
                "backend": "pytorch",
                "kv_connector_config": kv_connector_config,
                "cuda_graph_config": None,
                "kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.1),
                # Connector tests use at most 256 input tokens. Keep model
                # construction independent of the GPU's available KV capacity.
                "max_seq_len": 1024,
            }

            merged_kwargs = {**default_kwargs, **kwargs}

            # Tests that supply their own `KvCacheConfig` must still honour the
            # manager under test, otherwise the V2 parametrization silently
            # degrades into a second V1 run.
            kv_cache_config = merged_kwargs.get("kv_cache_config")
            if kv_cache_config is not None:
                kv_cache_config.use_kv_cache_manager_v2 = use_kv_cache_manager_v2

            return LLM(*args, **merged_kwargs)

        yield model_fn, mock_scheduler, mock_worker


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    yield


# Some KV connector API calls are made after a full response is returned
# (`request_finished`, the trailing `get_finished` polls and asynchronous
# saves), and there is no public signal for when they are complete. Instead of
# sleeping a fixed amount, wait until the connector mocks stop recording new
# calls. That returns as soon as the connector goes quiet, and - unlike a fixed
# sleep - stretches automatically when a slower path lengthens the tail.
CONNECTOR_QUIESCE_TIMEOUT_S = 60.0
CONNECTOR_QUIET_PERIOD_S = 0.5
CONNECTOR_POLL_INTERVAL_S = 0.01

# Fraction of generated tokens a connector-warmed run must reproduce exactly.
# See test_connector_e2e_persistent_cache for why this is not 1.0.
E2E_MIN_TOKEN_AGREEMENT = 0.75


def assert_kv_caches_registered(worker, use_kv_cache_manager_v2):
    """The two managers hand the worker its pools through different entry points.

    V1 passes a single pool tensor to `register_kv_caches`; V2 has no such
    tensor and passes a `KvCacheLayout` to `register_kv_cache_layout` instead.
    Asserting the V1 method unconditionally would silently pass on V2 only if
    the connector were never registered at all.
    """
    if use_kv_cache_manager_v2:
        assert worker.register_kv_cache_layout.call_count == 1
        assert worker.register_kv_caches.call_count == 0
    else:
        assert worker.register_kv_caches.call_count == 1
        assert worker.register_kv_cache_layout.call_count == 0


def wait_for_connector_quiescence(scheduler,
                                  worker,
                                  timeout=CONNECTOR_QUIESCE_TIMEOUT_S,
                                  quiet_period=CONNECTOR_QUIET_PERIOD_S):
    """Block until no new connector callback lands for `quiet_period` seconds.

    `MagicMock.mock_calls` records every call made on the mock and its children,
    so its length is a monotonic progress counter for connector activity.
    """
    deadline = time.monotonic() + timeout

    def total_calls():
        return len(scheduler.mock_calls) + len(worker.mock_calls)

    last_seen = total_calls()
    quiet_since = time.monotonic()

    while time.monotonic() < deadline:
        time.sleep(CONNECTOR_POLL_INTERVAL_S)

        current = total_calls()
        if current != last_seen:
            last_seen = current
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= quiet_period:
            return

    raise AssertionError(
        f"KV connector callbacks did not go quiet within {timeout}s "
        f"({last_seen} calls recorded). The connector is still active or a "
        "callback is blocked.")


def generate_and_wait(model, scheduler, worker, *args, **kwargs):
    """`model.generate`, then block until the connector callbacks settle."""
    outputs = model.generate(*args, **kwargs)
    wait_for_connector_quiescence(scheduler, worker)
    return outputs


def test_v2_connector_contract_does_not_reuse_the_v1_methods():
    """The V2 connector path implements none of the V1 accessors, by design.

    Something depends on that, and it does not ask: `update_and_build_data`
    reports `block_hashes` and `priorities` empty on V2 by branching on
    `isinstance(manager, KVCacheManagerV2)`, not on `hasattr`. Those
    short-circuits are only correct while V2 genuinely has no such accessor -
    the day one is added (retention priorities are a known gap; see
    `test_connector_priorities`) the branch keeps reporting nothing while the
    data exists, and this is what says so.

    A static check rather than an end-to-end run: under V2 the connector would
    die at the *first* method it reached, so no run can report more than one at
    a time.

    Needs no GPU.
    """
    stale = [
        name for name in CONNECTOR_V1_ONLY_KV_CACHE_MANAGER_METHODS +
        DISAGG_PATH_KV_CACHE_MANAGER_METHODS
        if not hasattr(KVCacheManager, name)
    ]
    assert stale == [], (
        f"{stale} are not defined on the V1 KVCacheManager either, so this "
        "test is measuring a stale method list rather than a real difference.")

    implemented = [
        name for name in CONNECTOR_V1_ONLY_KV_CACHE_MANAGER_METHODS
        if hasattr(KVCacheManagerV2, name)
    ]
    assert implemented == [], (
        f"KVCacheManagerV2 now implements {implemented}. A V1-shaped accessor "
        "on V2 is not automatically the right answer - a flat block-id list "
        "cannot describe more than one layer group - but if it is, revisit the "
        "`is_v2` short-circuits in "
        "`KvCacheConnectorSchedulerOutputRequest.update_and_build_data`, which "
        "report nothing on the strength of these methods being absent.")

    # The other half of the contract: what V2 offers instead.
    for name in ("get_page_indices_by_layer_group",
                 "get_connector_page_indices"):
        assert hasattr(KVCacheManagerV2, name), (
            f"KVCacheManagerV2.{name} is the V2 replacement for the V1 "
            "block-id accessors and every connector path on V2 goes through "
            "it.")
    assert hasattr(KvCacheConnectorWorker, "register_kv_cache_layout"), (
        "The worker ABC must keep a default `register_kv_cache_layout`, or "
        "every existing connector becomes abstract and fails to instantiate.")


@pytest.mark.threadleak(enabled=False)
def test_connector_runs_on_kv_cache_manager_v2(enforce_single_worker,
                                               monkeypatch, caplog):
    """Anti-vacuity guard for the `kv_cache_manager_v2` parametrization.

    Without this, every V2-parametrized test below could pass green while
    `_fallback_if_unsupported_kv_cache_manager_v2` silently swapped in the V1
    manager. It asserts positively that V2 is constructed, and that the
    downgrade warning is absent.

    Any construction failure is recorded rather than asserted on: what must
    hold is that the run reached V2 rather than being papered over by a
    fallback, which stays true regardless of how far bring-up gets.
    """
    constructed = []

    def record_construction(cls):
        original_init = cls.__init__

        def recording_init(self, *args, **kwargs):
            constructed.append(cls.__name__)
            return original_init(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", recording_init)

    record_construction(KVCacheManagerV2)
    record_construction(KVCacheManager)

    # The TensorRT-LLM logger sets `propagate = False`, so caplog only sees its
    # records once its handler is attached to that logger directly.
    trtllm_logger = logging.getLogger(TRTLLM_LOGGER_NAME)
    trtllm_logger.addHandler(caplog.handler)

    construction_error = None
    llm = None
    try:
        with patch(
                "tensorrt_llm._torch.pyexecutor.py_executor_creator.importlib"
        ) as importlib_mock:
            connector_module = importlib_mock.import_module.return_value
            connector_module.KvConnectorScheduler.return_value = MagicMock()
            connector_module.KvConnectorWorker.return_value = MagicMock()

            try:
                llm = LLM(
                    model=f"{llm_models_root()}/Qwen2-0.5B",
                    backend="pytorch",
                    kv_connector_config=KvCacheConnectorConfig(
                        connector_module="",
                        connector_scheduler_class="KvConnectorScheduler",
                        connector_worker_class="KvConnectorWorker",
                    ),
                    cuda_graph_config=None,
                    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1,
                                                  use_kv_cache_manager_v2=True),
                )
            # The V2 connector path is knowingly incomplete, so any construction
            # failure is an acceptable outcome. What must hold is that the run
            # reached V2 instead of being papered over by a V1 fallback.
            except Exception as exc:  # noqa: BLE001
                construction_error = exc
    finally:
        trtllm_logger.removeHandler(caplog.handler)
        if llm is not None:
            llm.shutdown()

    # Report the current frontier so the failure mode is visible in CI output
    # instead of being silently swallowed by the except above.
    print(f"\n[connector+V2 frontier] construction_error="
          f"{type(construction_error).__name__ if construction_error else None}"
          f": {construction_error}")

    assert "KVCacheManagerV2" in constructed, (
        "KVCacheManagerV2 was never constructed, so the connector run silently "
        f"fell back to V1 (managers constructed: {constructed}; construction "
        f"error: {construction_error!r}). Every kv_cache_manager_v2-"
        "parametrized connector test in this file is vacuous until this passes."
    )

    assert FALLBACK_WARNING_FRAGMENT not in caplog.text, (
        f"{FALLBACK_WARNING_FRAGMENT!r} was logged, so the connector was "
        "downgraded to the V1 KV cache manager.")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_simple(enforce_single_worker, model_with_connector,
                          use_overlap_scheduler, use_kv_cache_manager_v2):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    generate_and_wait(model, scheduler, worker, ["Hello, world"],
                      sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1

    # Allocate 1 block.
    assert len(scheduler.update_state_after_alloc.call_args.args[1]) == 1

    # With the overlap scheduler, we generate one extra token.
    assert scheduler.build_connector_meta.call_count == NUM_TOKENS

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
    assert worker.start_load_kv.call_count == NUM_TOKENS

    # Only called once when the request is received.
    assert scheduler.get_num_new_matched_tokens.call_count == 1

    num_layers = max(call.args[0]
                     for call in worker.wait_for_layer_load.call_args_list) + 1

    # Called num_layers * num_forward_passes times.
    assert worker.wait_for_layer_load.call_count == num_layers * (NUM_TOKENS)
    assert worker.save_kv_layer.call_count == num_layers * (NUM_TOKENS)

    for i, call in enumerate(worker.wait_for_layer_load.call_args_list):
        assert call.args[0] == i % num_layers

    for i, call in enumerate(worker.save_kv_layer.call_args_list):
        assert call.args[0] == i % num_layers

    assert worker.wait_for_save.call_count == NUM_TOKENS

    assert scheduler.request_finished.call_count == 1

    assert len(scheduler.request_finished.call_args.args[1]) == 1

    assert worker.get_finished.call_count == NUM_TOKENS + int(
        use_overlap_scheduler)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_async_onboard(enforce_single_worker, model_with_connector,
                                 use_overlap_scheduler,
                                 use_kv_cache_manager_v2):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 16, True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    generate_and_wait(model, scheduler, worker, [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    ], SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True))

    # Once for the initial poll, then once for each token. One extra token when using the overlap scheduler.
    assert worker.get_finished.call_count == NUM_TOKENS + 1 + int(
        use_overlap_scheduler)

    # In the first iteration, there should be a single request id provided.
    assert len(worker.get_finished.call_args_list[0].args[1]) == 1


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_async_save(enforce_single_worker, model_with_connector,
                              use_overlap_scheduler, use_kv_cache_manager_v2):
    NUM_TOKENS = 8

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    scheduler.request_finished.return_value = True

    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    generate_and_wait(model, scheduler, worker, ["Hello, world"],
                      sampling_params)

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
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_scheduler_output(enforce_single_worker, model_with_connector,
                                    use_overlap_scheduler,
                                    use_kv_cache_manager_v2):
    NUM_INPUT_TOKENS = 48
    NUM_TOKENS = 32
    BLOCK_SIZE = 32

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler, )

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

    generate_and_wait(model, scheduler, worker, [0] * NUM_INPUT_TOKENS,
                      sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1
    assert len(
        scheduler.update_state_after_alloc.call_args.args[1]) == math.ceil(
            NUM_INPUT_TOKENS / BLOCK_SIZE)

    assert scheduler.build_connector_meta.call_count == NUM_TOKENS

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

    generate_and_wait(model, scheduler, worker, [1] * NUM_INPUT_TOKENS,
                      sampling_params)

    # The initial computed position should be 0, since we haven't yet onboarded any blocks.
    assert scheduler.build_connector_meta.call_args_list[0].args[
        0].new_requests[0].computed_position == 0


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_overlap_scheduler", [True, False])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_scheduler_output_chunked_context(enforce_single_worker,
                                                    model_with_connector,
                                                    use_overlap_scheduler,
                                                    use_kv_cache_manager_v2):
    model_fn, scheduler, worker = model_with_connector

    CHUNK_SIZE = 128
    BLOCK_SIZE = 32

    model = model_fn(disable_overlap_scheduler=not use_overlap_scheduler,
                     enable_chunked_prefill=True,
                     max_num_tokens=CHUNK_SIZE)

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=BLOCK_SIZE, ignore_eos=True)

    generate_and_wait(model, scheduler, worker, [0] * (CHUNK_SIZE * 2),
                      sampling_params)

    assert scheduler.update_state_after_alloc.call_count == 1

    # V1 allocates for the whole prompt when the sequence is added, so every
    # block exists on the first chunk. V2 allocates per chunk, which is the
    # lower-peak-memory behaviour and the one to keep; the remaining blocks
    # arrive as append-deltas in `new_block_ids` on the next chunk, so no
    # information is lost. The expectation is split rather than V2 changed.
    total_blocks = math.ceil(CHUNK_SIZE * 2 / BLOCK_SIZE)
    first_chunk_blocks = (math.ceil(CHUNK_SIZE / BLOCK_SIZE)
                          if use_kv_cache_manager_v2 else total_blocks)

    assert len(scheduler.update_state_after_alloc.call_args.args[1]
               ) == first_chunk_blocks

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
            # The first prefill chunk. All of the prefill tokens are provided
            # upfront on both managers; the blocks are whatever has been
            # allocated so far.
            assert req.computed_position == 0
            assert len(req.new_tokens) == CHUNK_SIZE * 2
            assert len(req.new_block_ids) == first_chunk_blocks
            assert req.num_scheduled_tokens == CHUNK_SIZE
        elif i == 1:
            # The second prefill chunk.
            assert req.computed_position == CHUNK_SIZE
            assert len(req.new_tokens) == 0
            assert len(req.new_block_ids) == total_blocks - first_chunk_blocks
            assert req.num_scheduled_tokens == CHUNK_SIZE
        elif i == 2 and use_overlap_scheduler:
            assert len(req.new_tokens) == 0
            assert req.num_scheduled_tokens == 1
        else:
            assert len(req.new_tokens) == 1
            assert req.num_scheduled_tokens == 1
    assert len(scheduler.request_finished.call_args.args[1]) == math.ceil(
        (CHUNK_SIZE * 2 + BLOCK_SIZE) / BLOCK_SIZE)


# The mock scheduler answers every query with the same offer; these helpers
# record what it was asked and when, so a test can assert the connector was
# consulted exactly once per request rather than assuming it.
def record_connector_queries(scheduler, num_matched, load_async=False):
    """Answer every query with `num_matched`, recording when each one arrived.

    The third element of each record is `build_connector_meta.call_count` at
    query time -- the number of iterations whose connector hooks had already
    run. Phase 1 runs during scheduling, before those hooks, so a request asked
    in the first scheduling pass records 0. That is how the tests below *prove*
    a request was asked before the iteration it eventually ran in, rather than
    assuming the scheduler deferred anything.
    """
    queries = []

    def side_effect(request, num_computed_tokens):
        queries.append((request.request_id, num_computed_tokens,
                        scheduler.build_connector_meta.call_count))
        return num_matched, load_async

    scheduler.get_num_new_matched_tokens.side_effect = side_effect
    return queries


# Sliding-window coverage. `KvCacheConfig.max_attention_window` is repeated
# cyclically across layers (llm_args.py:3761-3765), so a one-element list gives
# every layer the same window -- one V2 layer group with a live window -- while
# a two-element list alternates and produces two. A window equal to
# `max_seq_len` is normalised to "no window" (kv_cache_manager_v2.py:856-858),
# which is how the full-attention half of the VSWA pair is spelled.
SWA_WINDOW = 64
SWA_MAX_SEQ_LEN = 512
SWA_NUM_INPUT_TOKENS = 256


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [True],
                         ids=["kv_cache_manager_v2"],
                         indirect=True)
def test_connector_uniform_sliding_window(enforce_single_worker,
                                          model_with_connector,
                                          use_kv_cache_manager_v2):
    """Connector against a KV cache in which every layer slides.

    There was no sliding-window connector coverage before this test. Uniform
    SWA is a single layer group, which is the configuration where the
    connector's flat `new_block_ids` list is still well defined -- so this pins
    the block-reporting contract, and the VSWA test below pins what happens once
    that assumption breaks.

    V2 only. The V1 guard rejects *variable* windows
    (py_executor_creator.py:845-850), so uniform SWA reaches V1's connector path
    and then dies inside it: `commit_and_get_block_hashes` (kv_cache_connector.py:386)
    raises "commitAndGetBlockHashesForRequest does not support sliding-window
    attention with detached front blocks" (kvCacheManager.cpp:4645) as soon as
    the window drops a front block. That is a pre-existing V1 limitation, not
    something this work introduces, so it is recorded rather than asserted here.
    """
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True,
                     max_seq_len=SWA_MAX_SEQ_LEN,
                     kv_cache_config=KvCacheConfig(
                         free_gpu_memory_fraction=0.1,
                         max_attention_window=[SWA_WINDOW]))

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    worker.get_finished.return_value = [], []

    generate_and_wait(model, scheduler, worker, [0] * SWA_NUM_INPUT_TOKENS,
                      SamplingParams(max_tokens=4, ignore_eos=True))

    sched_output = scheduler.build_connector_meta.call_args_list[0].args[0]
    assert len(sched_output.new_requests) == 1
    req = sched_output.new_requests[0]

    assert req.computed_position == 0
    assert req.num_scheduled_tokens == SWA_NUM_INPUT_TOKENS
    # A single layer group keeps the flat list meaningful on both managers.
    assert req.new_block_ids

    if use_kv_cache_manager_v2:
        # Anti-vacuity: prove the window really did collapse to one layer
        # group, otherwise the assertion above would hold for the wrong reason.
        layout = worker.register_kv_cache_layout.call_args.args[0]
        assert len(layout.groups) == 1
        assert layout.groups[0].window_size == SWA_WINDOW
        assert list(req.new_block_ids_by_layer_group) == [0]
        assert req.new_block_ids_by_layer_group[0] == req.new_block_ids
    else:
        assert req.new_block_ids_by_layer_group == {}


# Half the prompt, and well past the window, so the pages the offer does *not*
# need are a large enough fraction to assert on.
SWA_OFFER_TOKENS = 128


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [True],
                         ids=["kv_cache_manager_v2"],
                         indirect=True)
def test_connector_sliding_window_prefix_is_backed_by_history(
        enforce_single_worker, model_with_connector, use_kv_cache_manager_v2):
    """Phase 2 raises `history_length`, not just capacity -- observably.

    Under full attention the two are indistinguishable from outside: capacity is
    `prompt_len` whether or not a prefix was served, so the page count says
    nothing. Under a sliding window it does. `history_length` is the sole input
    to the stale-range computation, so raising it to the offer end tells V2 that
    the blocks the window has already left need no pages -- and raising capacity
    alone would allocate one for every block of the served prefix.

    So this is the test that would fail if `_reserve_connector_prefix` called
    `resize(capacity, None)`; `test_connector_uniform_sliding_window` above
    cannot, because it offers nothing.
    """
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True,
                     max_seq_len=SWA_MAX_SEQ_LEN,
                     kv_cache_config=KvCacheConfig(
                         free_gpu_memory_fraction=0.1,
                         max_attention_window=[SWA_WINDOW]))

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    record_connector_queries(scheduler, SWA_OFFER_TOKENS)
    worker.get_finished.return_value = [], []

    generate_and_wait(model, scheduler, worker, [0] * SWA_NUM_INPUT_TOKENS,
                      SamplingParams(max_tokens=4, ignore_eos=True))

    # Anti-vacuity, as in the test above: one layer group with a live window,
    # otherwise the page count below is being read off full attention.
    layout = worker.register_kv_cache_layout.call_args.args[0]
    assert len(layout.groups) == 1
    assert layout.groups[0].window_size == SWA_WINDOW

    req = scheduler.build_connector_meta.call_args_list[0].args[0].new_requests[
        0]

    # The offer was materialized: the position advanced over it, and the
    # runtime rolled the reported position back to the local match.
    assert req.computed_position == 0
    assert req.num_scheduled_tokens == SWA_NUM_INPUT_TOKENS - SWA_OFFER_TOKENS

    all_blocks = math.ceil(SWA_NUM_INPUT_TOKENS / 32)
    page_indices = scheduler.update_state_after_alloc.call_args.args[1]
    assert page_indices == req.new_block_ids
    # One entry per block ordinal either way -- a block with no page reads back
    # as BAD_PAGE_INDEX in place rather than shortening the list, so that block
    # ordinals stay aligned to token ranges (kv_cache_manager_v2.py:2628-2636).
    # The page count is therefore not the signal; *which* entries are bad is.
    assert len(page_indices) == all_blocks

    # With history at the offer end, every block that lies entirely below the
    # window got no page. Without the bump history would still be at the local
    # match, nothing would be stale, and all 8 blocks would be backed.
    stale_blocks = (SWA_OFFER_TOKENS - SWA_WINDOW) // 32
    assert stale_blocks > 0, "test sizes no longer put any block out of window"
    assert page_indices[:stale_blocks] == [BAD_PAGE_INDEX] * stale_blocks, (
        f"expected the first {stale_blocks} blocks of a {SWA_OFFER_TOKENS}-token "
        f"served prefix to fall outside a {SWA_WINDOW}-token window and hold no "
        f"page, got {page_indices}. History was not raised to the offer end.")

    live = page_indices[stale_blocks:]
    assert all(index != BAD_PAGE_INDEX for index in live), (
        f"a block inside the window has no page: {page_indices}")
    assert len(set(live)) == len(live), (
        f"page slots reported to the connector are not distinct: {live}")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_vswa_reports_page_indices_per_layer_group(
        enforce_single_worker, model_with_connector, use_kv_cache_manager_v2):
    """VSWA is where the connector's flat block list stops working.

    V1 cannot describe VSWA to a connector at all: it registers a single primary
    pool, but VSWA allocates one pool per window size. V2's layout describes one
    region set per layer group, so the combination runs there -- but a page index
    is scoped to a layer group, and there is no correct way to flatten indices
    from several groups into one list. `KVCacheManagerV2` therefore reports an
    empty `new_block_ids` and leaves `new_block_ids_by_layer_group` as the only
    correct source (kv_cache_manager_v2.py:2519-2526, kv_cache_connector.py:365-377).

    Both halves are asserted here because the rejection is deliberately
    conditional on the manager (py_executor_creator.py). Pinning only the V2 half
    would let the V1 guard silently disappear.
    """
    model_fn, scheduler, worker = model_with_connector

    def build():
        return model_fn(disable_overlap_scheduler=True,
                        max_seq_len=SWA_MAX_SEQ_LEN,
                        kv_cache_config=KvCacheConfig(
                            free_gpu_memory_fraction=0.1,
                            max_attention_window=[SWA_WINDOW, SWA_MAX_SEQ_LEN]))

    if not use_kv_cache_manager_v2:
        with pytest.raises(NotImplementedError, match="VSWA"):
            build()
        return

    model = build()

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    # Alternating windows must actually produce two groups, one sliding and one
    # full-attention, or the rest of this test is vacuous.
    layout = worker.register_kv_cache_layout.call_args.args[0]
    assert len(layout.groups) == 2
    assert {group.window_size for group in layout.groups} == {SWA_WINDOW, None}

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    worker.get_finished.return_value = [], []

    generate_and_wait(model, scheduler, worker, [0] * SWA_NUM_INPUT_TOKENS,
                      SamplingParams(max_tokens=4, ignore_eos=True))

    # The multi-group degradation reaches `update_state_after_alloc` too. A
    # connector reading only `new_block_ids` sees nothing and saves nothing,
    # which is silent unless pinned here.
    assert scheduler.update_state_after_alloc.call_args.args[1] == []

    sched_output = scheduler.build_connector_meta.call_args_list[0].args[0]
    assert len(sched_output.new_requests) == 1
    req = sched_output.new_requests[0]

    assert req.new_block_ids == []
    assert sorted(req.new_block_ids_by_layer_group) == [0, 1]
    # Ordinals stay positionally aligned across groups: a block with no page in
    # the sliding group reads back as BAD_PAGE_INDEX in place rather than
    # shortening the list (kv_cache_manager_v2.py:2481-2490).
    lengths = {
        len(indices)
        for indices in req.new_block_ids_by_layer_group.values()
    }
    assert len(lengths) == 1


def _disagg_transceiver_config(use_kv_cache_manager_v2):
    """The transceiver each KV cache manager can actually be driven by.

    `CacheTransceiverCpp` is bound to the V1 `BaseKVCacheManager`, while
    `KVCacheManagerV2.impl` is the Python V2 core's manager, so V2 can only use
    the Python transceiver -- which in turn only supports NIXL
    (kv_cache_transceiver.py, `create_kv_cache_transceiver`). This is spelled
    out per manager rather than left at the default because
    `transceiver_runtime` defaults to "auto", and "auto" is resolved from the
    *model's* preference (llm_utils._resolve_transceiver_runtime_auto), which
    knows nothing about which cache manager will be built. Qwen2 declares no
    preference, so the default resolves to the C++ transceiver -- which V2
    cannot use, and which is now rejected with an actionable error rather than
    a nanobind signature mismatch.
    """
    if use_kv_cache_manager_v2:
        return CacheTransceiverConfig(backend="NIXL",
                                      transceiver_runtime="PYTHON")
    return CacheTransceiverConfig(backend="DEFAULT")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("save_async", [False, True])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_disagg_prefill(enforce_single_worker, model_with_connector,
                                  save_async, use_kv_cache_manager_v2):
    model_fn, scheduler, worker = model_with_connector

    transceiver_config = _disagg_transceiver_config(use_kv_cache_manager_v2)

    prefill_worker = model_fn(disable_overlap_scheduler=True,
                              cache_transceiver_config=transceiver_config)

    decode_worker = model_fn(cache_transceiver_config=transceiver_config,
                             kv_connector_config=None)

    sampling_params = SamplingParams(ignore_eos=True, max_tokens=16)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    scheduler.get_num_new_matched_tokens.return_value = 0, False

    if save_async:
        scheduler.request_finished.return_value = True

        worker.get_finished.side_effect = lambda finished_gen, load_async: (
            finished_gen, load_async)
    else:
        scheduler.request_finished.return_value = False
        worker.get_finished.return_value = [], []

    result = generate_and_wait(prefill_worker,
                               scheduler,
                               worker, [0] * 48,
                               sampling_params=sampling_params,
                               disaggregated_params=disaggregated_params)

    gen_disagg_params = result.disaggregated_params
    gen_disagg_params.request_type = "generation_only"

    generate_and_wait(decode_worker,
                      scheduler,
                      worker, [0] * 48,
                      sampling_params=sampling_params,
                      disaggregated_params=gen_disagg_params)

    assert scheduler.build_connector_meta.call_count == 1

    scheduler_output = scheduler.build_connector_meta.call_args.args[0]

    assert len(scheduler_output.new_requests) == 1
    assert len(scheduler_output.cached_requests) == 0

    req = scheduler_output.new_requests[0]

    assert req.computed_position == 0
    assert req.num_scheduled_tokens == 48
    assert len(req.new_tokens) == 48

    assert scheduler.request_finished.call_count == 1


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_multi_request(enforce_single_worker, model_with_connector):
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True,
                     kv_cache_config=KvCacheConfig(max_tokens=144))

    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    scheduler.request_finished.return_value = True
    worker.get_finished.side_effect = lambda finished_gen, load_async: (
        finished_gen, load_async)

    model.generate([[0] * 48, [1] * 48],
                   sampling_params=[
                       SamplingParams(ignore_eos=True, max_tokens=4),
                       SamplingParams(ignore_eos=True, max_tokens=3)
                   ])

    # The KV cache of both prior requests should be freed, allowing the third request to run.
    model.generate([2] * 110, sampling_params=sampling_params)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [
    pytest.param(False, id="kv_cache_manager_v1"),
    pytest.param(
        True,
        id="kv_cache_manager_v2",
        marks=pytest.mark.xfail(
            strict=True,
            reason=
            "KvCacheRetentionConfig does not reach KVCacheManagerV2 at all "
            "(per-page priority comes from custom_priority_callback, which "
            "V2 never overrides), so the connector reports priorities=None."),
    ),
],
                         indirect=True)
def test_connector_priorities(enforce_single_worker, model_with_connector):
    """Test that retention priorities flow through the connector correctly.

    This test verifies that when KvCacheRetentionConfig is provided,
    the RequestData.priorities field is populated with the correct
    per-block priorities based on the token ranges.

    KNOWN GAP -- `xfail(strict=True)` on `kv_cache_manager_v2`.
    `KvCacheRetentionConfig` does not reach KVCacheManagerV2 at all: V2's
    per-page priority comes from `custom_priority_callback`
    (kv_cache_manager_v2/_core/_kv_cache_manager.py), which KVCacheManagerV2
    never overrides, so every page carries the default priority and the
    connector reports `priorities=None`. A user who sets a retention config on
    V2 silently gets none of it -- not only through the connector. The
    assertions below stay the correct expectation for both managers rather than
    being relaxed per manager, so wiring retention into V2 turns this green
    instead of needing the test rewritten; `strict=True` is what makes it fail
    loudly on that day rather than passing silently.
    """
    BLOCK_SIZE = 32
    NUM_INPUT_TOKENS = 64  # 2 blocks
    NUM_TOKENS = 4
    HIGH_PRIORITY = 80  # For system prompt blocks
    LOW_PRIORITY = 10  # For user input / decode blocks

    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True)

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    worker.get_finished.return_value = [], []

    # Create retention config with different priorities for different token ranges:
    # - First 32 tokens (block 0): high priority (e.g., system prompt)
    # - Remaining tokens (block 1+): low priority (e.g., user input)
    retention_config = KvCacheRetentionConfig(
        token_range_retention_configs=[
            KvCacheRetentionConfig.TokenRangeRetentionConfig(
                token_start=0,
                token_end=32,
                priority=HIGH_PRIORITY,
            ),
            KvCacheRetentionConfig.TokenRangeRetentionConfig(
                token_start=32,
                token_end=None,  # Extend to end of sequence
                priority=LOW_PRIORITY,
            ),
        ],
        decode_retention_priority=LOW_PRIORITY,
    )

    sampling_params = SamplingParams(max_tokens=NUM_TOKENS, ignore_eos=True)

    generate_and_wait(model,
                      scheduler,
                      worker, [0] * NUM_INPUT_TOKENS,
                      sampling_params=sampling_params,
                      kv_cache_retention_config=retention_config)

    # Verify that build_connector_meta was called
    assert scheduler.build_connector_meta.call_count >= 1

    # Check the first call (new request) has priorities set
    first_call = scheduler.build_connector_meta.call_args_list[0]
    sched_output = first_call.args[0]

    assert len(sched_output.new_requests) == 1
    request = sched_output.new_requests[0]

    # Should have 2 blocks for 64 input tokens with block size 32
    expected_num_blocks = math.ceil(NUM_INPUT_TOKENS / BLOCK_SIZE)
    assert len(request.new_block_ids) == expected_num_blocks

    # Priorities should be set and match the retention config
    assert request.priorities is not None
    assert len(request.priorities) == len(request.new_block_ids)

    # First block should have high priority, second block should have low priority
    assert request.priorities[
        0] == HIGH_PRIORITY, f"Expected priority {HIGH_PRIORITY} for block 0, got {request.priorities[0]}"
    assert request.priorities[
        1] == LOW_PRIORITY, f"Expected priority {LOW_PRIORITY} for block 1, got {request.priorities[1]}"


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_priorities_default(enforce_single_worker,
                                      model_with_connector):
    """Test that priorities are None when no retention config is provided."""
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True)

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    worker.get_finished.return_value = [], []

    sampling_params = SamplingParams(max_tokens=4, ignore_eos=True)

    # Generate without retention config
    generate_and_wait(model,
                      scheduler,
                      worker, [0] * 48,
                      sampling_params=sampling_params)

    first_call = scheduler.build_connector_meta.call_args_list[0]
    sched_output = first_call.args[0]

    assert len(sched_output.new_requests) == 1
    request = sched_output.new_requests[0]

    # Without retention config, priorities should be None
    assert request.priorities is None


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "llm_kwargs,match_v1,match_v2",
    [
        # The two managers refuse offloading for the same reason -- a page that
        # leaves GPU has its slot reassigned, invalidating what the connector
        # registered -- but say so differently, and V2 names the resolved tier
        # rather than the config field so it also catches the disk tier. Match
        # the manager-specific wording: both messages contain the bare word
        # "host", so matching that would still pass if a silent fallback to V1
        # ever crept back in, which is the one thing this parametrization
        # exists to rule out.
        pytest.param(
            dict(kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1,
                                               host_cache_size=1024**3)),
            "host offloading",
            "cache tiers below GPU",
            id="host_offloading",
        ),
        pytest.param(
            dict(max_beam_width=2),
            "beam",
            "beam",
            id="beam_search",
        ),
        pytest.param(
            dict(enable_attention_dp=True),
            "attention data parallelism",
            "attention data parallelism",
            id="attention_dp",
        ),
    ],
)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_rejects_unsupported_config(enforce_single_worker,
                                              model_with_connector,
                                              use_kv_cache_manager_v2,
                                              llm_kwargs, match_v1, match_v2):
    # Configurations the connector cannot handle today must fail loudly at
    # construction time rather than silently miscompute. This pins the set of
    # constructor-time exclusions in `_maybe_init_kv_connector_manager`.
    model_fn, _, _ = model_with_connector
    match = match_v2 if use_kv_cache_manager_v2 else match_v1

    with pytest.raises(NotImplementedError, match=match):
        model_fn(**llm_kwargs)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_e2e_persistent_cache(enforce_single_worker,
                                        use_kv_cache_manager_v2, monkeypatch):
    """End-to-end KV connector test using PersistentKvCacheConnector from examples.

    Runs the same prompt through two separate LLM instances sharing a
    disk-backed connector cache and asserts that:

      1. the first (cold) run matches nothing and writes cache files,
      2. the second (warm) run actually reads blocks back from disk, and
      3. both runs produce identical text and token ids.

    (3) on its own proves nothing - two deterministic runs of the same prompt
    agree whether or not the cache is ever consulted - so (2) is what makes
    this a real correctness test rather than a tautology.
    """
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "..", "examples", "llm-api")
    examples_dir = os.path.abspath(examples_dir)
    sys.path.insert(0, examples_dir)

    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CONNECTOR_CACHE_FOLDER", cache_dir)

    try:
        import llm_kv_cache_connector

        # Record how many tokens the connector served from disk on each run.
        # The leader logs this, but the TensorRT-LLM logger does not propagate
        # to the root logger, so read it from the return value instead.
        matched_tokens = []
        leader_cls = llm_kv_cache_connector.PersistentKvCacheConnectorLeader
        original_get_num_new_matched_tokens = (
            leader_cls.get_num_new_matched_tokens)

        def recording_get_num_new_matched_tokens(self, request,
                                                 num_computed_tokens):
            result = original_get_num_new_matched_tokens(
                self, request, num_computed_tokens)
            matched_tokens.append(result[0])
            return result

        monkeypatch.setattr(leader_cls, "get_num_new_matched_tokens",
                            recording_get_num_new_matched_tokens)

        kv_connector_config = KvCacheConnectorConfig(
            connector_module="llm_kv_cache_connector",
            connector_scheduler_class="PersistentKvCacheConnectorLeader",
            connector_worker_class="PersistentKvCacheConnectorWorker",
        )

        llm_kwargs = dict(
            model=f"{llm_models_root()}/Qwen3/Qwen3-0.6B",
            backend="pytorch",
            kv_connector_config=kv_connector_config,
            cuda_graph_config=None,
            disable_overlap_scheduler=True,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.1,
                use_kv_cache_manager_v2=use_kv_cache_manager_v2),
        )

        prompt = (
            "Nvidia Corporation is an American technology company "
            "headquartered in Santa Clara, California. Founded in 1993 by "
            "Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops "
            "graphics processing units (GPUs), system on a chips (SoCs), and "
            "application programming interfaces (APIs) for data science, "
            "high-performance computing, and mobile and automotive "
            "applications. Tell me about the company.")

        sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)

        llm1 = LLM(**llm_kwargs)
        try:
            output1 = llm1.generate([prompt], sampling_params)
            cold_text = output1[0].outputs[0].text
            cold_token_ids = list(output1[0].outputs[0].token_ids)
        finally:
            llm1.shutdown()

        assert matched_tokens and all(count == 0 for count in matched_tokens), (
            "The first run should be a cold miss, but the connector reported "
            f"matched token counts {matched_tokens}. The cache directory was "
            "not clean, so the comparison below is meaningless.")

        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(cache_files) > 0, "No cache files written by connector"

        matched_tokens.clear()

        llm2 = LLM(**llm_kwargs)
        try:
            output2 = llm2.generate([prompt], sampling_params)
            warm_text = output2[0].outputs[0].text
            warm_token_ids = list(output2[0].outputs[0].token_ids)
        finally:
            llm2.shutdown()

        assert matched_tokens and max(matched_tokens) > 0, (
            "The second run read nothing back from the connector cache "
            f"(matched token counts {matched_tokens}), so the comparisons "
            "below would pass just as well with the connector disabled.")

        assert len(warm_token_ids) == len(cold_token_ids), (
            f"Generation length changed: cold {len(cold_token_ids)} tokens, "
            f"warm {len(warm_token_ids)} tokens.")

        # Exact equality is NOT asserted. Reusing cached KV skips prefill for
        # the matched blocks, which changes the attention reduction order, so
        # the logits differ in the last bits even though the restored K/V are
        # bit-identical (the connector round-trips them through torch.save /
        # torch.load). Greedy decoding turns a near-tie into a different token.
        # Observed on V1: the two runs agreed on 31 of 32 tokens and split on
        # the final one ("The company's" vs "The company is").
        #
        # A corrupted or misaddressed cache does not look like that - it
        # diverges early and degenerates - so requiring a long common prefix
        # keeps the test meaningful without making it a coin flip.
        common_prefix = 0
        for cold_id, warm_id in zip(cold_token_ids, warm_token_ids):
            if cold_id != warm_id:
                break
            common_prefix += 1

        min_common_prefix = math.floor(
            len(cold_token_ids) * E2E_MIN_TOKEN_AGREEMENT)
        assert common_prefix >= min_common_prefix, (
            f"Connector cache reuse diverged at token {common_prefix} of "
            f"{len(cold_token_ids)}, below the {min_common_prefix}-token "
            "floor. Early divergence indicates the restored KV is wrong, not "
            "just numerically different.\n"
            f"  cold run: {cold_text!r}\n"
            f"  warm run: {warm_text!r}\n"
            f"  cold ids: {cold_token_ids}\n"
            f"  warm ids: {warm_token_ids}")
    finally:
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)

        shutil.rmtree(cache_dir, ignore_errors=True)
