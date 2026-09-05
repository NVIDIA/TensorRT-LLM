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

import hashlib
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

from ..conftest import get_sm_version, llm_models_root

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
# `get_page_indices_by_layer_group` instead.
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

        # A cache that reports per layer group always calls the per-group form,
        # so mirror what a real connector's default does with it: fold a single
        # group back onto the flat callback, and leave several groups to a test
        # that asserts the per-group form directly. Without this every mock
        # would answer the per-group call with a truthy `Mock`, which reads as
        # "saving asynchronously" and parks every request.
        def _alloc_by_group(request, block_ids_by_layer_group):
            if len(block_ids_by_layer_group) == 1:
                mock_scheduler.update_state_after_alloc(
                    request, block_ids_by_layer_group[0])

        def _finished_by_group(request, cache_block_ids_by_layer_group):
            if len(cache_block_ids_by_layer_group) == 1:
                return mock_scheduler.request_finished(
                    request, cache_block_ids_by_layer_group[0])
            return False

        mock_scheduler.update_state_after_alloc_by_layer_group.side_effect = (
            _alloc_by_group)
        mock_scheduler.request_finished_by_layer_group.side_effect = (
            _finished_by_group)

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

    # The other half of the contract: what V2 offers instead. Every connector
    # path on V2 goes through this one accessor - `update_and_build_data`,
    # `kv_connector_request_finished` and `_run_kv_connector_hooks` each call it
    # and derive the flat list from `[0]` when there is a single layer group.
    assert hasattr(KVCacheManagerV2, "get_page_indices_by_layer_group"), (
        "KVCacheManagerV2.get_page_indices_by_layer_group is the V2 "
        "replacement for the V1 block-id accessors and every connector path on "
        "V2 goes through it.")
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
        assert len(req.new_block_ids_by_layer_group) == 1
        assert req.new_block_ids_by_layer_group[0] == req.new_block_ids
    else:
        assert req.new_block_ids_by_layer_group == []


# Half the prompt, and well past the window, so the pages the offer does *not*
# need are a large enough fraction to assert on.
SWA_OFFER_TOKENS = 128


# The mock scheduler answers every query with the same offer; this records what
# it was asked and when, so a test can assert the connector was consulted
# exactly once per request rather than assuming it.
def record_connector_queries(scheduler, num_matched, load_async=False):
    """Answer every query with `num_matched`, recording when each one arrived.

    The third element of each record is `build_connector_meta.call_count` at
    query time -- the number of iterations whose connector hooks had already
    run. The connector is asked from `prepare_resources`, in the same iteration
    the request runs, so a request that runs in iteration `n` records `n`. A
    speculative ask would record a smaller number.
    """
    queries = []

    def side_effect(request, num_computed_tokens):
        queries.append((request.request_id, num_computed_tokens,
                        scheduler.build_connector_meta.call_count))
        return num_matched, load_async

    scheduler.get_num_new_matched_tokens.side_effect = side_effect
    return queries


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_prefix_is_asked_once_and_shrinks_the_forward_pass(
        enforce_single_worker, model_with_connector, use_kv_cache_manager_v2):
    """The offer is asked for once and removes work from the forward pass.

    Both managers ask on the batch that runs, so both record the same query
    count and the same iteration index, and both report a prefill shortened by
    the offer. This is the parity assertion for A0: a connector cannot tell the
    two managers apart from what it is asked and what it is told.
    """
    model_fn, scheduler, worker = model_with_connector

    OFFER_TOKENS = 64
    NUM_INPUT_TOKENS = 256

    model = model_fn(disable_overlap_scheduler=True)

    queries = record_connector_queries(scheduler, OFFER_TOKENS)
    worker.get_finished.return_value = [], []

    generate_and_wait(model, scheduler, worker, [0] * NUM_INPUT_TOKENS,
                      SamplingParams(max_tokens=4, ignore_eos=True))

    # Once per request, anchored at the local match, in the first iteration --
    # which is also the iteration the request ran in.
    assert len(queries) == 1
    _, num_computed_tokens, iterations_before = queries[0]
    assert num_computed_tokens == 0
    assert iterations_before == 0

    req = scheduler.build_connector_meta.call_args_list[0].args[0].new_requests[
        0]
    # The runtime rolls the reported position back to the local match, so the
    # connector sees where its load begins; the offer shows up as work removed.
    assert req.computed_position == 0
    assert req.num_scheduled_tokens == NUM_INPUT_TOKENS - OFFER_TOKENS


# Chunked-prefill prefix sizes, chosen either side of the chunk boundary so the
# two arms of the fit are both exercised: 64 lands inside the first chunk and
# only moves its start, 192 lands past it and has to shift the chunk window and
# grow the allocation.
CHUNKED_PREFIX_CHUNK_SIZE = 128
CHUNKED_PREFIX_NUM_INPUT_TOKENS = 256


def _context_reports(scheduler):
    """`(computed_position, num_scheduled_tokens)` per prefill iteration.

    A generation iteration schedules exactly one token, which is what separates
    the two here. Each `build_connector_meta` call is one iteration and carries
    one request, in `new_requests` on its first report and `cached_requests`
    after.
    """
    reports = []
    for call in scheduler.build_connector_meta.call_args_list:
        sched_output = call.args[0]
        requests = sched_output.new_requests + sched_output.cached_requests
        assert len(requests) == 1
        req = requests[0]
        if req.num_scheduled_tokens > 1:
            reports.append((req.computed_position, req.num_scheduled_tokens))
    return reports


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("offer_tokens", [64, 192],
                         ids=["offer_inside_chunk", "offer_past_chunk"])
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_prefix_under_chunked_prefill(enforce_single_worker,
                                                model_with_connector,
                                                use_kv_cache_manager_v2,
                                                offer_tokens):
    """A served prefix removes exactly its own tokens without breaking chunking.

    Chunked prefill is where the two managers allocate differently -- one covers
    the whole prompt when the sequence is added, the other only the chunk it is
    about to compute -- so an offer reaching past the chunk is the case where a
    prefix can outrun the pages that exist. What a connector observes has to be
    the same either way: asked once at the local match, a prefill shortened by
    the whole offer, and no chunk larger than `max_num_tokens`.
    """
    model_fn, scheduler, worker = model_with_connector

    model = model_fn(disable_overlap_scheduler=True,
                     enable_chunked_prefill=True,
                     max_num_tokens=CHUNKED_PREFIX_CHUNK_SIZE)

    assert_kv_caches_registered(worker, use_kv_cache_manager_v2)

    queries = record_connector_queries(scheduler, offer_tokens)
    worker.get_finished.return_value = [], []

    generate_and_wait(model, scheduler, worker,
                      [0] * CHUNKED_PREFIX_NUM_INPUT_TOKENS,
                      SamplingParams(max_tokens=4, ignore_eos=True))

    # Once per request, anchored at the local match, in the iteration the
    # request ran in -- chunking must not turn one allocation into one ask per
    # chunk.
    assert len(queries) == 1
    _, num_computed_tokens, iterations_before = queries[0]
    assert num_computed_tokens == 0
    assert iterations_before == 0

    reports = _context_reports(scheduler)
    assert reports, "no prefill iteration was reported to the connector"

    # The reported position starts at the local match, so the connector knows
    # where its load begins rather than where the runtime resumed.
    assert reports[0][0] == 0

    # Every prefill chunk still fits the token budget. Honouring an offer past
    # the chunk moves the window, it does not widen it.
    assert all(scheduled <= CHUNKED_PREFIX_CHUNK_SIZE
               for _, scheduled in reports), reports

    # The offer removed exactly its own tokens from the forward pass, whether it
    # was honoured in one chunk or spread across several.
    total_scheduled = sum(scheduled for _, scheduled in reports)
    assert total_scheduled == CHUNKED_PREFIX_NUM_INPUT_TOKENS - offer_tokens

    # Positions are contiguous over what was actually computed. Only the first
    # report is rolled back to the local match, so from the second onwards each
    # one resumes where the previous chunk stopped, offer included. How the
    # first chunk is sized is left to the manager: covering the whole prompt up
    # front leaves the chunk at its full width, allocating per chunk keeps the
    # end where the scheduler put it and only moves the start.
    expected = offer_tokens + reports[0][1]
    for position, scheduled in reports[1:]:
        assert position == expected, reports
        expected += scheduled

    # The save at the end covers the whole prompt, offer included -- the served
    # range is not silently dropped from what the connector is told to persist.
    assert len(scheduler.request_finished.call_args.args[1]) >= math.ceil(
        CHUNKED_PREFIX_NUM_INPUT_TOKENS / 32)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [True],
                         ids=["kv_cache_manager_v2"],
                         indirect=True)
def test_connector_sliding_window_prefix_is_backed_by_real_pages(
        enforce_single_worker, model_with_connector, use_kv_cache_manager_v2):
    """A served prefix is reported as pages for the window and `-1` before it.

    The offer reaches back past the window, so the served range splits: blocks
    inside the live window carry real distinct pages, and blocks the window has
    passed carry `BAD_PAGE_INDEX`. Skipping the compute for the whole offer
    while moving bytes only for the window is the point of the design, and the
    split is what a connector has to see to do it.

    The block reuse policy is deliberately left at its default. It decides
    whether an out-of-window page survives the `history_length` bump as a
    holder, which is exactly what the reported list must not depend on.
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
    # One entry per block ordinal either way -- an out-of-window block reads back
    # as BAD_PAGE_INDEX in place rather than shortening the list, so that block
    # ordinals stay aligned to token ranges. The count is therefore not the
    # signal; *which* entries are bad is.
    assert len(page_indices) == all_blocks

    # The stale end the manager masks to, recomputed here from the sizes rather
    # than read back, so a change to either one fails instead of adapting.
    stale_end = max(0, (SWA_OFFER_TOKENS + 1 - SWA_WINDOW) // 32)
    assert 0 < stale_end < SWA_OFFER_TOKENS // 32, (
        "test sizes no longer split the served range across the window edge")

    assert all(index == BAD_PAGE_INDEX for index in page_indices[:stale_end]), (
        f"a block the window has passed was reported as a page: {page_indices}")
    live = page_indices[stale_end:]
    assert all(index != BAD_PAGE_INDEX for index in live), (
        f"an in-window block was reported with no page: {page_indices}")
    assert len(set(live)) == len(live), (
        f"page slots reported to the connector are not distinct: {page_indices}"
    )


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
    is scoped to a layer group, and several groups cannot be flattened into one
    list. Every block-id callback therefore switches to its per-layer-group
    form, on the way in and on the way out.

    Both halves are asserted here because the rejection is deliberately
    conditional on the manager. Pinning only the V2 half would let the V1 guard
    silently disappear.
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

    # The flat list is empty with several groups, and the per-group list is what
    # carries the pages. A connector reading only `new_block_ids` sees nothing,
    # which is why implementing the per-group callbacks is required at bring-up
    # for this model -- `MagicMock` satisfies that check.
    assert scheduler.update_state_after_alloc_by_layer_group.call_count == 1
    by_group = scheduler.update_state_after_alloc_by_layer_group.call_args.args[
        1]
    assert len(by_group) == 2

    sched_output = scheduler.build_connector_meta.call_args_list[0].args[0]
    assert len(sched_output.new_requests) == 1
    req = sched_output.new_requests[0]

    assert req.new_block_ids == []
    assert len(req.new_block_ids_by_layer_group) == 2
    # Ordinals stay positionally aligned across groups: a block with no page in
    # the sliding group, or one the window has passed, reads back as
    # BAD_PAGE_INDEX in place rather than shortening the list.
    lengths = {len(indices) for indices in req.new_block_ids_by_layer_group}
    assert len(lengths) == 1

    # The save direction switches with it. Without the per-group form the
    # connector is handed an empty flat list at the end of the request and can
    # persist nothing at all, which no other assertion here would catch.
    assert scheduler.request_finished.call_count == 0
    assert scheduler.request_finished_by_layer_group.call_count == 1
    saved = scheduler.request_finished_by_layer_group.call_args.args[1]
    assert len(saved) == 2

    # Whatever the sliding group offers to save sits inside its window. The
    # full-attention group keeps everything, which is what makes the comparison
    # non-vacuous.
    windows = [group.window_size for group in layout.groups]
    sliding = windows.index(SWA_WINDOW)
    full = windows.index(None)
    live = [
        len([index for index in saved[group] if index != BAD_PAGE_INDEX])
        for group in (sliding, full)
    ]
    assert live[0] <= math.ceil(SWA_WINDOW / 32) + 1, (
        f"the sliding group offered more than its window to save: {saved[sliding]}"
    )
    assert live[1] > live[0], (
        f"the full-attention group should keep more than the sliding one: {saved}"
    )


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
    preference, so the default resolves to the C++ transceiver, which V2 cannot
    use.
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
@pytest.mark.parametrize("use_kv_cache_manager_v2", [True],
                         ids=["kv_cache_manager_v2"],
                         indirect=True)
def test_connector_warns_that_retention_is_ignored_on_v2(
        enforce_single_worker, model_with_connector, caplog):
    """A retention config that has no effect must say so.

    `KvCacheRetentionConfig` does not reach KVCacheManagerV2 at all, so the
    connector reports `priorities=None` there. `test_connector_priorities`
    pins that gap as `xfail(strict=True)`; this pins the diagnostic, which is
    the only thing standing between a user's retention config and it being
    dropped in silence.

    V2 only: on V1 the config is honoured and there is nothing to warn about.
    """
    model_fn, scheduler, worker = model_with_connector
    model = model_fn(disable_overlap_scheduler=True)

    scheduler.get_num_new_matched_tokens.return_value = 0, False
    worker.get_finished.return_value = [], []

    retention_config = KvCacheRetentionConfig(token_range_retention_configs=[
        KvCacheRetentionConfig.TokenRangeRetentionConfig(token_start=0,
                                                         token_end=32,
                                                         priority=80)
    ],
                                              decode_retention_priority=10)

    # The TensorRT-LLM logger sets `propagate = False`, so caplog only sees its
    # records once its handler is attached to that logger by name.
    trtllm_logger = logging.getLogger(TRTLLM_LOGGER_NAME)
    trtllm_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            generate_and_wait(model,
                              scheduler,
                              worker, [0] * 64,
                              sampling_params=SamplingParams(max_tokens=4,
                                                             ignore_eos=True),
                              kv_cache_retention_config=retention_config)
    finally:
        trtllm_logger.removeHandler(caplog.handler)

    assert "KvCacheRetentionConfig has no effect" in caplog.text, (
        "A retention config was set on KVCacheManagerV2 and nothing said it "
        "would be ignored. The user's configuration is silently dropped:\n"
        f"{caplog.text}")

    # The other half: the warning describes what actually happened.
    request = scheduler.build_connector_meta.call_args_list[0].args[
        0].new_requests[0]
    assert request.priorities is None, (
        "priorities are populated on V2 after all, so the warning is now "
        "wrong -- revisit it together with `test_connector_priorities`.")


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
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True],
                         ids=["kv_cache_manager_v1", "kv_cache_manager_v2"],
                         indirect=True)
def test_connector_block_reuse_off_is_rejected_on_v2_only(
        enforce_single_worker, model_with_connector, use_kv_cache_manager_v2):
    """`enable_block_reuse=False` with a connector: V2 refuses, V1 does not.

    The asymmetry is deliberate and both halves are pinned here, because each
    is wrong on its own.

    **V2 must refuse.** It honours a connector-served prefix whatever
    `enable_block_reuse` says -- that flag governs the local radix tree, and the
    connector is a separate source -- and with reuse off the restored KV is
    wrong. Measured with the reference connector on a plain full-attention
    model: the offer is honoured (18 of 82 tokens scheduled rather than 82) and
    generation drifts within a few tokens. Refusing at startup is what turns
    silent wrong output into a message.

    **V1 must not refuse.** It never honours the offer in this configuration --
    it asks the connector, then schedules all 82 tokens anyway -- so nothing
    miscomputes and there is no correctness reason to reject a setup that
    works. It does waste the connector's lookup and hand it a negative
    `computed_position`; that is a separate defect, tracked in the backlog
    rather than papered over with a guard here.

    Asserting V1 still constructs is the load-bearing half: a guard written
    against the config instead of the manager would reject both, and this is
    what says so.
    """
    model_fn, _, _ = model_with_connector
    llm_kwargs = dict(kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.1, enable_block_reuse=False))

    if use_kv_cache_manager_v2:
        with pytest.raises(NotImplementedError, match="block reuse disabled"):
            model_fn(**llm_kwargs)
    else:
        model = model_fn(**llm_kwargs)
        model.shutdown()


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
    # sys.path, not __extra_import_path__: the connector module is imported by
    # name from inside tensorrt_llm, not by this file. monkeypatch restores the
    # entry at teardown so a failure cannot leak it into later tests.
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "..", "examples", "llm-api")
    examples_dir = os.path.abspath(examples_dir)
    monkeypatch.syspath_prepend(examples_dir)

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


# The VSWA end-to-end sizes. The window is deliberately larger than the whole
# run (prompt + generation), so nothing goes out of window and the save/load
# round trip is the only thing under test. `test_connector_vswa_reports_page_
# indices_per_layer_group` covers the routing when blocks *do* go stale, and
# `test_connector_sliding_window_prefix_is_backed_by_real_pages` covers the
# masking arithmetic; mixing all three into one test would leave a failure
# ambiguous.
VSWA_E2E_WINDOW = 256
VSWA_E2E_MAX_SEQ_LEN = 512


@pytest.mark.threadleak(enabled=False)
def test_connector_multi_pool_e2e_persistent_cache(enforce_single_worker,
                                                   monkeypatch):
    """The multi-pool data path, end to end, with a connector that moves real bytes.

    Qwen3-0.6B has a single attention type; the two windows here are imposed by
    `max_attention_window`, so this covers a VSWA *cache* over a uniform model
    -- the cache plumbing, on a small fast model. The interleaved-attention
    case, where the model itself has two attention types and the layer groups
    come out different sizes, is `test_connector_vswa_e2e_gemma3`.

    Every other multi-group test in this file drives a `MagicMock` connector, so
    they assert routing and shape and never touch memory. This one runs
    `examples/llm-api/llm_kv_cache_connector_vswa.py` -- which addresses pages
    through `KvCacheLayout` regions and nothing else -- across two LLM
    instances sharing a disk cache, and asserts:

      1. the cold run writes one file per (block, layer group), not per block,
      2. no two of those files hold the same bytes, and
      3. the warm run reads blocks back and reproduces the cold run's tokens.

    (1) is what fails if `layer_group_id` is dropped from the cache key: the
    groups collide and half the files disappear. (2) is what fails if the
    layout reports the same pool base for both groups -- the connector would
    then read one group's pages twice and write identical bytes under two
    names, while every shape assertion still passed. (3) is what fails if the
    page slots address the wrong pool on the way back in.

    V2 only: VSWA has no connector path on the V1 manager, which registers a
    single primary pool.
    """
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "examples", "llm-api"))
    sys.path.insert(0, examples_dir)

    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CONNECTOR_CACHE_FOLDER", cache_dir)

    try:
        import llm_kv_cache_connector_vswa as vswa

        leader_cls = vswa.VswaKvCacheConnectorLeader

        matched_tokens = []
        original_query = leader_cls.get_num_new_matched_tokens

        def recording_query(self, request, num_computed_tokens):
            result = original_query(self, request, num_computed_tokens)
            matched_tokens.append(result[0])
            return result

        saves = []
        original_build = leader_cls.build_connector_meta

        def recording_build(self, scheduler_output):
            metadata = original_build(self, scheduler_output)
            saves.extend(metadata.save)
            return metadata

        monkeypatch.setattr(leader_cls, "get_num_new_matched_tokens",
                            recording_query)
        monkeypatch.setattr(leader_cls, "build_connector_meta", recording_build)

        def build():
            return vswa.build_llm(
                model=f"{llm_models_root()}/Qwen3/Qwen3-0.6B",
                # Two distinct windows: one sliding layer group and one
                # full-attention layer group. An entry equal to max_seq_len
                # normalizes to None.
                max_attention_window=[VSWA_E2E_WINDOW, VSWA_E2E_MAX_SEQ_LEN],
                max_seq_len=VSWA_E2E_MAX_SEQ_LEN,
                free_gpu_memory_fraction=0.1,
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

        llm1 = build()
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

        # (1) One save per (block ordinal, layer group). Two groups, so the
        # ordinals must repeat exactly twice and every key must be distinct.
        assert saves, "the connector saved nothing on the cold run"
        groups = {group_id for _, group_id, _ in saves}
        assert groups == {
            0, 1
        }, (f"expected saves in both layer groups, got groups {sorted(groups)}. "
            "The run is not VSWA, so nothing below tests per-group addressing.")
        paths = [path for path, _, _ in saves]
        assert len(set(paths)) == len(paths), (
            "two saves shared a cache key. The layer group is missing from the "
            "key, so one group's KV overwrites the other's:\n"
            f"  {sorted(paths)}")
        per_group = {group_id: 0 for group_id in groups}
        for _, group_id, _ in saves:
            per_group[group_id] += 1
        assert len(set(per_group.values())) == 1, (
            f"layer groups saved different numbers of blocks: {per_group}. "
            "Block ordinals must stay aligned across groups.")

        cache_files = sorted(f for f in os.listdir(cache_dir)
                             if f.endswith(".pt"))
        assert len(cache_files) == len(set(paths)), (
            f"{len(set(paths))} distinct save keys produced {len(cache_files)} "
            "files on disk.")

        # (2) Distinct bytes per file. Identical content across two groups
        # would mean the connector read the same pool twice.
        digests = {}
        for name in cache_files:
            with open(os.path.join(cache_dir, name), "rb") as handle:
                digests.setdefault(
                    hashlib.sha256(handle.read()).hexdigest(), []).append(name)
        collisions = {
            digest: names
            for digest, names in digests.items() if len(names) > 1
        }
        assert not collisions, (
            "two cache files hold identical bytes, so the same pool was read "
            "for more than one (block, layer group). Layer group g's page "
            f"slots are not addressing layer group g's pool: {collisions}")

        matched_tokens.clear()

        llm2 = build()
        try:
            output2 = llm2.generate([prompt], sampling_params)
            warm_text = output2[0].outputs[0].text
            warm_token_ids = list(output2[0].outputs[0].token_ids)
        finally:
            llm2.shutdown()

        # (3) The load direction. Without this the test proves only that bytes
        # were written somewhere.
        assert matched_tokens and max(matched_tokens) > 0, (
            "The second run read nothing back from the connector cache "
            f"(matched token counts {matched_tokens}), so the comparison "
            "below would pass just as well with the connector disabled.")

        assert len(warm_token_ids) == len(cold_token_ids), (
            f"Generation length changed: cold {len(cold_token_ids)} tokens, "
            f"warm {len(warm_token_ids)} tokens.")

        # Exact equality is not asserted, for the reason spelled out in
        # test_connector_e2e_persistent_cache: skipping prefill for the matched
        # blocks changes the attention reduction order, so greedy decoding can
        # split on a near-tie. Misaddressed KV diverges early instead.
        common_prefix = 0
        for cold_id, warm_id in zip(cold_token_ids, warm_token_ids):
            if cold_id != warm_id:
                break
            common_prefix += 1

        min_common_prefix = math.floor(
            len(cold_token_ids) * E2E_MIN_TOKEN_AGREEMENT)
        assert common_prefix >= min_common_prefix, (
            f"VSWA connector cache reuse diverged at token {common_prefix} of "
            f"{len(cold_token_ids)}, below the {min_common_prefix}-token "
            "floor. Early divergence means the restored KV went into the "
            "wrong layer group's pages, not that it is numerically "
            "different.\n"
            f"  cold run: {cold_text!r}\n"
            f"  warm run: {warm_text!r}\n"
            f"  cold ids: {cold_token_ids}\n"
            f"  warm ids: {warm_token_ids}")
    finally:
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)


# Gemma-3-1B interleaves sliding and full attention on a 6-layer cycle
# (`sliding_window_pattern: 6`, `sliding_window: 512`), so 26 layers split
# 22 sliding / 4 full. That uneven split is the thing a forced
# `max_attention_window` on a uniform model cannot produce, and it is what
# makes this an interleaved-attention run rather than a VSWA cache config.
GEMMA3_SLIDING_WINDOW = 512
GEMMA3_GLOBAL_WINDOW = 32768
GEMMA3_NUM_LAYERS = 26
GEMMA3_CYCLE = 6
GEMMA3_MAX_SEQ_LEN = 2048


@pytest.mark.threadleak(enabled=False)
def test_connector_vswa_e2e_gemma3(enforce_single_worker, monkeypatch):
    """A KV connector on a model whose architecture is variable-window.

    `test_connector_multi_pool_e2e_persistent_cache` imposes two windows on a
    uniform-attention model, which exercises the cache plumbing on something
    small and fast. This runs Gemma-3-1B, which interleaves sliding and full
    attention natively, and which asks for `KVCacheManagerV2` itself
    (`Gemma3ForCausalLM.get_preferred_kv_cache_manager_version` returns "V2"
    for exactly this layout). Before this series that combination was rejected
    at bring-up, so it is the case the change exists to enable.

    `use_kv_cache_manager_v2` is left at "auto" on purpose: the model's own
    preference has to be what lands the run on V2, or a user gets the rejection
    without knowing to override anything. The assertions check that it did.

    Two things are asserted that the uniform-model test cannot show:

    1. **The layer groups are different sizes.** 22 layers slide and 4 do not.
       A forced window list on a uniform model splits evenly, so a builder that
       partitioned by layer index rather than by window would pass there and
       fail here.
    2. **Every layer is covered exactly once**, against the model's real layer
       count rather than a number the test chose.

    Then the same save/load round trip: one cache file per (block, layer
    group), no two files holding the same bytes, and a warm run that reads
    blocks back and reproduces the cold run's tokens.

    The warm run is a separate process, so its radix tree starts empty and a
    warm hit can only have come from the connector.

    Scope: the sliding window never engages here. The whole run fits inside the
    512-token window, so this covers per-layer-group addressing and the save /
    load round trip on a model with two real attention types, not the
    out-of-window `-1` path. See the comment on `enable_block_reuse` below.
    """
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "examples", "llm-api"))
    sys.path.insert(0, examples_dir)

    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CONNECTOR_CACHE_FOLDER", cache_dir)

    try:
        import llm_kv_cache_connector_vswa as vswa

        leader_cls = vswa.VswaKvCacheConnectorLeader
        worker_cls = vswa.VswaKvCacheConnectorWorker

        matched_tokens = []
        original_query = leader_cls.get_num_new_matched_tokens

        def recording_query(self, request, num_computed_tokens):
            result = original_query(self, request, num_computed_tokens)
            matched_tokens.append(result[0])
            return result

        saves = []
        # `num_scheduled_tokens` is how many tokens the upcoming forward pass
        # will compute. It is the only thing separating "the connector was
        # asked" from "the connector's answer was used": if the offer were
        # ignored the prompt would just be recomputed, and recomputation
        # produces the same tokens as the cold run. Matching output therefore
        # cannot show the prefix was honoured, and this can.
        scheduled = []
        original_build = leader_cls.build_connector_meta

        def recording_build(self, scheduler_output):
            metadata = original_build(self, scheduler_output)
            saves.extend(metadata.save)
            scheduled.extend((rd.num_scheduled_tokens, len(rd.new_tokens))
                             for rd in scheduler_output.new_requests)
            return metadata

        layouts = []
        original_register = worker_cls.register_kv_cache_layout

        def recording_register(self, layout):
            layouts.append(layout)
            return original_register(self, layout)

        monkeypatch.setattr(leader_cls, "get_num_new_matched_tokens",
                            recording_query)
        monkeypatch.setattr(leader_cls, "build_connector_meta", recording_build)
        monkeypatch.setattr(worker_cls, "register_kv_cache_layout",
                            recording_register)

        # The 5-sliding : 1-global cycle Gemma-3 actually uses. The final entry
        # is clamped to max_seq_len and normalizes to None, i.e. full attention.
        windows = [GEMMA3_SLIDING_WINDOW] * (GEMMA3_CYCLE - 1) + [
            GEMMA3_GLOBAL_WINDOW
        ]

        def build():
            return vswa.build_llm(
                model=f"{llm_models_root()}/gemma/gemma-3-1b-it",
                max_attention_window=windows,
                max_seq_len=GEMMA3_MAX_SEQ_LEN,
                free_gpu_memory_fraction=0.3,
                # The model asks for V2 itself; "auto" is what a user gets.
                use_kv_cache_manager_v2="auto",
                # Block reuse stays on, which needs two justifications.
                #
                # It does not weaken the test: the two runs are separate
                # processes, so the second starts with an empty radix tree and a
                # warm hit can only have come from the connector.
                #
                # It also does not walk into the WAR the Gemma-3 accuracy suite
                # carries ("gaps in kernel support for Gemma3's non-inclusive
                # sliding window size", test_llm_api_pytorch.py). That boundary
                # is only reached once a sequence outgrows the window, and this
                # one does not come close: window 512 against a prompt under
                # 96 tokens plus 32 generated. The stale range stays empty
                # throughout, so the non-inclusive boundary is never evaluated.
                # A longer prompt here would exercise the `-1` masking end to
                # end, and would also collide with that WAR -- worth doing, but
                # as its own test rather than by stretching this one.
                #
                # Turning reuse off would make the connector prefix restore
                # wrong KV on V2 (V1 is unaffected); that is a separate defect,
                # tracked in the backlog, not something for this test to carry.
                enable_block_reuse=True,
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

        llm1 = build()
        try:
            output1 = llm1.generate([prompt], sampling_params)
            cold_text = output1[0].outputs[0].text
            cold_token_ids = list(output1[0].outputs[0].token_ids)
            prompt_len = len(output1[0].prompt_token_ids)
        finally:
            llm1.shutdown()

        # Control for the honoured-prefix check below: with an empty cache the
        # cold run must schedule the whole prompt.
        cold_sched = [n for n, tok in scheduled if tok == prompt_len]
        assert cold_sched == [
            prompt_len
        ], (f"cold run scheduled {cold_sched} for a {prompt_len}-token prompt; "
            "something served a prefix on a cold cache, so the warm comparison "
            "is not measuring the connector.")

        # (0) "auto" landed on V2 and the layout path ran. Without this the
        # rest could be describing a run that never reached KVCacheManagerV2.
        assert layouts, (
            "register_kv_cache_layout was never called, so this run did not go "
            "through KVCacheManagerV2 and nothing below tests the V2 path.")
        layout = layouts[0]

        # (1) The interleave, read off the layout rather than assumed.
        assert len(layout.groups) == 2, (
            f"Gemma-3 must produce one sliding and one full-attention layer "
            f"group, got {len(layout.groups)}: "
            f"{[(g.layer_group_id, g.window_size) for g in layout.groups]}")
        by_window = {group.window_size: group for group in layout.groups}
        assert set(by_window) == {
            GEMMA3_SLIDING_WINDOW, None
        }, (f"unexpected window set {set(by_window)}; the final entry should "
            "clamp to max_seq_len and normalize to None")

        sliding = by_window[GEMMA3_SLIDING_WINDOW]
        full = by_window[None]
        expected_full = GEMMA3_NUM_LAYERS // GEMMA3_CYCLE
        assert len(full.layer_ids) == expected_full, (
            f"{len(full.layer_ids)} full-attention layers, expected "
            f"{expected_full} for a {GEMMA3_CYCLE}-layer cycle over "
            f"{GEMMA3_NUM_LAYERS} layers: {full.layer_ids}")
        assert len(sliding.layer_ids) == GEMMA3_NUM_LAYERS - expected_full
        assert len(sliding.layer_ids) != len(full.layer_ids), (
            "the two groups came out the same size, which a real interleaved "
            "model does not do -- this is the uniform-model case, not Gemma-3")

        # (2) Every layer covered exactly once, against the model's own count.
        covered = sorted(list(sliding.layer_ids) + list(full.layer_ids))
        assert covered == list(range(GEMMA3_NUM_LAYERS)), (
            f"layer coverage is not the full model: {covered}")

        # (3) Cold run: a miss, and one save per (block, layer group).
        assert matched_tokens and all(count == 0 for count in matched_tokens), (
            "The first run should be a cold miss, but the connector reported "
            f"matched token counts {matched_tokens}. The cache directory was "
            "not clean, so the comparison below is meaningless.")
        assert saves, "the connector saved nothing on the cold run"
        groups = {group_id for _, group_id, _ in saves}
        assert groups == {group.layer_group_id
                          for group in layout.groups
                          }, (f"saves reached layer groups {sorted(groups)}, "
                              "not every group in the layout")
        paths = [path for path, _, _ in saves]
        assert len(set(paths)) == len(paths), (
            "two saves shared a cache key, so the layer group is missing from "
            f"the key and one group's KV overwrites the other's:\n{sorted(paths)}"
        )

        cache_files = sorted(f for f in os.listdir(cache_dir)
                             if f.endswith(".pt"))
        assert len(cache_files) == len(set(paths))

        # (4) Distinct bytes per file. The sliding group and the full group hold
        # different layers, so identical content would mean the connector read
        # one group's pool for both.
        digests = {}
        for name in cache_files:
            with open(os.path.join(cache_dir, name), "rb") as handle:
                digests.setdefault(
                    hashlib.sha256(handle.read()).hexdigest(), []).append(name)
        collisions = {
            digest: names
            for digest, names in digests.items() if len(names) > 1
        }
        assert not collisions, (
            "two cache files hold identical bytes, so the same pool was read "
            "for more than one (block, layer group). Layer group g's page "
            f"slots are not addressing layer group g's pool: {collisions}")

        # The two groups hold different numbers of layers, so their pages differ
        # in size. That is only visible on an interleaved model.
        assert sliding.bytes_per_page != full.bytes_per_page, (
            "the two layer groups report the same bytes per page, which "
            f"{len(sliding.layer_ids)} and {len(full.layer_ids)} layers cannot "
            "both produce")

        matched_tokens.clear()
        scheduled.clear()

        llm2 = build()
        try:
            output2 = llm2.generate([prompt], sampling_params)
            warm_text = output2[0].outputs[0].text
            warm_token_ids = list(output2[0].outputs[0].token_ids)
        finally:
            llm2.shutdown()

        # (5) The load direction. Block reuse is off, so this can only be the
        # connector.
        assert matched_tokens and max(matched_tokens) > 0, (
            "The second run read nothing back from the connector cache "
            f"(matched token counts {matched_tokens}), so the comparison "
            "below would pass just as well with the connector disabled.")

        # The offer above is what the connector proposed. This is what the
        # runtime did with it: prefill must be skipped for the served range, or
        # the token comparison below is satisfied by plain recomputation and
        # says nothing about the connector.
        warm_sched = [n for n, tok in scheduled if tok == prompt_len]
        assert warm_sched, (
            f"no context request with a {prompt_len}-token prompt reached the "
            f"connector on the warm run: {scheduled}")
        assert min(warm_sched) < prompt_len, (
            f"the warm run scheduled {warm_sched} tokens for a {prompt_len}-"
            f"token prompt after the connector offered {matched_tokens}. The "
            "offer was made but not honoured, so the whole prompt was "
            "recomputed and the token agreement below is vacuous.")
        assert min(warm_sched) == prompt_len - max(matched_tokens), (
            f"warm run scheduled {min(warm_sched)}; expected {prompt_len} - "
            f"{max(matched_tokens)} for a fully honoured offer.")

        assert len(warm_token_ids) == len(cold_token_ids), (
            f"Generation length changed: cold {len(cold_token_ids)} tokens, "
            f"warm {len(warm_token_ids)} tokens.")

        common_prefix = 0
        for cold_id, warm_id in zip(cold_token_ids, warm_token_ids):
            if cold_id != warm_id:
                break
            common_prefix += 1

        min_common_prefix = math.floor(
            len(cold_token_ids) * E2E_MIN_TOKEN_AGREEMENT)
        assert common_prefix >= min_common_prefix, (
            f"Gemma-3 connector cache reuse diverged at token {common_prefix} "
            f"of {len(cold_token_ids)}, below the {min_common_prefix}-token "
            "floor. Early divergence means the restored KV went into the wrong "
            "layer group's pages, not that it is numerically different.\n"
            f"  cold run: {cold_text!r}\n"
            f"  warm run: {warm_text!r}\n"
            f"  cold ids: {cold_token_ids}\n"
            f"  warm ids: {warm_token_ids}")
    finally:
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)


# The sliding window has to actually pass blocks for this test to mean
# anything, so the window is set well below the prompt length rather than above
# it as in `test_connector_vswa_e2e_gemma3`. 82 prompt + 64 generated with a
# 64-token window and 32-token blocks puts the boundary at
# max(0, (146 + 1 - 64) // 32) = 2 blocks out of window.
GEMMA3_ENGAGED_WINDOW = 64
GEMMA3_ENGAGED_GEN_TOKENS = 64


@pytest.mark.threadleak(enabled=False)
def test_connector_vswa_out_of_window_blocks_reach_the_connector(
        enforce_single_worker, monkeypatch):
    """The sliding window passing blocks, delivered per layer group.

    `test_connector_vswa_e2e_gemma3` keeps the whole run inside the window on
    purpose, so it covers addressing and the save/load round trip and never the
    out-of-window path. This is the other half: the window is set below the
    prompt length so blocks genuinely go stale, and the assertions are about
    what the connector is handed for them.

    **Why this cannot also be a round trip.** Staleness is a prefix property --
    the earliest blocks go first -- and a prefix cache must serve from ordinal
    0. Once the window has passed block 0 the sequence can never be served back
    as a prefix, whatever the full-attention group still holds. An engaged
    window and a warm hit are mutually exclusive for a sliding group, so this
    test asserts the masking and deliberately makes no round-trip claim.

    **The control.** Gemma-3's accuracy suite disables block reuse citing "gaps
    in kernel support for Gemma3's non-inclusive sliding window size", and that
    boundary is reached only when the window engages -- which is what this test
    does on purpose. So the same configuration is run without a connector and
    the outputs are compared. If they disagree, the model is not self-consistent
    at this window and nothing here can be attributed to the connector.
    """
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "examples", "llm-api"))
    sys.path.insert(0, examples_dir)
    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CONNECTOR_CACHE_FOLDER", cache_dir)

    try:
        import llm_kv_cache_connector_vswa as vswa

        leader_cls = vswa.VswaKvCacheConnectorLeader
        worker_cls = vswa.VswaKvCacheConnectorWorker

        finished = []
        original_finished = leader_cls.request_finished_by_layer_group

        def recording_finished(self, request, cache_block_ids_by_layer_group):
            finished.append([list(g) for g in cache_block_ids_by_layer_group])
            return original_finished(self, request,
                                     cache_block_ids_by_layer_group)

        layouts = []
        original_register = worker_cls.register_kv_cache_layout

        def recording_register(self, layout):
            layouts.append(layout)
            return original_register(self, layout)

        monkeypatch.setattr(leader_cls, "request_finished_by_layer_group",
                            recording_finished)
        monkeypatch.setattr(worker_cls, "register_kv_cache_layout",
                            recording_register)

        windows = [GEMMA3_ENGAGED_WINDOW] * (GEMMA3_CYCLE - 1) + [
            GEMMA3_MAX_SEQ_LEN
        ]
        model_path = f"{llm_models_root()}/gemma/gemma-3-1b-it"
        prompt = (
            "Nvidia Corporation is an American technology company "
            "headquartered in Santa Clara, California. Founded in 1993 by "
            "Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops "
            "graphics processing units (GPUs), system on a chips (SoCs), and "
            "application programming interfaces (APIs) for data science, "
            "high-performance computing, and mobile and automotive "
            "applications. Tell me about the company.")
        sampling_params = SamplingParams(max_tokens=GEMMA3_ENGAGED_GEN_TOKENS,
                                         ignore_eos=True)

        llm = vswa.build_llm(model=model_path,
                             max_attention_window=windows,
                             max_seq_len=GEMMA3_MAX_SEQ_LEN,
                             free_gpu_memory_fraction=0.3,
                             use_kv_cache_manager_v2="auto",
                             enable_block_reuse=True)
        try:
            out = llm.generate([prompt], sampling_params)[0]
            with_connector = list(out.outputs[0].token_ids)
            prompt_len = len(out.prompt_token_ids)
        finally:
            llm.shutdown()

        # --- The control, first: is the model self-consistent at this window? -
        # Run identically with no connector at all. If this disagrees, the
        # non-inclusive-window kernel gap is in play and nothing below can be
        # read as a statement about the connector.
        baseline_llm = LLM(model=model_path,
                           backend="pytorch",
                           cuda_graph_config=None,
                           disable_overlap_scheduler=True,
                           max_seq_len=GEMMA3_MAX_SEQ_LEN,
                           kv_cache_config=KvCacheConfig(
                               free_gpu_memory_fraction=0.3,
                               enable_block_reuse=True,
                               max_attention_window=windows,
                               use_kv_cache_manager_v2=True))
        try:
            baseline = list(
                baseline_llm.generate([prompt],
                                      sampling_params)[0].outputs[0].token_ids)
        finally:
            baseline_llm.shutdown()

        assert with_connector == baseline, (
            "the connector run and the connector-free run disagree at an "
            "engaged sliding window, so the model is not self-consistent here "
            "and the masking assertions below cannot be attributed to the "
            f"connector.\n  with connector: {with_connector}\n  baseline: "
            f"{baseline}")

        # --- The layout is genuinely two groups ------------------------------
        assert layouts, "register_kv_cache_layout never ran; this is not a V2 run"
        layout = layouts[0]
        by_window = {g.window_size: g for g in layout.groups}
        assert set(by_window) == {
            GEMMA3_ENGAGED_WINDOW, None
        }, (f"expected a sliding and a full-attention group, got "
            f"{[(g.layer_group_id, g.window_size) for g in layout.groups]}")
        sliding = by_window[GEMMA3_ENGAGED_WINDOW].layer_group_id
        full = by_window[None].layer_group_id

        # --- What request_finished offered, per group ------------------------
        assert finished, "request_finished_by_layer_group was never called"
        offered = finished[-1]
        assert len(offered) == 2

        stale = [
            i for i, slot in enumerate(offered[sliding])
            if slot == BAD_PAGE_INDEX
        ]
        live = [slot for slot in offered[sliding] if slot != BAD_PAGE_INDEX]

        # Anti-vacuity: if the window never engaged, nothing here is being
        # tested. This is the assertion that fails if the masking is removed.
        assert stale, (
            "no block went out of window, so the sliding group offered its "
            f"whole history and this test proves nothing: {offered[sliding]}")

        # The stale entries are a prefix, in place, not dropped from the list.
        assert stale == list(range(len(stale))), (
            f"out-of-window entries are not a leading run: {offered[sliding]}")
        assert len(offered[sliding]) == len(offered[full]), (
            "block ordinals are not aligned across groups, so an append-delta "
            f"over these lists would be invalid: {offered}")

        # The full-attention group keeps everything. Without this the test
        # would pass if masking were applied to every group indiscriminately.
        assert BAD_PAGE_INDEX not in offered[full], (
            f"the full-attention group lost blocks to a window it does not "
            f"have: {offered[full]}")
        assert len(live) < len(offered[full]), (
            "the sliding group offered as much as the full-attention group, so "
            "the window did not bound it")

        # The live range is bounded by the window, not merely smaller.
        max_live = math.ceil(GEMMA3_ENGAGED_WINDOW / 32) + 1
        assert len(live) <= max_live, (
            f"the sliding group offered {len(live)} live blocks for a "
            f"{GEMMA3_ENGAGED_WINDOW}-token window; at most {max_live} can hold "
            f"readable KV: {offered[sliding]}")
        assert len(set(live)) == len(live), (
            f"live page slots are not distinct: {live}")

        # And the prompt really did outrun the window, which is what put the
        # boundary inside the sequence rather than at either end.
        assert prompt_len > GEMMA3_ENGAGED_WINDOW, (
            f"prompt is {prompt_len} tokens against a "
            f"{GEMMA3_ENGAGED_WINDOW}-token window; the window cannot engage")
    finally:
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)


# Long enough that a 64-token window (2 blocks of 32) leaves most of the prompt
# behind once a served prefix moves history to the end of it.
SELECTIVE_PARA = (
    "Nvidia Corporation is an American technology company headquartered in "
    "Santa Clara, California. Founded in 1993 by Jensen Huang, Chris "
    "Malachowsky, and Curtis Priem, it develops graphics processing units "
    "(GPUs), system on a chips (SoCs), and application programming interfaces "
    "(APIs) for data science, high-performance computing, and mobile and "
    "automotive applications. ")
SELECTIVE_PROMPT = SELECTIVE_PARA * 3 + "Tell me about the company."

# A longer, self-contained prompt for models whose window is wide enough that
# the paragraph above would not outrun it. Kept in a file so the same text can
# be reused by hand when eyeballing output quality.
SELECTIVE_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "data",
                                     "kv_connector_vswa_prompt.txt")
SELECTIVE_WINDOW = 64


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "model_rel,assert_round_trip",
    [
        # Real interleaved attention. Output correctness is NOT asserted: at an
        # engaged window Gemma-3 does not survive prefix reuse at all, with or
        # without a connector. Measured, same window and prompt, local reuse and
        # no connector: 0/16 token agreement, against 16/16 with the window
        # never engaging. That is the gap the accuracy suite's WAR names ("gaps
        # in kernel support for Gemma3's non-inclusive sliding window size"), so
        # a round-trip assertion here would fail for a reason that has nothing
        # to do with the connector.
        pytest.param("gemma/gemma-3-1b-it", False, id="gemma3_interleaved"),
        # Uniform attention with the windows imposed, which is where the
        # restoration can actually be checked: same window and prompt under
        # local reuse gives 16/16. Weaker as a VSWA claim, stronger as a
        # correctness claim, so the pair covers what neither does alone.
        pytest.param("Qwen3/Qwen3-0.6B", True, id="qwen3_forced_windows"),
        # The case this test wants most: a natively interleaved model whose
        # prefix-reuse path is sound, so the round trip IS assertable on real
        # VSWA. GPT-OSS applies a 128-token window to every even layer
        # (modeling_gpt_oss.py) and asks for KVCacheManagerV2 itself.
        #
        # NEVER EXECUTED. It requires SM100: the model is MXFP4 (no Ampere MoE
        # kernel) and forces attn_backend="TRTLLM" because of its attention
        # sinks. Gated below rather than deleted so the case is queued rather
        # than forgotten -- but nothing here has run, and it should not be
        # counted as coverage until it has.
        pytest.param("gpt_oss/gpt-oss-20b",
                     True,
                     id="gpt_oss_interleaved",
                     marks=pytest.mark.skipif(
                         get_sm_version() not in (100, 103),
                         reason="gpt-oss-20b is MXFP4 and needs SM100/SM103")),
    ])
def test_connector_transfers_only_in_window_blocks_to_the_sliding_group(
        enforce_single_worker, monkeypatch, model_rel, assert_round_trip):
    """The connector transfers each layer group only what that group can read.

    Target: under a sliding window that has genuinely passed part of the
    prompt, the connector loads **only the in-window blocks** into the sliding
    layer group and **every block** into the full-attention group.

    The window engages because the prefix is served, not despite it: honouring
    the offer moves `history_length` to the end of the served range, which is
    what puts the earlier blocks out of the sliding group's window. They are
    then reported as `-1`, and the connector must skip them -- there is no
    readable KV there and attention will never look.
    """
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "examples", "llm-api"))
    sys.path.insert(0, examples_dir)
    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CONNECTOR_CACHE_FOLDER", cache_dir)

    try:
        import llm_kv_cache_connector_vswa as vswa

        L, Wk = vswa.VswaKvCacheConnectorLeader, vswa.VswaKvCacheConnectorWorker
        layouts, loads, offers, scheduled, by_group = [], [], [], [], []
        original_build = L.build_connector_meta
        original_query = L.get_num_new_matched_tokens
        original_register = Wk.register_kv_cache_layout

        def recording_build(self, scheduler_output):
            metadata = original_build(self, scheduler_output)
            loads.extend(metadata.load)
            for rd in scheduler_output.new_requests:
                scheduled.append((rd.num_scheduled_tokens, len(rd.new_tokens)))
                by_group.append(
                    [list(g) for g in rd.new_block_ids_by_layer_group])
            return metadata

        def recording_query(self, request, num_computed_tokens):
            result = original_query(self, request, num_computed_tokens)
            offers.append(result[0])
            return result

        def recording_register(self, layout):
            layouts.append(layout)
            return original_register(self, layout)

        monkeypatch.setattr(L, "build_connector_meta", recording_build)
        monkeypatch.setattr(L, "get_num_new_matched_tokens", recording_query)
        monkeypatch.setattr(Wk, "register_kv_cache_layout", recording_register)

        sampling_params = SamplingParams(max_tokens=16, ignore_eos=True)

        def gen():
            llm = vswa.build_llm(model=f"{llm_models_root()}/{model_rel}",
                                 max_attention_window=[SELECTIVE_WINDOW] *
                                 (GEMMA3_CYCLE - 1) + [GEMMA3_MAX_SEQ_LEN],
                                 max_seq_len=GEMMA3_MAX_SEQ_LEN,
                                 free_gpu_memory_fraction=0.3,
                                 use_kv_cache_manager_v2=True,
                                 enable_block_reuse=True)
            try:
                out = llm.generate([SELECTIVE_PROMPT], sampling_params)[0]
                return (list(out.outputs[0].token_ids),
                        len(out.prompt_token_ids))
            finally:
                llm.shutdown()

        cold, prompt_len = gen()
        assert prompt_len > 3 * SELECTIVE_WINDOW, (
            f"prompt is {prompt_len} tokens against a {SELECTIVE_WINDOW}-token "
            "window; too short for the window to leave most of it behind")
        for recorder in (loads, offers, scheduled, by_group):
            recorder.clear()

        warm, _ = gen()

        layout = layouts[0]
        by_window = {g.window_size: g.layer_group_id for g in layout.groups}
        assert set(by_window) == {
            SELECTIVE_WINDOW, None
        }, (f"expected a sliding and a full group, got {by_window}")
        sliding, full = by_window[SELECTIVE_WINDOW], by_window[None]

        # The prefix was honoured -- otherwise the window never moves and the
        # rest of this measures nothing.
        warm_sched = [n for n, tok in scheduled if tok == prompt_len]
        assert warm_sched and min(warm_sched) < prompt_len, (
            f"the warm run scheduled {warm_sched} of {prompt_len} tokens after "
            f"the connector offered {offers}; the offer was not honoured, so "
            "history never advanced and no block went out of window")

        # The masking: a leading run of -1 in the sliding group, none in the
        # full group. The boundary is recomputed from the offer rather than read
        # back from the implementation.
        slots = by_group[0]
        stale = [i for i, s in enumerate(slots[sliding]) if s == BAD_PAGE_INDEX]
        expected_stale = max(0, (max(offers) + 1 - SELECTIVE_WINDOW) //
                             layout.tokens_per_block)
        assert stale == list(range(len(stale))), (
            f"out-of-window entries are not a leading run: {slots[sliding]}")
        assert len(stale) == expected_stale, (
            f"{len(stale)} blocks out of window; expected {expected_stale} for "
            f"a {SELECTIVE_WINDOW}-token window at history {max(offers)}")
        # Not just "engaged" -- the window must leave a substantial part of
        # the prompt behind, or the two groups barely differ and the selective
        # transfer below is not really being exercised.
        assert expected_stale >= 2, (
            f"only {expected_stale} block(s) went out of window at history "
            f"{max(offers)}; size the prompt so the window leaves more behind")
        assert BAD_PAGE_INDEX not in slots[full], (
            f"the full-attention group lost blocks it should keep: {slots[full]}"
        )

        # The point of the test: what was actually transferred, per group.
        sliding_loads = [x for x in loads if x[1] == sliding]
        full_loads = [x for x in loads if x[1] == full]
        assert sliding_loads, (
            "nothing was loaded into the sliding group; it still holds an "
            "in-window range and skipping it entirely is not correct either")
        assert len(sliding_loads) < len(full_loads), (
            f"the sliding group was loaded {len(sliding_loads)} blocks and the "
            f"full group {len(full_loads)}; the window did not bound the "
            "transfer")
        max_live = math.ceil(SELECTIVE_WINDOW / layout.tokens_per_block) + 1
        assert len(sliding_loads) <= max_live, (
            f"{len(sliding_loads)} blocks loaded into a "
            f"{SELECTIVE_WINDOW}-token window; at most {max_live} can be read")
        assert len({x[2] for x in sliding_loads}) == len(sliding_loads)

        if assert_round_trip:
            assert warm == cold, (
                "the selectively restored prefix did not reproduce the cold "
                f"run.\n  cold: {cold}\n  warm: {warm}")
    finally:
        if examples_dir in sys.path:
            sys.path.remove(examples_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
