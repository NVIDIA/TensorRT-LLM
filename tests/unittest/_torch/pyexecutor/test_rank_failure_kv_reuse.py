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
"""Tests that a context chunk's KV is reusable only after rank failure recovery accepts its step.

The tests use the real KV cache manager V2 with block reuse and chunked prefill, in the overlap
executor's order.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

PROMPT_LEN = 96
CHUNK = 32  # one KV block


@pytest.fixture
def manager():
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=1024,
            enable_block_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=CHUNK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )
    yield manager
    manager.shutdown()


def _executor(manager, accept_step):
    """Return the executor's context update and commit paths over ``manager``.

    ``accept_step`` is rank failure recovery's verdict on each step, or None without fault
    tolerance.
    """
    executor = SimpleNamespace(
        rank_failure_recovery=None
        if accept_step is None
        else SimpleNamespace(accept_step=lambda batch: accept_step),
        kv_cache_manager=manager,
        enable_joint_kv_cache_reuse=False,
        resource_manager=SimpleNamespace(get_resource_manager=lambda _type: None),
        _wait_for_model_engine_input_copy=lambda: None,
        previous_batch=None,
        drafter=None,
        use_spec_decode=False,
        capturable_guided_decoder=None,
    )
    executor._discard_previous_batch = partial(PyExecutor._discard_previous_batch, executor)

    def finish(step):
        """Accept or discard ``step`` as the executor does before it handles the step's results."""
        executor.previous_batch = step
        PyExecutor._accept_or_discard_previous_batch(executor)

    executor.finish = finish
    return executor


def _request(request_id, first_token):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(first_token, first_token + PROMPT_LEN)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _schedule_chunk(manager, request, chunk_size=CHUNK):
    """Allocate the request's next context chunk as the V2 scheduler does before the forward."""
    batch = ScheduledRequests()
    batch.append_context_request(request)
    if request.context_current_position == 0:
        assert manager.prepare_context(request)
    request.context_chunk_size = min(chunk_size, request.context_remaining_length)
    assert manager.resize_context(request, request.context_chunk_size)
    manager.prepare_resources(batch)
    return batch


def _launch(executor, batch):
    """Launch the forward, update the caches as the executor does, and return the step."""
    for request in batch.context_requests:
        request.move_to_next_context_chunk()
    if executor.rank_failure_recovery is None:
        PyExecutor._update_v2_context_resources(executor, batch)
        return BatchState(scheduled_requests=batch, sample_state=None)
    context_commits = []
    PyExecutor._update_v2_context_resources(executor, batch, context_commits)
    return BatchState(scheduled_requests=batch, sample_state=None, context_commits=context_commits)


def _reused_prefix(manager, request_id, first_token):
    """Return how much of the prompt starting at ``first_token`` a new request can reuse."""
    probe = _request(request_id, first_token)
    _schedule_chunk(manager, probe, PROMPT_LEN)
    manager.free_resources(probe)
    return probe.context_current_position


@pytest.mark.parametrize(
    "accept_step", [None, True, False], ids=["off", "on-accepted", "on-discarded"]
)
def test_chunk_reusable_after_step_accepted(manager, accept_step):
    executor = _executor(manager, accept_step)
    request = _request(1, first_token=1000)
    step = _launch(executor, _schedule_chunk(manager, request))
    executor.finish(step)
    # A discarded step stands for a rank failure, which frees the request for recompute.
    assert (executor.previous_batch is step) == (accept_step is not False)
    manager.free_resources(request)
    # Without fault tolerance the chunk is committed at its launch.
    assert _reused_prefix(manager, 2, first_token=1000) == (0 if accept_step is False else CHUNK)


def test_discarded_middle_chunk_recomputed(manager):
    """Keep a prompt's accepted first chunk, drop its discarded second one, and recompute it.

    A rank failure discards the step of the second chunk and frees the request. Only the first
    chunk is then reusable, and the recompute runs the rest of the prompt, which becomes reusable.
    """
    executor = _executor(manager, True)
    request = _request(1, first_token=1000)
    executor.finish(_launch(executor, _schedule_chunk(manager, request)))
    executor.rank_failure_recovery.accept_step = lambda batch: False
    step = _launch(executor, _schedule_chunk(manager, request))
    executor.finish(step)
    assert executor.previous_batch is None
    manager.free_resources(request)
    assert _reused_prefix(manager, 2, first_token=1000) == CHUNK

    executor.rank_failure_recovery.accept_step = lambda batch: True
    recomputed = _request(3, first_token=1000)
    while recomputed.context_remaining_length > 0:
        executor.finish(_launch(executor, _schedule_chunk(manager, recomputed)))
    manager.free_resources(recomputed)
    assert _reused_prefix(manager, 4, first_token=1000) >= 2 * CHUNK
