# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request admission respects the sequence budget of each cache layout."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.utils import RequestError, context_length_exceeded_message
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def worker(monkeypatch):
    instance = BaseWorker()
    monkeypatch.setattr(instance, "start", lambda: None)
    instance.rank = 0
    instance._lora_manager = None
    instance.engine = Mock()
    instance.engine.enqueue_request.return_value = 1
    yield instance
    instance.doing_shutdown = True


@pytest.mark.parametrize(
    "cp_size,cp_type,use_v2,max_seq_len,prompt_len,max_tokens,expected",
    [
        (8, "HELIX", True, 1048576, 985700, 65536, 62876),
        (8, "HELIX", True, 1048576, 985700, None, 62876),
        (8, "HELIX", True, 1048576, 985700, 100, 100),
        (8, "HELIX", True, 1024, 1017, 32, 7),
        (1, "HELIX", True, 1024, 1000, 64, 24),
        (8, "HELIX", False, 128, 1000, 64, 3),
        (8, "ULYSSES", False, 128, 1000, 64, 3),
    ],
)
def test_submit_respects_sequence_budget(
    worker, cp_size, cp_type, use_v2, max_seq_len, prompt_len, max_tokens, expected
):
    worker.mapping = Mapping(world_size=cp_size, cp_size=cp_size, cp_config={"cp_type": cp_type})
    worker.engine.kv_cache_manager = Mock(spec=KVCacheManagerV2) if use_v2 else SimpleNamespace()
    worker.max_seq_len = max_seq_len
    request = GenerationRequest(
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )

    worker.submit(request)

    admitted = worker.engine.enqueue_request.call_args.args[0]
    assert admitted.max_tokens == expected
    assert admitted.num_input_tokens == prompt_len


@pytest.mark.parametrize("prompt_len", [1048576, 1048577])
def test_submit_rejects_helix_prompt_without_generation_room(worker, prompt_len):
    worker.mapping = Mapping(world_size=8, cp_size=8, cp_config={"cp_type": "HELIX"})
    worker.engine.kv_cache_manager = Mock(spec=KVCacheManagerV2)
    worker.max_seq_len = 1048576
    request = GenerationRequest(
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(max_tokens=1),
    )

    with pytest.raises(RequestError) as exc:
        worker.submit(request)

    assert str(exc.value) == context_length_exceeded_message(1048576, prompt_len)
    worker.engine.enqueue_request.assert_not_called()
