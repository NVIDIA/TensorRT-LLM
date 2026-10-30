# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config
from _torch.pyexecutor.engine_test_utils import DummyModel, kv_cache_manager

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    DraftTargetDecodingConfig,
    KvCacheConfig,
    PARDDecodingConfig,
)


@pytest.mark.parametrize("config_cls", [DraftTargetDecodingConfig, PARDDecodingConfig])
@pytest.mark.parametrize(
    "draft_len", [None, 0, 3, 1], ids=["general", "cuda_graph_0", "cuda_graph_3", "cuda_graph_1"]
)
def test_warmup_builders_resynchronize_stale_draft_buffers(
    config_cls: type[DecodingBaseConfig], draft_len: int | None
) -> None:
    spec = config_cls(max_draft_len=3, speculative_model="dummy", draft_len_schedule={1: 3, 4: 2})
    config = decoder_config(spec_config=spec)
    model = DummyModel(torch.half)
    requests = DummyRequestFactory(
        config=config,
        model=model,
        get_runtime_tokens_per_gen_step=spec.get_runtime_tokens_per_gen_step,
        release_request_state=None,
    )
    with kv_cache_manager(
        model=model,
        config=KvCacheConfig(max_tokens=258, use_kv_cache_manager_v2=True),
        max_num_tokens=258,
    ) as manager:
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
        batch_size = 2
        if draft_len is None:
            expected_draft_len = config.max_draft_len
            warmup_request = requests.create_warmup_request(
                resource_manager,
                num_tokens=batch_size * spec.tokens_per_gen_step,
                num_gen_requests=batch_size,
            )
        else:
            expected_draft_len = draft_len
            warmup_request = requests.create_cuda_graph_warmup_request(
                resource_manager, batch_size, draft_len, max_seq_len=16
            )
        with requests.release_batch_context(warmup_request, resource_manager) as batch:
            assert batch is not None
            assert len(batch.generation_requests) == batch_size
            expected_buffer_width = spec.get_runtime_tokens_per_gen_step(expected_draft_len) - 1
            assert all(
                len(request.py_draft_tokens) == expected_buffer_width
                for request in batch.generation_requests
            )
