# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-VL admission reads of CUDA mRoPE deltas (GPU)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import cap_mrope_decode_tokens
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.sampling_params import SamplingParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_child_takes_the_parent_delta_while_its_copy_is_queued() -> None:
    deltas = torch.tensor([[7]], dtype=torch.int32, device="cuda")
    parent = LlmRequest(
        request_id=1,
        max_new_tokens=100,
        input_tokens=[0] * 10,
        sampling_config=SamplingConfig(SamplingParams(n=2, top_k=4)._get_sampling_config()),
        is_streaming=False,
        py_multimodal_data={"mrope_config": {"mrope_position_deltas": deltas}},
    )
    # Leave a sentinel in the block the child's deep copy will reuse.
    sentinel = torch.full_like(deltas, 10**9)
    del sentinel
    torch.cuda.synchronize()
    # The child's copy queues on the current stream behind a step in flight.
    torch.cuda._sleep(1 << 28)
    parent.create_child_request(2)
    child = parent.child_requests[0]
    text_config = SimpleNamespace(max_position_embeddings=40)
    cap_mrope_decode_tokens(parent, text_config)
    cap_mrope_decode_tokens(child, text_config)
    assert parent.py_max_new_tokens == 24
    assert child.py_max_new_tokens == parent.py_max_new_tokens
    torch.cuda.synchronize()
