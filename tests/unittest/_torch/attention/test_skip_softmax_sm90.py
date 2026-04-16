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

from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.attention_backend.trtllm as trtllm_attention_backend
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig


def _make_no_cache_metadata(attention: TrtllmAttention, seq_len: int):
    metadata = attention.Metadata(
        max_num_requests=1,
        max_num_tokens=seq_len,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None,
    )
    metadata.seq_lens = torch.tensor([seq_len], dtype=torch.int)
    metadata.num_contexts = 1
    metadata.request_ids = torch.tensor([0], dtype=torch.int)
    metadata.max_seq_len = seq_len
    metadata.prepare()
    return metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("attention_mask", "expect_prefill_threshold"),
    (
        (PredefinedAttentionMask.FULL, None),
        (PredefinedAttentionMask.CAUSAL, 5000.0),
    ),
)
def test_skip_softmax_prefill_threshold_is_guarded_on_sm90(
        monkeypatch: pytest.MonkeyPatch, attention_mask,
        expect_prefill_threshold):
    attention = TrtllmAttention(
        layer_idx=0,
        num_heads=24,
        head_dim=128,
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            threshold_scale_factor={
                "prefill": 5000.0,
                "decode": 7.0,
            }),
    )
    metadata = _make_no_cache_metadata(attention, seq_len=16)
    qkv = torch.randn(16,
                      24 * 128 * 3,
                      dtype=torch.bfloat16,
                      device="cuda")

    captured_plan_kwargs = {}
    warning_once = Mock()

    monkeypatch.setattr(trtllm_attention_backend, "get_sm_version", lambda: 90)
    monkeypatch.setattr(trtllm_attention_backend.logger, "warning_once",
                        warning_once)
    monkeypatch.setattr(attention.wrapper, "plan",
                        lambda **kwargs: captured_plan_kwargs.update(kwargs))
    monkeypatch.setattr(attention.wrapper, "run", lambda *args, **kwargs: None)

    attention.forward(qkv,
                      None,
                      None,
                      metadata,
                      attention_mask=attention_mask)

    assert captured_plan_kwargs[
        "skip_softmax_threshold_scale_factor_prefill"] == expect_prefill_threshold
    assert captured_plan_kwargs[
        "skip_softmax_threshold_scale_factor_decode"] == 7.0

    if attention_mask == PredefinedAttentionMask.FULL:
        warning_once.assert_called_once()
    else:
        warning_once.assert_not_called()
