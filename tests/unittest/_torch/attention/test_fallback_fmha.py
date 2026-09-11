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

"""Dispatch contract of ``FallbackFmha.is_supported``.

The native op silently drops requests it cannot serve, so the fallback has to
refuse them itself rather than lower them and read back an untouched output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.fmha.fallback import (
    FallbackFmha,
    _set_context_workspace_shape,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaParams, StaticAttentionConfig
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention


@pytest.mark.parametrize(
    ("is_cross", "update_kv_cache", "expected"),
    (
        (False, False, False),
        (False, True, True),
        (True, False, True),
    ),
)
def test_fallback_support_matches_thop_kv_update_contract(is_cross, update_kv_cache, expected):
    """Do not dispatch requests that the native attention op rejects."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=is_cross)
    forward_args = AttentionForwardArgs(update_kv_cache=update_kv_cache)

    assert fmha.is_supported(None, None, None, metadata, forward_args) is expected


def test_fallback_rejects_raw_fp8_input():
    """Do not dispatch raw FP8 QKV to the native attention op."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=False)
    forward_args = AttentionForwardArgs(update_kv_cache=True)
    q = torch.empty((1, 128), dtype=torch.float8_e4m3fn)

    assert not fmha.is_supported(q, None, None, metadata, forward_args)


def test_context_workspace_shape_uses_active_batch_extents():
    params = FmhaParams(max_num_requests=2048, max_context_length=8192)

    _set_context_workspace_shape(
        params,
        num_contexts=1,
        num_ctx_tokens=1,
        host_context_lengths=torch.tensor([1], dtype=torch.int32),
        host_past_key_value_lengths=torch.tensor([0], dtype=torch.int32),
        host_total_kv_lens=torch.tensor([1], dtype=torch.int32),
        max_context_q_len_override=None,
    )

    assert params.num_seqs == 1
    assert params.num_requests == 1
    assert params.num_tokens == 1
    assert params.input_seq_length == 1
    assert params.max_past_kv_length == 0
    assert params.total_kv_len == 1


def test_context_workspace_shape_clears_context_for_generation_only():
    params = FmhaParams(
        num_seqs=2048,
        num_requests=2048,
        num_tokens=8192,
        input_seq_length=8192,
        max_past_kv_length=8192,
    )

    _set_context_workspace_shape(
        params,
        num_contexts=0,
        num_ctx_tokens=0,
        host_context_lengths=torch.empty(0, dtype=torch.int32),
        host_past_key_value_lengths=torch.empty(0, dtype=torch.int32),
        host_total_kv_lens=torch.empty(0, dtype=torch.int32),
        max_context_q_len_override=None,
    )

    assert params.num_seqs == 0
    assert params.num_requests == 0
    assert params.num_tokens == 0
    assert params.input_seq_length == 0
    assert params.max_past_kv_length == 0
    assert params.total_kv_len == 0


def test_prepare_workspace_keeps_scheduler_and_multi_ctas_counters_separate(monkeypatch):
    scheduler_counter = torch.empty(1, dtype=torch.uint32)
    multi_ctas_kv_counter = torch.empty(64, dtype=torch.uint8)
    allocations = []

    def allocate_counter(counter, device, num_heads, max_num_sequences):
        allocations.append((counter, device, num_heads, max_num_sequences))
        return multi_ctas_kv_counter

    class AttentionOp:
        def get_attention_workspace_size(self, *args):
            return 0

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.fallback.get_multi_ctas_kv_counter",
        allocate_counter,
    )
    monkeypatch.setattr(FallbackFmha, "_to_thop_params", lambda self, params: object())

    fmha = object.__new__(FallbackFmha)
    attn = SimpleNamespace(num_heads=8, attention_op=lambda params: AttentionOp())
    fmha._attn_ref = lambda: attn
    fmha._multi_ctas_kv_counter = None
    params = FmhaParams(
        fwd=AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            fmha_scheduler_counter=scheduler_counter,
        ),
        qkv_or_q=torch.empty((1, 8)),
        workspace=torch.empty(0, dtype=torch.uint8),
        host_context_lengths=torch.empty(0, dtype=torch.int32),
        host_past_key_value_lengths=torch.empty(0, dtype=torch.int32),
        cyclic_attention_window_size=128,
        max_attention_window_size=128,
    )
    metadata = SimpleNamespace(
        num_generations=1,
        num_contexts=0,
        num_ctx_tokens=0,
        max_num_sequences=4,
        max_num_requests=2,
        host_total_kv_lens=torch.tensor([0, 1], dtype=torch.int32),
        effective_beam_width=1,
    )

    fmha.prepare_workspace(params, metadata)

    assert allocations == [(None, params.qkv_or_q.device, 8, 4)]
    assert params.fwd.fmha_scheduler_counter is scheduler_counter
    assert params.multi_ctas_kv_counter is multi_ctas_kv_counter


def test_attention_op_cache_separates_static_configurations(monkeypatch):
    class FakeStaticConfig(str):
        def to_thop_config(self):
            return str(self)

    created = []

    class AttentionOp:
        def __init__(self, config):
            self.config = config
            created.append(config)

    monkeypatch.setattr(
        StaticAttentionConfig,
        "from_params",
        classmethod(lambda cls, params, **kwargs: params),
    )
    monkeypatch.setattr(trtllm_backend, "thop", SimpleNamespace(AttentionOp=AttentionOp))
    owner = SimpleNamespace(_attention_ops={}, skip_correction_threshold=0.0)
    causal = FakeStaticConfig("causal")
    full = FakeStaticConfig("full")

    causal_op = TrtllmAttention.attention_op(owner, causal)

    assert TrtllmAttention.attention_op(owner, causal) is causal_op
    assert TrtllmAttention.attention_op(owner, full) is not causal_op
    assert created == ["causal", "full"]

    TrtllmAttention.release(owner)
    assert owner._attention_ops == {}
