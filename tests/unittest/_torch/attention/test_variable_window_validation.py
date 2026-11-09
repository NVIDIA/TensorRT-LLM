# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._utils import get_sm_version


def _run_variable_window_attention(
    head_dim: int,
    bounds_device: torch.device,
    *,
    include_starts: bool = True,
    include_ends: bool = True,
    attention_input_type: AttentionInputType = AttentionInputType.context_only,
) -> None:
    num_heads = 2
    num_kv_heads = 1
    num_tokens = 4
    with torch.cuda.device(0):
        attention = create_attention(
            "TRTLLM",
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        metadata = attention.Metadata(
            max_num_requests=1,
            max_num_tokens=num_tokens,
            kv_cache_manager=None,
            mapping=None,
            runtime_features=None,
        )
        metadata.seq_lens = torch.tensor([num_tokens], dtype=torch.int32)
        metadata.num_contexts = 1
        metadata.request_ids = torch.tensor([0], dtype=torch.int32)
        metadata.max_seq_len = num_tokens
        metadata.prepare()

        qkv = torch.zeros(
            num_tokens,
            (num_heads + 2 * num_kv_heads) * head_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        starts = torch.zeros(num_tokens, dtype=torch.int32, device=bounds_device)
        ends = torch.arange(num_tokens, dtype=torch.int32, device=bounds_device)
        forward_args = AttentionForwardArgs(
            attention_input_type=attention_input_type,
            variable_window_token_starts=starts if include_starts else None,
            variable_window_token_ends=ends if include_ends else None,
        )
        attention.forward(qkv, None, None, metadata, forward_args=forward_args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires SM100 or SM103")
def test_variable_window_bounds_must_match_query_device():
    with pytest.raises(RuntimeError, match="must be on the same device as qkv_or_q"):
        _run_variable_window_attention(head_dim=128, bounds_device=torch.device("cuda:1"))


@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires SM100 or SM103")
def test_variable_window_rejects_unsupported_kernel_configuration():
    with pytest.raises(
        RuntimeError, match="requires a supported context FMHA kernel configuration"
    ):
        _run_variable_window_attention(head_dim=64, bounds_device=torch.device("cuda:0"))


@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires SM100 or SM103")
def test_variable_window_probe_rejects_invalid_head_size():
    assert not torch.ops.trtllm.attention_supports_variable_window(
        torch.bfloat16,
        torch.bfloat16,
        2,
        1,
        7,
        None,
        0,
        1,
        4,
        4,
        None,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("include_starts", "include_ends"),
    [(True, False), (False, True)],
    ids=["starts-only", "ends-only"],
)
def test_variable_window_bounds_must_be_paired(include_starts, include_ends):
    with pytest.raises(
        ValueError,
        match="variable_window_token_starts and variable_window_token_ends must be provided together",
    ):
        _run_variable_window_attention(
            head_dim=128,
            bounds_device=torch.device("cuda:0"),
            include_starts=include_starts,
            include_ends=include_ends,
        )


def test_variable_window_bounds_reject_generation_only_attention():
    with pytest.raises(
        ValueError,
        match="variable-window bounds are supported only for context attention",
    ):
        _run_variable_window_attention(
            head_dim=128,
            bounds_device=torch.device("cuda:0"),
            attention_input_type=AttentionInputType.generation_only,
        )


@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires SM100 or SM103")
def test_variable_window_attention_matches_dense_reference():
    num_tokens = 7
    num_heads = 2
    head_dim = 128
    attention = create_attention(
        "TRTLLM", layer_idx=0, num_heads=num_heads, num_kv_heads=1, head_dim=head_dim
    )
    metadata = attention.Metadata(
        max_num_requests=1,
        max_num_tokens=num_tokens,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None,
    )
    metadata.seq_lens = torch.tensor([num_tokens], dtype=torch.int32)
    metadata.num_contexts = 1
    metadata.request_ids = torch.tensor([0], dtype=torch.int32)
    metadata.max_seq_len = num_tokens
    metadata.prepare()

    torch.manual_seed(7)
    qkv = (
        torch.randn(
            num_tokens,
            (num_heads + 2) * head_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        * 0.2
    )
    starts = torch.tensor([0, 0, 0, 0, 2, 4, 4], dtype=torch.int32, device="cuda:0")
    ends = torch.tensor([0, 3, 3, 3, 4, 6, 6], dtype=torch.int32, device="cuda:0")
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    actual = attention.forward(qkv, None, None, metadata, forward_args=forward_args)

    q, k, v = qkv.split((num_heads * head_dim, head_dim, head_dim), dim=-1)
    q = q.reshape(num_tokens, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k.reshape(num_tokens, 1, head_dim).transpose(0, 1).expand(1, num_heads, -1, -1)
    v = v.reshape(num_tokens, 1, head_dim).transpose(0, 1).expand(1, num_heads, -1, -1)
    kv_positions = torch.arange(num_tokens, device="cuda:0")
    dense_mask = (kv_positions >= starts[:, None]) & (kv_positions <= ends[:, None])
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask[None, None])
    expected = expected.squeeze(0).transpose(0, 1).reshape(num_tokens, num_heads * head_dim)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)
