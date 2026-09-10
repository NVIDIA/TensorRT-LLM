# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
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
    force_fallback: bool = False,
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
            attention_input_type=AttentionInputType.context_only,
            variable_window_token_starts=starts if include_starts else None,
            variable_window_token_ends=ends if include_ends else None,
        )
        if force_fallback:
            fallback = FallbackFmha(attention)
            with mock.patch.object(attention._fmha_manager, "select", return_value=fallback):
                attention.forward(qkv, None, None, metadata, forward_args=forward_args)
        else:
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


@pytest.mark.parametrize(
    ("include_starts", "include_ends"),
    [(True, False), (False, True)],
    ids=["starts-only", "ends-only"],
)
def test_variable_window_bounds_must_be_paired(include_starts, include_ends):
    with pytest.raises(
        RuntimeError,
        match="variable_window_token_starts and variable_window_token_ends must be provided together",
    ):
        _run_variable_window_attention(
            head_dim=128,
            bounds_device=torch.device("cuda:0"),
            include_starts=include_starts,
            include_ends=include_ends,
            force_fallback=True,
        )
