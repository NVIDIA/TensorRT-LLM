# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._utils import get_sm_version


def _run_variable_window_attention(head_dim: int, bounds_device: torch.device) -> None:
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
        attention.forward(
            qkv,
            None,
            None,
            metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                variable_window_token_starts=starts,
                variable_window_token_ends=ends,
            ),
        )


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
