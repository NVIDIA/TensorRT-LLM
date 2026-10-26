# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.features import HelixMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention


@pytest.mark.parametrize("helix_spec_tokens_valid", [False, True])
def test_mla_rope_generation_forwards_kernel_controls(helix_spec_tokens_valid: bool) -> None:
    helix = SimpleNamespace(
        position_offsets=object(),
        is_inactive_rank=object(),
        spec_tokens_valid=helix_spec_tokens_valid,
        local_slots=object(),
    )
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        max_seq_len=1024,
        find_ext=lambda cls: helix if cls is HelixMetadata else None,
    )
    dispatch = Mock()
    attention = SimpleNamespace(
        is_mla_enable=True,
        mla_params=object(),
        uses_fp4_mla_attention=False,
        _ensure_rope_table_size=Mock(),
        presented_token_major_for=Mock(return_value=nullcontext()),
        _mla_rope_generation_impl=dispatch,
    )
    tensors = [object() for _ in range(9)]
    weight, scale = object(), object()
    controls = dict(
        kv_norm_weight=weight,
        kv_norm_eps=1e-5,
        precomputed_cu_seqlens=True,
        precomputed_fmha_scheduler=True,
        kv_only=True,
        kv_done_elsewhere=False,
        quant_scale_qkv=scale,
        q_rope_applied=True,
    )
    TrtllmAttention.mla_rope_generation(
        attention,
        *tensors[:3],
        metadata,
        *tensors[3:],
        **controls,
    )
    attention._ensure_rope_table_size.assert_called_once_with(1024)
    attention.presented_token_major_for.assert_called_once_with(metadata)
    dispatch.assert_called_once()
    assert dispatch.call_args.args[0] is metadata
    assert dispatch.call_args.args[1:10] == tuple(tensors)
    helix_tensor_params = [helix.position_offsets, helix.is_inactive_rank]
    if helix_spec_tokens_valid:
        helix_tensor_params.append(helix.local_slots)
    assert dispatch.call_args.args[11] == helix_tensor_params
    assert dispatch.call_args.args[-len(controls) :] == tuple(controls.values())
