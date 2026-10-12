# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention


@pytest.mark.parametrize("helix_spec_tokens_valid", [False, True])
def test_mla_rope_generation_forwards_kernel_controls(helix_spec_tokens_valid: bool) -> None:
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        max_seq_len=1024,
        helix_position_offsets=object(),
        helix_is_inactive_rank=object(),
        _helix_spec_tokens_valid=helix_spec_tokens_valid,
        helix_local_slots=object(),
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
    helix_tensor_params = [metadata.helix_position_offsets, metadata.helix_is_inactive_rank]
    if helix_spec_tokens_valid:
        helix_tensor_params.append(metadata.helix_local_slots)
    assert dispatch.call_args.args[11] == helix_tensor_params
    assert dispatch.call_args.args[-7:] == tuple(controls.values())
