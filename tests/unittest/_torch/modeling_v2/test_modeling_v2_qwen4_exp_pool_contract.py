# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's KV cache pool contract rejects the mismatches the kernels would not.

The checks read only `.shape` / `.dtype`, so these run on meta tensors with no GPU and
no engine. Each case is one wrong pool that the kernel consuming it would accept and
mis-address.
"""

from __future__ import annotations

import pytest
import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

G = M._PoolGeometry(
    tokens_per_block=32,
    heads_kv=2,
    head_dim=256,
    index_kv_heads=1,
    index_head_dim=128,
    conv_dim=8192,
    conv_kernel=4,
    num_v_heads=48,
    head_v_dim=128,
    head_k_dim=128,
    stream=10240,
    ple_state_len=9,
    ngram_context_len=3,
    dtype=torch.bfloat16,
)
SLOTS = 6


def _t(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _conv(**kw):
    return _t(SLOTS, G.conv_dim, G.conv_kernel - 1, **kw)


def _ssm(**kw):
    return _t(SLOTS, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32)


def test_valid_pools_pass() -> None:
    M.check_position_plane(G, _t(9, 32, 3, dtype=torch.int32))
    M.check_attention_pools(
        G, 3, _t(9, 32, 1, 128), _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    )
    M.check_recurrent_pools(G, _conv(), _ssm())
    M.check_ple_pools(
        G,
        _t(SLOTS, G.stream, G.ple_state_len),
        _t(SLOTS, G.ngram_context_len, dtype=torch.int64),
        _conv(),
    )
    M.check_step_buffers(
        G,
        _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1),
        _t(SLOTS, 2, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32),
        _conv(),
        _ssm(),
        _t(SLOTS, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "plane, match",
    [
        (None, "did not allocate"),
        (_t(9, 32, 3), "torch.bfloat16"),
        (_t(9, 16, 3, dtype=torch.int32), "expected"),
        (_t(9, 32, 2, dtype=torch.int32), "expected"),
    ],
)
def test_position_plane_rejects(plane, match) -> None:
    with pytest.raises(AssertionError, match=match):
        M.check_position_plane(G, plane)


def test_attention_pools_reject_wrong_index_plane_and_kv_layout() -> None:
    kv = _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    with pytest.raises(AssertionError, match="QSA index plane"):
        M.check_attention_pools(G, 3, _t(9, 32, 1, 64), kv)
    with pytest.raises(AssertionError, match="QSA index plane"):
        M.check_attention_pools(G, 3, _t(9, 32, 1, 128, dtype=torch.float16), kv)
    with pytest.raises(AssertionError, match="paged K/V"):
        M.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 2, 4, 32, 256))
    # NHD instead of HND: heads and block swapped.
    with pytest.raises(AssertionError, match="paged K/V"):
        M.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 2, 32, 2, 256))
    # K and V not adjacent role planes.
    with pytest.raises(AssertionError, match="role planes"):
        M.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 3, 2, 32, 256))


def test_recurrent_pools_reject_wrong_key_dim_order_and_dtype() -> None:
    with pytest.raises(AssertionError, match="recurrent state pool"):
        M.check_recurrent_pools(
            G, _conv(), _t(SLOTS, G.num_v_heads, G.head_k_dim, G.head_v_dim + 1)
        )
    with pytest.raises(AssertionError, match="conv state pool"):
        M.check_recurrent_pools(G, _conv(dtype=torch.float32), _ssm())


def test_ple_pools_reject_slot_mismatch_and_int32_history() -> None:
    ple_conv = _t(SLOTS, G.stream, G.ple_state_len)
    ngram = _t(SLOTS, G.ngram_context_len, dtype=torch.int64)
    with pytest.raises(AssertionError, match="one slot space"):
        M.check_ple_pools(
            G, ple_conv, _t(SLOTS + 1, G.ngram_context_len, dtype=torch.int64), _conv()
        )
    with pytest.raises(AssertionError, match="n-gram context pool"):
        M.check_ple_pools(G, ple_conv, _t(SLOTS, G.ngram_context_len, dtype=torch.int32), _conv())
    with pytest.raises(AssertionError, match="did not allocate"):
        M.check_ple_pools(G, None, ngram, _conv())


def test_step_buffers_reject_missing_dtype_drift_and_index() -> None:
    int_conv = _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1)
    int_ssm = _t(SLOTS, 2, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32)
    idx = _t(SLOTS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="no per-step"):
        M.check_step_buffers(G, None, None, _conv(), _ssm(), idx)
    # Rank without the steps dim.
    with pytest.raises(AssertionError, match="per-step conv buffer"):
        M.check_step_buffers(G, _conv(), int_ssm, _conv(), _ssm(), idx)
    # The promotion kernel copies without converting, so the dtypes must match the pools.
    with pytest.raises(AssertionError, match="per-step recurrent buffer"):
        M.check_step_buffers(G, int_conv, int_ssm.to(torch.bfloat16), _conv(), _ssm(), idx)
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        M.check_step_buffers(G, int_conv, int_ssm, _conv(), _ssm(), idx.to(torch.int64))
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        M.check_step_buffers(G, int_conv, int_ssm, _conv(), _ssm(), None)
