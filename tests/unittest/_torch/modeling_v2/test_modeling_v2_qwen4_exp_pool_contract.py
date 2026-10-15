# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's KV cache pool contract rejects the mismatches the kernels would not.

The checks read only `.shape` / `.dtype`, so these run on meta tensors with no GPU and
no engine. Each case is one wrong pool that the kernel consuming it would accept and
mis-address.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

G = M._PoolGeometry(
    tokens_per_block=32,
    heads_kv=2,
    head_dim=256,
    index_kv_heads=1,
    index_head_dim=128,
    conv_dim=8192,
    conv_kernel=4,
    num_k_heads=16,
    num_v_heads=48,
    head_v_dim=128,
    head_k_dim=128,
    stream=10240,
    ple_state_len=9,
    ngram_context_len=3,
    dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
)
# The same pools under an fp8 KV cache: K/V are e4m3, every other plane keeps its type.
G_FP8 = G._replace(kv_dtype=torch.float8_e4m3fn)
SLOTS = 6
HISTORY = 16


def _t(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _conv(**kw):
    return _t(SLOTS, G.conv_dim, G.conv_kernel - 1, **kw)


def _ssm(**kw):
    return _t(SLOTS, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32)


def _history(slots=SLOTS, history=HISTORY):
    """One GDN layer's cached-replay history buffers (old_x, old_B, old_dt)."""
    return (
        _t(slots, 2, history, G.num_v_heads, G.head_v_dim),
        _t(slots, 2, history, G.num_k_heads, G.head_k_dim),
        _t(slots, 2, G.num_v_heads, history, dtype=torch.float32),
    )


def test_valid_pools_pass() -> None:
    M.check_position_plane(G, _t(9, 32, 3, dtype=torch.int32))
    M.check_attention_pools(
        G, 3, _t(9, 32, 1, 128), _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    )
    M.check_attention_pools(
        G_FP8,
        3,
        _t(9, 32, 1, 128),
        _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim, dtype=torch.float8_e4m3fn),
    )
    M.check_recurrent_pools(G, _conv(), _ssm())
    M.check_ple_pools(
        G,
        _t(SLOTS, G.stream, G.ple_state_len),
        _t(SLOTS, G.ngram_context_len, dtype=torch.int64),
        _conv(),
    )
    M.check_step_buffers(
        G, _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1), _conv(), _t(SLOTS, dtype=torch.int32)
    )
    M.check_legacy_state(
        G, _t(SLOTS, 2, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32), _ssm()
    )
    M.check_replay_state(
        HISTORY, 3, _t(SLOTS, dtype=torch.int32), _t(SLOTS, dtype=torch.int32), _ssm()
    )
    M.check_replay_history(G, 3, HISTORY, SLOTS, *_history())
    M.check_shared_selection(_t(2, 8, 512, dtype=torch.int32), {3: 1}, 3, 512)


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


def test_attention_pools_reject_the_other_storage_dtype() -> None:
    """The attention op reads the pool through one `quant_mode`, so a pool of the other
    storage would be read through the wrong format. The draft layer's bind runs the same check."""
    index = _t(9, 32, 1, 128)
    bf16_kv = _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    fp8_kv = bf16_kv.to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="paged K/V"):
        M.check_attention_pools(G, 3, index, fp8_kv)
    with pytest.raises(AssertionError, match="paged K/V"):
        M.check_attention_pools(G, 3, index, bf16_kv.to(torch.float16))
    with pytest.raises(AssertionError, match="paged K/V"):
        M.check_attention_pools(G_FP8, 3, index, bf16_kv)
    # The index plane is a kernel contract, not KV storage: bf16 under either algorithm.
    with pytest.raises(AssertionError, match="QSA index plane"):
        M.check_attention_pools(G_FP8, 3, index.to(torch.float8_e4m3fn), fp8_kv)


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


def test_step_buffers_reject_missing_conv_rank_dtype_and_index() -> None:
    int_conv = _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1)
    idx = _t(SLOTS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="no per-step conv"):
        M.check_step_buffers(G, None, _conv(), idx)
    # Rank without the steps dim.
    with pytest.raises(AssertionError, match="per-step conv buffer"):
        M.check_step_buffers(G, _conv(), _conv(), idx)
    # The promotion kernel copies without converting, so the dtype must match the pool.
    with pytest.raises(AssertionError, match="per-step conv buffer"):
        M.check_step_buffers(G, int_conv.to(torch.float32), _conv(), idx)
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        M.check_step_buffers(G, int_conv, _conv(), idx.to(torch.int64))
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        M.check_step_buffers(G, int_conv, _conv(), None)


def test_legacy_state_rejects_missing_rank_and_dtype_drift() -> None:
    int_ssm = _t(SLOTS, 2, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=torch.float32)
    with pytest.raises(AssertionError, match="no per-step recurrent"):
        M.check_legacy_state(G, None, _ssm())
    with pytest.raises(AssertionError, match="per-step recurrent buffer"):
        M.check_legacy_state(G, _ssm(), _ssm())
    with pytest.raises(AssertionError, match="per-step recurrent buffer"):
        M.check_legacy_state(G, int_ssm.to(torch.bfloat16), _ssm())


def test_replay_state_rejects_pool_dtype_geometry_and_bookkeeping() -> None:
    pnat = _t(SLOTS, dtype=torch.int32)
    buf = _t(SLOTS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="fp32 checkpoint"):
        M.check_replay_state(HISTORY, 3, pnat, buf, _ssm().to(torch.bfloat16))
    with pytest.raises(AssertionError, match="replay_history_size"):
        M.check_replay_state(12, 3, pnat, buf, _ssm())
    for width in (1, 5):
        with pytest.raises(AssertionError, match="replay_step_width"):
            M.check_replay_state(HISTORY, width, pnat, buf, _ssm())
    # The bookkeeping is one int32 per pool slot, so it must cover the whole pool.
    with pytest.raises(AssertionError, match="prev_num_accepted_tokens"):
        M.check_replay_state(HISTORY, 3, _t(SLOTS - 1, dtype=torch.int32), buf, _ssm())
    with pytest.raises(AssertionError, match="cache_buf_idx"):
        M.check_replay_state(HISTORY, 3, pnat, buf.to(torch.int64), _ssm())


def test_replay_width_admits_only_the_managers_step_width() -> None:
    replay = SimpleNamespace(width=3)
    M.check_replay_width(None, 1)  # the legacy recurrence takes any width
    M.check_replay_width(replay, 3)
    for rows in (1, 2, 4):
        with pytest.raises(AssertionError, match="step width 3"):
            M.check_replay_width(replay, rows)


def test_replay_history_rejects_missing_short_pool_and_wrong_geometry() -> None:
    old_x, old_B, old_dt = _history()
    with pytest.raises(AssertionError, match="no old_x"):
        M.check_replay_history(G, 3, HISTORY, SLOTS, None, old_B, old_dt)
    with pytest.raises(AssertionError, match="slots for a"):
        M.check_replay_history(G, 3, HISTORY, SLOTS, *_history(slots=SLOTS - 1))
    # The kernel addresses the history by `history_size`, so a buffer of another depth
    # would be read at the wrong offsets.
    with pytest.raises(AssertionError, match="old_x"):
        M.check_replay_history(G, 3, HISTORY, SLOTS, *_history(history=12))
    with pytest.raises(AssertionError, match="old_B"):
        M.check_replay_history(G, 3, HISTORY, SLOTS, old_x, old_B.to(torch.float32), old_dt)
    with pytest.raises(AssertionError, match="old_dt"):
        M.check_replay_history(G, 3, HISTORY, SLOTS, old_x, old_B, old_dt.to(torch.bfloat16))
    with pytest.raises(AssertionError, match="double-buffered"):
        M.check_replay_history(
            G,
            3,
            HISTORY,
            SLOTS,
            old_x,
            old_B,
            _t(SLOTS, 3, G.num_v_heads, HISTORY, dtype=torch.float32),
        )


def test_shared_selection_rejects_missing_row_and_width() -> None:
    shared = _t(2, 8, 512, dtype=torch.int32)
    with pytest.raises(AssertionError, match="no shared buffers"):
        M.check_shared_selection(None, {3: 0}, 3, 512)
    with pytest.raises(AssertionError, match="no row"):
        M.check_shared_selection(shared, {4: 0}, 3, 512)
    with pytest.raises(AssertionError, match="no row"):
        M.check_shared_selection(shared, {3: 2}, 3, 512)
    with pytest.raises(AssertionError, match="blocks per row"):
        M.check_shared_selection(shared, {3: 1}, 3, 256)


def test_routing_admits_exactly_the_moe_backends_the_target_implements() -> None:
    """`route` declines a backend request the constructor would assert on."""
    assert set(routing._MOE_BACKENDS) == set(M._TRUNK_MOE_BACKENDS)
