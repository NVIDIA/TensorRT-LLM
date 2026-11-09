# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qwen4_exp target's KV cache pool contract (`pool.py`) rejects the mismatches the
kernels would not.

The checks read only `.shape` / `.dtype`, so these run on meta tensors with no GPU and
no engine. Each case is one wrong pool that the kernel consuming it would accept and
mis-address. The geometry is one rank's at tp1, which holds every head.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    pool as P,
)

pytestmark = pytest.mark.cpu_only

G = P.PoolGeometry(
    tokens_per_block=32,
    heads_kv=2,
    kv_head_start=0,
    head_dim=256,
    index_kv_heads=1,
    index_head_dim=128,
    conv_dim=8192,
    conv_kernel=4,
    num_k_heads=16,
    k_head_start=0,
    num_v_heads=48,
    v_head_start=0,
    head_v_dim=128,
    head_k_dim=128,
    stream=10240,
    ple_state_len=9,
    ngram_context_len=3,
    dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    kv_width=256,
)
# The same pools under an fp8 KV cache: K/V are e4m3, every other plane keeps its type.
G_FP8 = G._replace(kv_dtype=torch.float8_e4m3fn)
# Under NVFP4 the K/V pool is uint8 e2m1 pairs, half a head wide.
G_NVFP4 = G._replace(kv_dtype=torch.uint8, kv_width=128)
SLOTS = 6
HISTORY = 16


def _t(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _conv(**kw):
    return _t(SLOTS, G.conv_dim, G.conv_kernel - 1, **kw)


def _ssm(dtype=torch.float32):
    return _t(SLOTS, G.num_v_heads, G.head_v_dim, G.head_k_dim, dtype=dtype)


def _history(slots=SLOTS, history=HISTORY):
    """One GDN layer's cached-replay history buffers (old_x, old_B, old_dt)."""
    return (
        _t(slots, 2, history, G.num_v_heads, G.head_v_dim),
        _t(slots, 2, history, G.num_k_heads, G.head_k_dim),
        _t(slots, 2, G.num_v_heads, history, dtype=torch.float32),
    )


def test_valid_pools_pass() -> None:
    P.check_position_plane(G, _t(9, 32, 3, dtype=torch.int32))
    P.check_attention_pools(
        G, 3, _t(9, 32, 1, 128), _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    )
    P.check_attention_pools(
        G_FP8,
        3,
        _t(9, 32, 1, 128),
        _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim, dtype=torch.float8_e4m3fn),
    )
    nvfp4_kv = _t(9, 2, G.heads_kv, G.tokens_per_block, 128, dtype=torch.uint8)
    P.check_attention_pools(G_NVFP4, 3, _t(9, 32, 1, 128), nvfp4_kv)
    for dtype in (torch.float32, torch.bfloat16):
        P.check_recurrent_pools(G, _conv(), _ssm(dtype))
    P.check_ple_pools(
        G,
        _t(SLOTS, G.stream, G.ple_state_len),
        _t(SLOTS, G.ngram_context_len, dtype=torch.int64),
        _conv(),
    )
    P.check_step_buffers(
        G, _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1), _conv(), _t(SLOTS, dtype=torch.int32)
    )
    for dtype in (torch.float32, torch.bfloat16):
        for width in (2, 5, P.MAX_VERIFY_WIDTH):
            P.check_replay_state(
                HISTORY,
                width,
                _t(SLOTS, dtype=torch.int32),
                _t(SLOTS, dtype=torch.int32),
                _ssm(dtype),
            )
    P.check_replay_history(G, 3, HISTORY, SLOTS, *_history())
    for rows in range(1, P.MAX_VERIFY_WIDTH + 1):
        P.check_verify_width(rows)
    P.check_shared_selection(_t(2, 8, 512, dtype=torch.int32), {3: 1}, 3, 512)


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
        P.check_position_plane(G, plane)


def test_attention_pools_reject_wrong_index_plane_and_kv_layout() -> None:
    kv = _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    with pytest.raises(AssertionError, match="QSA index plane"):
        P.check_attention_pools(G, 3, _t(9, 32, 1, 64), kv)
    with pytest.raises(AssertionError, match="QSA index plane"):
        P.check_attention_pools(G, 3, _t(9, 32, 1, 128, dtype=torch.float16), kv)
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 2, 4, 32, 256))
    # NHD instead of HND: heads and block swapped.
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 2, 32, 2, 256))
    # K and V not adjacent role planes.
    with pytest.raises(AssertionError, match="role planes"):
        P.check_attention_pools(G, 3, _t(9, 32, 1, 128), _t(9, 3, 2, 32, 256))


def test_attention_pools_reject_the_other_storage_dtype() -> None:
    """The attention op reads the pool through one `quant_mode`, so a pool of the other
    storage would be read through the wrong format. The draft layer's bind runs the same check."""
    index = _t(9, 32, 1, 128)
    bf16_kv = _t(9, 2, G.heads_kv, G.tokens_per_block, G.head_dim)
    fp8_kv = bf16_kv.to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(G, 3, index, fp8_kv)
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(G, 3, index, bf16_kv.to(torch.float16))
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(G_FP8, 3, index, bf16_kv)
    # The index plane is a kernel contract, not KV storage: bf16 under either algorithm.
    with pytest.raises(AssertionError, match="QSA index plane"):
        P.check_attention_pools(G_FP8, 3, index.to(torch.float8_e4m3fn), fp8_kv)
    # An NVFP4 pool is half a head wide: a full-width uint8 pool is the other storage.
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(
            G_NVFP4, 3, index, _t(9, 2, G.heads_kv, G.tokens_per_block, 256, dtype=torch.uint8)
        )
    with pytest.raises(AssertionError, match="paged K/V"):
        P.check_attention_pools(
            G, 3, index, _t(9, 2, G.heads_kv, G.tokens_per_block, 128, dtype=torch.uint8)
        )


def _strided(shape, slot_stride, dtype=torch.uint8):
    """A view of `shape` whose slot stride is `slot_stride`, as a manager's role planes of a
    lifecycle slot holding several layers' pages are."""
    pages = shape[0] * slot_stride
    flat = torch.empty(pages, dtype=dtype, device="meta")
    return flat.as_strided(shape, (slot_stride, *torch.empty(shape[1:], device="meta").stride()))


def test_kv_scale_planes_must_hold_the_data_pools_pages_per_slot() -> None:
    """The attention op addresses the scale pages with the data pages' block table, so a
    scale pool with another number of pages per slot would be read at other slots' scales.
    The engine's trunk holds 24 data role pages per slot (12 layers x K, V): 196608 bytes
    against 24576 of scales."""
    data = _strided((9, 2, G.heads_kv, G.tokens_per_block, 128), 24 * 8192)
    scales = _strided((9, 2, G.heads_kv, G.tokens_per_block, 16), 24 * 1024)
    P.check_kv_scale_planes(G_NVFP4, 3, data, scales)
    # A side page coalesced into the data pool: 36 pages per slot against 24.
    coalesced = _strided((9, 2, G.heads_kv, G.tokens_per_block, 128), 36 * 8192)
    with pytest.raises(AssertionError, match="36 pages per slot.*24"):
        P.check_kv_scale_planes(G_NVFP4, 3, coalesced, scales)
    with pytest.raises(AssertionError, match="whole number"):
        P.check_kv_scale_planes(G_NVFP4, 3, data, _strided((9, 2, 2, 32, 16), 24 * 1024 + 1))


def test_kv_scale_planes_reject_the_wrong_block_scale_width() -> None:
    data = _t(9, 2, G.heads_kv, G.tokens_per_block, 128, dtype=torch.uint8)
    with pytest.raises(AssertionError, match="NVFP4 block scales"):
        P.check_kv_scale_planes(G_NVFP4, 3, data, _t(9, 2, 2, 32, 8, dtype=torch.uint8))


def test_recurrent_pools_reject_wrong_key_dim_order_and_dtype() -> None:
    with pytest.raises(AssertionError, match="recurrent state pool"):
        P.check_recurrent_pools(
            G, _conv(), _t(SLOTS, G.num_v_heads, G.head_k_dim, G.head_v_dim + 1)
        )
    with pytest.raises(AssertionError, match="conv state pool"):
        P.check_recurrent_pools(G, _conv(dtype=torch.float32), _ssm())
    # The engine accepts an fp16 pool; no recurrence entry the target binds is certified at it.
    with pytest.raises(AssertionError, match="recurrent state pool"):
        P.check_recurrent_pools(G, _conv(), _ssm(torch.float16))


def test_ple_pools_reject_slot_mismatch_and_int32_history() -> None:
    ple_conv = _t(SLOTS, G.stream, G.ple_state_len)
    ngram = _t(SLOTS, G.ngram_context_len, dtype=torch.int64)
    with pytest.raises(AssertionError, match="one slot space"):
        P.check_ple_pools(
            G, ple_conv, _t(SLOTS + 1, G.ngram_context_len, dtype=torch.int64), _conv()
        )
    with pytest.raises(AssertionError, match="n-gram context pool"):
        P.check_ple_pools(G, ple_conv, _t(SLOTS, G.ngram_context_len, dtype=torch.int32), _conv())
    with pytest.raises(AssertionError, match="did not allocate"):
        P.check_ple_pools(G, None, ngram, _conv())


def test_step_buffers_reject_missing_conv_rank_dtype_and_index() -> None:
    int_conv = _t(SLOTS, 2, G.conv_dim, G.conv_kernel - 1)
    idx = _t(SLOTS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="no per-step conv"):
        P.check_step_buffers(G, None, _conv(), idx)
    # Rank without the steps dim.
    with pytest.raises(AssertionError, match="per-step conv buffer"):
        P.check_step_buffers(G, _conv(), _conv(), idx)
    # The promotion kernel copies without converting, so the dtype must match the pool.
    with pytest.raises(AssertionError, match="per-step conv buffer"):
        P.check_step_buffers(G, int_conv.to(torch.float32), _conv(), idx)
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        P.check_step_buffers(G, int_conv, _conv(), idx.to(torch.int64))
    with pytest.raises(AssertionError, match="intermediate_state_indices"):
        P.check_step_buffers(G, int_conv, _conv(), None)


def test_replay_state_rejects_pool_dtype_geometry_and_bookkeeping() -> None:
    pnat = _t(SLOTS, dtype=torch.int32)
    buf = _t(SLOTS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="fp32 or a bf16 checkpoint"):
        P.check_replay_state(HISTORY, 3, pnat, buf, _ssm(torch.float16))
    with pytest.raises(AssertionError, match="replay_history_size"):
        P.check_replay_state(12, 3, pnat, buf, _ssm())
    for width in (1, P.MAX_VERIFY_WIDTH + 1):
        with pytest.raises(AssertionError, match="replay_step_width"):
            P.check_replay_state(HISTORY, width, pnat, buf, _ssm())
    # The bookkeeping is one int32 per pool slot, so it must cover the whole pool.
    with pytest.raises(AssertionError, match="prev_num_accepted_tokens"):
        P.check_replay_state(HISTORY, 3, _t(SLOTS - 1, dtype=torch.int32), buf, _ssm())
    with pytest.raises(AssertionError, match="cache_buf_idx"):
        P.check_replay_state(HISTORY, 3, pnat, buf.to(torch.int64), _ssm())


def test_verify_width_admits_one_to_eight_rows_per_request() -> None:
    for rows in (0, P.MAX_VERIFY_WIDTH + 1):
        with pytest.raises(AssertionError, match="predicted_tokens_per_seq"):
            P.check_verify_width(rows)


def test_replay_width_admits_only_the_managers_step_width() -> None:
    replay = SimpleNamespace(width=3)
    P.check_replay_width(None, 1)  # without the cached replay any width
    P.check_replay_width(replay, 3)
    for rows in (1, 2, 4):
        with pytest.raises(AssertionError, match="step width 3"):
            P.check_replay_width(replay, rows)


def test_replay_history_rejects_missing_short_pool_and_wrong_geometry() -> None:
    old_x, old_B, old_dt = _history()
    with pytest.raises(AssertionError, match="no old_x"):
        P.check_replay_history(G, 3, HISTORY, SLOTS, None, old_B, old_dt)
    with pytest.raises(AssertionError, match="slots for a"):
        P.check_replay_history(G, 3, HISTORY, SLOTS, *_history(slots=SLOTS - 1))
    # The kernel addresses the history by `history_size`, so a buffer of another depth
    # would be read at the wrong offsets.
    with pytest.raises(AssertionError, match="old_x"):
        P.check_replay_history(G, 3, HISTORY, SLOTS, *_history(history=12))
    with pytest.raises(AssertionError, match="old_B"):
        P.check_replay_history(G, 3, HISTORY, SLOTS, old_x, old_B.to(torch.float32), old_dt)
    with pytest.raises(AssertionError, match="old_dt"):
        P.check_replay_history(G, 3, HISTORY, SLOTS, old_x, old_B, old_dt.to(torch.bfloat16))
    with pytest.raises(AssertionError, match="double-buffered"):
        P.check_replay_history(
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
        P.check_shared_selection(None, {3: 0}, 3, 512)
    with pytest.raises(AssertionError, match="no row"):
        P.check_shared_selection(shared, {4: 0}, 3, 512)
    with pytest.raises(AssertionError, match="no row"):
        P.check_shared_selection(shared, {3: 2}, 3, 512)
    with pytest.raises(AssertionError, match="blocks per row"):
        P.check_shared_selection(shared, {3: 1}, 3, 256)
