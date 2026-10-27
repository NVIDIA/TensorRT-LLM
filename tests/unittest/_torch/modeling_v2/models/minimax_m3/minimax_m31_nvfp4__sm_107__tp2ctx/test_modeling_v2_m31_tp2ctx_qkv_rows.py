# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_QKV_ROWS (#20629 L2): the rows producer is bitwise the per-token NVFP4 QKV + indexer producer.

minimax_m31_nvfp4_qkv_indexer_producer_rows (one warp per head slot and 4 consecutive tokens) against
minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert on the same inputs, at the tp2ctx per-rank geometry (32 query heads, 2
KV heads, 2 index heads): equal FP8 q / index-q and equal bytes in every paged cache (NVFP4 K/V data and scales, FP8
index-K), including untouched slots, for chunk lengths on and off the 4-token warp boundary. Also the gate that sends
only long chunks there, from source. GPU (any sm_10x build carrying the op).
"""

from pathlib import Path

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"
HQ, HKV, HIDX, D, PAGE = 32, 2, 2, 128, 128


def test_long_chunks_take_the_rows_producer_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert '_QKV_ROWS = not m31_disabled("TP2CTX_QKV_ROWS")' in src
    assert "_QKV_ROWS_MIN_TOKENS = 1024" in src
    assert "q, index_q = _nvfp4_qkv_producer(num_tokens)(" in src
    assert "return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert" in src


def _inputs(num_tokens, num_pages, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    packed = torch.randn(num_tokens, (HQ + 2 * HKV + HIDX + 1) * D, generator=g, device=dev).to(
        torch.bfloat16
    )
    kv_data = torch.randint(
        0, 256, (num_pages, 2, HKV, PAGE, D // 2), generator=g, device=dev, dtype=torch.uint8
    )
    kv_scale = torch.randint(
        0, 256, (num_pages, 2, HKV, PAGE, D // 16), generator=g, device=dev, dtype=torch.uint8
    )
    index_k = torch.randint(
        0, 256, (num_pages, 1, PAGE, D), generator=g, device=dev, dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    slots = torch.randperm(num_pages * PAGE, generator=g, device=dev)[:num_tokens].to(torch.int32)
    quant = torch.tensor([1.0, 0.75, 1.25], device=dev, dtype=torch.float32)
    weights = [torch.randn(D, generator=g, device=dev).to(torch.bfloat16) * 0.1 for _ in range(4)]
    max_pos = 65536
    inv = 1.0 / (5_000_000.0 ** (torch.arange(0, 64, 2, device=dev, dtype=torch.float32) / 64))
    freq = torch.outer(torch.arange(max_pos, device=dev, dtype=torch.float32), inv)
    rope = torch.stack((freq.cos(), freq.sin()), dim=1).contiguous()
    positions = torch.randint(0, max_pos, (num_tokens,), generator=g, device=dev, dtype=torch.int32)
    return packed, kv_data, kv_scale, index_k, slots, quant, weights, rope, positions


def _run(op, inputs):
    packed, kv_data, kv_scale, index_k, slots, quant, weights, rope, positions = inputs
    kv_data, kv_scale, index_k = kv_data.clone(), kv_scale.clone(), index_k.clone()
    q, index_q = op(
        packed,
        kv_data,
        kv_scale,
        index_k,
        slots,
        quant,
        HQ,
        HKV,
        HIDX,
        D,
        64,
        1e-6,
        *weights,
        rope,
        positions,
    )
    torch.cuda.synchronize()
    return q, index_q, kv_data, kv_scale, index_k


@pytest.mark.parametrize("num_tokens", [1024, 1029, 4096, 8192])
def test_rows_producer_is_bitwise_the_per_token_producer(num_tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("sm_10x GPU required")
    rows_op = getattr(torch.ops.trtllm, "minimax_m31_nvfp4_qkv_indexer_producer_rows", None)
    if rows_op is None:
        pytest.skip("this build does not carry minimax_m31_nvfp4_qkv_indexer_producer_rows")
    inputs = _inputs(num_tokens, num_pages=num_tokens // PAGE + 8, seed=num_tokens)
    stock = _run(torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert, inputs)
    rows = _run(rows_op, inputs)
    for name, a, b in zip(("q", "index_q", "kv_data", "kv_scale", "index_k"), stock, rows):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
            f"{name} differs at {num_tokens} tokens"
        )
