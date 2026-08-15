# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for the fused DSpark CuteDSL attention op."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.models.dspark.attention import (
    dspark_sparse_attn,
    get_dspark_topk_idxs_batched,
)
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="DSpark CuteDSL attention requires an SM100-family CUDA GPU",
)


def _make_inputs(seed: int = 0, batch: int = 2):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    block, heads, head_dim, window = 5, 24, 512, 128
    q = torch.randn(batch, block, heads, head_dim, device=device, dtype=torch.bfloat16)
    main_kv = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    block_kv = torch.randn(batch, block, head_dim, device=device, dtype=torch.bfloat16)

    # A real DSpark stage window is a strided view of
    # [max_batch, num_stages, window, head_dim]. Exercise that contract here.
    cache_storage = torch.randn(
        max(4, batch + 1), 3, window, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = cache_storage[:, 1]
    slots = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.long)
    start_pos = torch.arange(batch, device=device, dtype=torch.long) * 199 + 1
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    return q, main_kv, block_kv, kv_cache, slots, start_pos, sink


def _reference(q, main_kv, block_kv, kv_cache, slots, start_pos, sink):
    cache = kv_cache.clone()
    window = cache.shape[1]
    cache[slots, start_pos % window] = main_kv
    kv_full = torch.cat([cache[slots], block_kv], dim=1)
    topk = get_dspark_topk_idxs_batched(window, q.shape[1], start_pos)
    return dspark_sparse_attn(q, kv_full, sink, topk, q.shape[-1] ** -0.5), cache


@pytest.mark.parametrize(
    "invalid_case,reason",
    [
        ("q_dtype", "q dtype must be BF16"),
        ("head_dim", "head_dim must be 512"),
        ("q_layout", "q must be contiguous"),
        ("index_dtype", "slots and start_pos dtypes must match"),
        ("q_rank", "expected q/main_kv/block_kv/kv_cache ranks 4/2/3/3"),
    ],
)
def test_fused_dspark_attention_support_gate_rejects_invalid_inputs(
    monkeypatch, invalid_case, reason
):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    inputs = list(_make_inputs())
    if invalid_case == "q_dtype":
        inputs[0] = inputs[0].float()
    elif invalid_case == "head_dim":
        inputs[0] = inputs[0][..., :-1].contiguous()
    elif invalid_case == "q_layout":
        inputs[0] = inputs[0].transpose(0, 1).contiguous().transpose(0, 1)
    elif invalid_case == "index_dtype":
        inputs[5] = inputs[5].to(torch.int32)
    else:
        inputs[0] = inputs[0][0]

    messages = []
    monkeypatch.setattr(
        dspark_attention_op.logger,
        "debug_once",
        lambda *message, key: messages.append((" ".join(map(str, message)), key)),
    )

    assert not dspark_attention_op.is_fused_dspark_attention_supported(*inputs)
    assert len(messages) == 1
    assert reason in messages[0][0]
    assert messages[0][1][0] == "fused_dspark_attention_unsupported"


def test_cute_dsl_dspark_attention_rejects_invalid_inputs():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    inputs = list(_make_inputs())
    inputs[0] = inputs[0].float()
    with pytest.raises(ValueError, match="requires contiguous BF16"):
        cute_dsl_dspark_attention(*inputs, 512**-0.5)


def test_cute_dsl_dspark_attention_matches_reference():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    inputs = _make_inputs()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
    expected, expected_cache = _reference(*inputs)

    actual = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        sink,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_cute_dsl_dspark_attention_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    inputs = _make_inputs(3)
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
    scale = q.shape[-1] ** -0.5

    # Compile/JIT before capture. The replay must launch only the cached kernel.
    cute_dsl_dspark_attention(*inputs, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cute_dsl_dspark_attention(*inputs, scale)

    main_kv.copy_(torch.randn_like(main_kv))
    block_kv.copy_(torch.randn_like(block_kv))
    expected, expected_cache = _reference(
        q, main_kv, block_kv, kv_cache.clone(), slots, start_pos, sink
    )
    graph.replay()

    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_cute_dsl_dspark_attention_compiles_once_across_batch_sizes():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _compile_fused_dspark_attention,
        cute_dsl_dspark_attention,
    )

    _compile_fused_dspark_attention.cache_clear()
    for batch in (1, 3):
        inputs = _make_inputs(4 + batch, batch=batch)
        q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
        expected, _ = _reference(*inputs)
        actual = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            kv_cache,
            slots,
            start_pos,
            sink,
            q.shape[-1] ** -0.5,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    cache_info = _compile_fused_dspark_attention.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_dspark_attention_forward_batched_fused_matches_fallback(monkeypatch):
    import tensorrt_llm._torch.models.dspark.attention as dspark_attention

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, block, hidden = 1, 5, 64
    heads, head_dim, rope_dim = 24, 512, 64
    q_rank, groups, o_rank, window = 1024, 8, 32, 128

    def scaled_randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.02

    x = torch.randn(batch, block, hidden, device=device, dtype=dtype) * 0.1
    main_x = torch.randn(batch, 1, hidden, device=device, dtype=dtype) * 0.1
    start_pos = torch.tensor([5], device=device, dtype=torch.long)
    slots = torch.tensor([1], device=device, dtype=torch.long)
    kwargs = {
        "wq_a": scaled_randn(q_rank, hidden),
        "q_norm_w": torch.ones(q_rank, device=device, dtype=dtype),
        "wq_b": scaled_randn(heads * head_dim, q_rank),
        "wkv": scaled_randn(head_dim, hidden),
        "kv_norm_w": torch.ones(head_dim, device=device, dtype=dtype),
        "wo_a": scaled_randn(groups * o_rank, heads * head_dim // groups),
        "wo_b": scaled_randn(hidden, groups * o_rank),
        "attn_sink": torch.randn(heads, device=device, dtype=torch.float32) * 0.1,
        "n_heads": heads,
        "head_dim": head_dim,
        "rope_head_dim": rope_dim,
        "n_groups": groups,
        "o_lora_rank": o_rank,
        "window_size": window,
        "eps": 1e-6,
        "softmax_scale": head_dim**-0.5,
        "freqs_cis": dspark_attention.precompute_dspark_freqs_cis(rope_dim, 256, device=device),
        "persist": True,
    }
    cache_storage = torch.randn(3, 3, window, head_dim, device=device, dtype=dtype) * 0.1
    fused_cache = cache_storage[:, 1]
    fallback_cache = fused_cache.clone()
    calls = {"attention": 0, "rmsnorm_rope": 0}
    fused_attention = dspark_attention.cute_dsl_dspark_attention
    fused_rmsnorm_rope = dspark_attention.cute_dsl_dspark_rmsnorm_rope

    def counted_attention(*args):
        calls["attention"] += 1
        return fused_attention(*args)

    def counted_rmsnorm_rope(*args):
        calls["rmsnorm_rope"] += 1
        return fused_rmsnorm_rope(*args)

    with monkeypatch.context() as patch:
        patch.setattr(dspark_attention, "cute_dsl_dspark_attention", counted_attention)
        patch.setattr(dspark_attention, "cute_dsl_dspark_rmsnorm_rope", counted_rmsnorm_rope)
        actual = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, fused_cache, slots, **kwargs
        )

    assert calls == {"attention": 1, "rmsnorm_rope": 5}

    with monkeypatch.context() as patch:
        patch.setattr(
            dspark_attention,
            "is_fused_dspark_attention_supported",
            lambda *args: False,
        )
        patch.setattr(
            dspark_attention,
            "is_fused_dspark_rmsnorm_rope_supported",
            lambda *args: False,
        )
        expected = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, fallback_cache, slots, **kwargs
        )

    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(fused_cache, fallback_cache, rtol=2e-2, atol=2e-2)
