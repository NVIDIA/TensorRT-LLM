# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for the DSpark CuteDSL attention op."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.models.dspark.attention import (
    _rope_last_dims_batched,
    dspark_sparse_attn,
    get_dspark_topk_idxs_batched,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._utils import get_sm_version

_ROPE_DIM = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103),
    reason="DSpark CuteDSL attention requires an SM100 or SM103 CUDA GPU",
)


def _make_inputs(
    seed: int = 0,
    batch: int = 2,
    block: int = 6,
    start_pos_values=None,
    cache_pages: int | None = None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    if start_pos_values is not None:
        batch = len(start_pos_values)
    heads, head_dim, window = 128, 512, 128
    q = torch.randn(batch, block, heads, head_dim, device=device, dtype=torch.bfloat16)
    main_kv = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    block_kv = torch.randn(batch, block, head_dim, device=device, dtype=torch.bfloat16)

    # The worker cache window is a strided view. Exercise that layout contract.
    if cache_pages is None:
        cache_pages = max(4, batch + 1)
    cache_storage = torch.randn(
        cache_pages, 3, window, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = cache_storage[:, 1]
    assert not kv_cache.is_contiguous()
    # Slots deliberately differ from decode positions so a kernel that derives
    # the window-validity mask from anything but start_pos fails the tests.
    slots = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
    if start_pos_values is None:
        start_pos_values = [199 * i + 1 for i in range(batch)]
    start_pos = torch.tensor(start_pos_values, device=device, dtype=torch.int32)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    # Per-request RoPE phases match the runtime indexing contract.
    table = precompute_dspark_freqs_cis(_ROPE_DIM, int(start_pos.max()) + block + 2, device=device)
    blk_freqs = table[start_pos.long().unsqueeze(1) + 1 + torch.arange(block, device=device)]
    inverse_rope_freqs = torch.view_as_real(blk_freqs).contiguous()
    return q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs


def _precompile_inputs(q, kv_cache):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        precompile_dspark_attention,
    )

    precompile_dspark_attention(
        q.shape[1],
        q.shape[2],
        kv_cache,
        q.shape[-1] ** -0.5,
    )


def _legacy_valid_len(start_pos):
    return (start_pos.long() + 1).clamp(max=128)


def _reference(q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, blk_freqs):
    cache = kv_cache.clone()
    window = cache.shape[1]
    cache[slots.long(), (start_pos % window).long()] = main_kv
    kv_full = torch.cat([cache[slots.long()], block_kv], dim=1)
    topk = get_dspark_topk_idxs_batched(window, q.shape[1], start_pos.long(), valid_len)
    o = dspark_sparse_attn(q, kv_full, sink, topk, q.shape[-1] ** -0.5)
    return _rope_last_dims_batched(o, _ROPE_DIM, blk_freqs, inverse=True), cache


@pytest.mark.parametrize(
    "invalid_case,reason",
    [
        ("q_dtype", "q dtype must be BF16"),
        ("head_dim", "q must have 128 heads and head_dim 512"),
        ("q_layout", "q must be contiguous"),
        ("index_dtype", "slots and start_pos must use INT32 or INT64"),
        ("q_rank", "expected q/main_kv/block_kv/kv_cache ranks 4/2/3/3"),
    ],
)
def test_fused_dspark_attention_support_gate_rejects_invalid_inputs(
    monkeypatch, invalid_case, reason
):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _legacy_valid_len(start_pos)
    inputs = [q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, freqs]
    if invalid_case == "q_dtype":
        inputs[0] = inputs[0].float()
    elif invalid_case == "head_dim":
        inputs[0] = inputs[0][..., :-1].contiguous()
    elif invalid_case == "q_layout":
        inputs[0] = inputs[0].transpose(0, 1).contiguous().transpose(0, 1)
    elif invalid_case == "index_dtype":
        inputs[5] = inputs[5].float()
    else:
        inputs[0] = inputs[0][0]

    messages = []
    monkeypatch.setattr(
        dspark_attention_op.logger,
        "debug_once",
        lambda *message, key: messages.append((" ".join(map(str, message)), key)),
    )

    assert not dspark_attention_op.is_cute_dsl_dspark_attention_supported(*inputs)
    assert len(messages) == 1
    assert reason in messages[0][0]
    assert messages[0][1][0] == "fused_dspark_attention_unsupported"


def test_cute_dsl_dspark_attention_rejects_invalid_inputs():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _legacy_valid_len(start_pos)
    inputs = [q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, freqs]
    inputs[0] = inputs[0].float()
    with pytest.raises(ValueError, match="requires contiguous BF16"):
        cute_dsl_dspark_attention(*inputs, 512**-0.5)


@pytest.mark.parametrize("block", (5, 6))
@pytest.mark.parametrize(
    ("start_pos_values", "valid_len_values"),
    [
        # Full windows with slots != start_pos: catches the window-validity
        # mask being fed anything but the absolute decode position.
        ([257, 390], [128, 128]),
        # Partially filled windows: only rows 0..start_pos are attended.
        ([5, 100], [6, 101]),
        # An odd batch exercises the same dynamic kernel as the even batches.
        ([257, 5, 390], [128, 6, 128]),
        # Bootstrapped positions with short physical suffixes exercise wraparound.
        ([257, 390], [3, 5]),
    ],
    ids=["full_window", "partial_window", "odd_batch", "wrapped_suffix"],
)
def test_cute_dsl_dspark_attention_matches_reference(block, start_pos_values, valid_len_values):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs = (
        _make_inputs(29, block=block, start_pos_values=start_pos_values)
    )
    valid_len = torch.tensor(valid_len_values, device=q.device, dtype=torch.long)
    expected, expected_cache = _reference(
        q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, blk_freqs
    )
    _precompile_inputs(q, kv_cache)

    actual = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        inverse_rope_freqs,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_cute_dsl_dspark_attention_prepared_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention_prepared,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
        cute_dsl_dspark_rmsnorm_rope_draft_block,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs = (
        _make_inputs(3, start_pos_values=[257, 5, 390])
    )
    batch, block, dim = block_kv.shape
    scale = dim**-0.5
    main_x = main_kv.unsqueeze(1)
    block_x = block_kv
    weight = torch.ones(dim, device=q.device, dtype=q.dtype)
    main_freqs = torch.zeros(batch, _ROPE_DIM // 2, 2, device=q.device)
    block_freqs = torch.zeros(batch * block, _ROPE_DIM // 2, 2, device=q.device)
    main_freqs[..., 0] = 1
    block_freqs[..., 0] = 1
    slots = slots.long()
    start_pos = start_pos.long()
    valid_len = _legacy_valid_len(start_pos)

    # All three dynamic-batch kernels are precompiled before capture.
    _precompile_inputs(q, kv_cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_x, weight, main_freqs, kv_cache, slots, start_pos, 1e-6
        )
        draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(block_x, weight, block_freqs, 1e-6)
        captured = cute_dsl_dspark_attention_prepared(
            q,
            draft_block,
            kv_cache,
            slots_i32,
            cache_seqs,
            valid_len,
            sink,
            inverse_rope_freqs,
            scale,
        )

    main_x.copy_(torch.randn_like(main_x))
    block_x.copy_(torch.randn_like(block_x))
    valid_len.copy_(torch.tensor([3, 2, 7], device=q.device))
    expected_main = cute_dsl_dspark_rmsnorm_rope(
        main_x, weight, main_freqs, 1, _ROPE_DIM, 1e-6, True, True, False
    ).squeeze(1)
    expected_block = cute_dsl_dspark_rmsnorm_rope(
        block_x, weight, block_freqs, 1, _ROPE_DIM, 1e-6, True, True, False
    )
    expected, expected_cache = _reference(
        q,
        expected_main,
        expected_block,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        blk_freqs,
    )
    graph.replay()

    torch.testing.assert_close(captured, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(draft_block[:, :block], expected_block, rtol=0, atol=0)
    torch.testing.assert_close(
        draft_block[:, block:], torch.zeros_like(draft_block[:, block:]), rtol=0, atol=0
    )


def test_cute_dsl_dspark_attention_rejects_unsupported_shapes():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        is_cute_dsl_dspark_attention_supported,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _legacy_valid_len(start_pos)
    assert is_cute_dsl_dspark_attention_supported(
        q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, freqs
    )
    # Draft lengths outside 5/6 and unsupported head counts fall back to
    # the pure-PyTorch path.
    q4 = q[:, :4].contiguous()
    assert not is_cute_dsl_dspark_attention_supported(
        q4,
        main_kv,
        block_kv[:, :4].contiguous(),
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        freqs,
    )
    q_heads = q[:, :, :24].contiguous()
    assert not is_cute_dsl_dspark_attention_supported(
        q_heads,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink[:24].contiguous(),
        freqs,
    )


@pytest.mark.parametrize("persist", (False, True))
def test_dspark_attention_forward_batched_matches_fallback(monkeypatch, persist):
    import tensorrt_llm._torch.models.dspark.attention as dspark_attention
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        precompile_dspark_attention,
    )

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, block, hidden = 2, 6, 64
    heads, head_dim, rope_dim = 128, 512, 64
    q_rank, groups, o_rank, window = 1024, 8, 32, 128

    def scaled_randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.02

    x = torch.randn(batch, block, hidden, device=device, dtype=dtype) * 0.1
    main_x = torch.randn(batch, 1, hidden, device=device, dtype=dtype) * 0.1
    start_pos = torch.tensor([5, 390], device=device, dtype=torch.long)
    slots = torch.tensor([1, 0], device=device, dtype=torch.long)
    valid_len = torch.tensor([6, 3], device=device, dtype=torch.long)
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
        "freqs_cis": dspark_attention.precompute_dspark_freqs_cis(rope_dim, 512, device=device),
        "persist": persist,
    }
    cache_storage = torch.randn(3, 3, window, head_dim, device=device, dtype=dtype) * 0.1
    op_cache = cache_storage[:, 1]
    fallback_cache = op_cache.clone()
    precompile_dspark_attention(block, heads, op_cache, head_dim**-0.5)
    calls = {
        "attention_prepared": 0,
        "cache_write": 0,
        "draft_block": 0,
        "rmsnorm_rope": 0,
    }
    op_attention = dspark_attention.cute_dsl_dspark_attention_prepared
    op_cache_write = dspark_attention.cute_dsl_dspark_rmsnorm_rope_cache_write
    op_draft_block = dspark_attention.cute_dsl_dspark_rmsnorm_rope_draft_block
    op_rmsnorm_rope = dspark_attention.cute_dsl_dspark_rmsnorm_rope

    def counted_attention(*args):
        calls["attention_prepared"] += 1
        return op_attention(*args)

    def counted_cache_write(*args):
        calls["cache_write"] += 1
        return op_cache_write(*args)

    def counted_draft_block(*args):
        calls["draft_block"] += 1
        return op_draft_block(*args)

    def counted_rmsnorm_rope(*args):
        calls["rmsnorm_rope"] += 1
        return op_rmsnorm_rope(*args)

    with monkeypatch.context() as patch:
        patch.setattr(dspark_attention, "cute_dsl_dspark_attention_prepared", counted_attention)
        patch.setattr(
            dspark_attention,
            "cute_dsl_dspark_rmsnorm_rope_cache_write",
            counted_cache_write,
        )
        patch.setattr(
            dspark_attention,
            "cute_dsl_dspark_rmsnorm_rope_draft_block",
            counted_draft_block,
        )
        patch.setattr(dspark_attention, "cute_dsl_dspark_rmsnorm_rope", counted_rmsnorm_rope)
        actual = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, op_cache, slots, valid_len, **kwargs
        )

    # Main K/V and block K/V reuse their existing RMSNorm/RoPE launches for
    # physical preparation; only q_a and per-head q use the generic op.
    assert calls == {
        "attention_prepared": 1,
        "cache_write": 1,
        "draft_block": 1,
        "rmsnorm_rope": 2,
    }

    with monkeypatch.context() as patch:
        patch.setattr(
            dspark_attention,
            "is_cute_dsl_dspark_attention_prepared_supported",
            lambda *args: False,
        )
        patch.setattr(
            dspark_attention,
            "is_cute_dsl_dspark_attention_supported",
            lambda *args: False,
        )
        patch.setattr(
            dspark_attention,
            "is_fused_dspark_rmsnorm_rope_supported",
            lambda *args: False,
        )
        expected = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, fallback_cache, slots, valid_len, **kwargs
        )

    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(op_cache, fallback_cache, rtol=2e-2, atol=2e-2)


def test_precompile_builds_one_dynamic_batch_kernel_without_runtime_jit():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _compile_dspark_attention,
        _get_dspark_arch_str,
        cute_dsl_dspark_attention,
        precompile_dspark_attention,
    )

    assert [_get_dspark_arch_str(sm) for sm in (100, 103)] == [
        "sm_100",
        "sm_103",
    ]
    assert _get_dspark_arch_str(101) is None
    assert _get_dspark_arch_str(109) is None
    assert _get_dspark_arch_str() in ("sm_100", "sm_103")

    _compile_dspark_attention.cache_clear()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs(
        11, start_pos_values=[300, 4, 250], cache_pages=40
    )
    scale = q.shape[-1] ** -0.5
    valid_len = _legacy_valid_len(start_pos)

    # A missing key must fail before mutating cache and must never invoke JIT.
    cache_before = kv_cache.clone()
    with pytest.raises(RuntimeError, match="not precompiled"):
        cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            kv_cache,
            slots,
            start_pos,
            valid_len,
            sink,
            freqs,
            scale,
        )
    torch.testing.assert_close(kv_cache, cache_before, rtol=0, atol=0)
    info = _compile_dspark_attention.cache_info()
    assert info.currsize == 0
    assert info.misses == 1
    _compile_dspark_attention.cache_clear()

    # Unsupported geometry silently no-ops rather than compiling at runtime.
    precompile_dspark_attention(4, 128, kv_cache, scale)
    precompile_dspark_attention(q.shape[1], 24, kv_cache, scale)
    assert _compile_dspark_attention.cache_info().misses == 0

    precompile_dspark_attention(q.shape[1], 128, kv_cache, scale)
    info = _compile_dspark_attention.cache_info()
    assert info.currsize == 1
    assert info.misses == 1

    # One compiled object covers different batch, page-count, and stride values.
    for batch in (1, 3, 8, 32):
        values = [5 + (37 * i) % 386 for i in range(batch)]
        args = _make_inputs(17 + batch, start_pos_values=values, cache_pages=batch + 41)
        q_b, main_b, block_b, cache_b, slots_b, pos_b, sink_b, blk_freqs_b, freqs_b = args
        if batch == 3:
            cache_b = cache_b.clone()
            assert cache_b.is_contiguous()
        valid_len_b = _legacy_valid_len(pos_b)
        expected, expected_cache = _reference(
            q_b,
            main_b,
            block_b,
            cache_b,
            slots_b,
            pos_b,
            valid_len_b,
            sink_b,
            blk_freqs_b,
        )
        actual = cute_dsl_dspark_attention(
            q_b,
            main_b,
            block_b,
            cache_b,
            slots_b,
            pos_b,
            valid_len_b,
            sink_b,
            freqs_b,
            scale,
        )
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
        torch.testing.assert_close(cache_b, expected_cache, rtol=0, atol=0)
    assert _compile_dspark_attention.cache_info().misses == 1


def test_preparation_precompile_covers_dynamic_batches_without_runtime_jit():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        precompile_dspark_attention,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_dspark_rmsnorm_rope_cache_write,
        _compile_dspark_rmsnorm_rope_draft_block,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
        cute_dsl_dspark_rmsnorm_rope_draft_block,
    )

    _compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    _compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    q, main_kv, block_kv, kv_cache, slots, start_pos, _, _, _ = _make_inputs(
        23, start_pos_values=[300, 4, 250], cache_pages=40
    )
    weight = torch.ones(512, device=q.device, dtype=q.dtype)
    main_freqs = torch.zeros(q.shape[0], _ROPE_DIM // 2, 2, device=q.device)
    block_freqs = torch.zeros(q.shape[0] * q.shape[1], _ROPE_DIM // 2, 2, device=q.device)
    main_freqs[..., 0] = 1
    block_freqs[..., 0] = 1

    cache_before = kv_cache.clone()
    with pytest.raises(RuntimeError, match="not precompiled"):
        cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_kv.unsqueeze(1),
            weight,
            main_freqs,
            kv_cache,
            slots.long(),
            start_pos.long(),
            1e-6,
        )
    with pytest.raises(RuntimeError, match="not precompiled"):
        cute_dsl_dspark_rmsnorm_rope_draft_block(block_kv, weight, block_freqs, 1e-6)
    torch.testing.assert_close(kv_cache, cache_before, rtol=0, atol=0)

    _compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    _compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    precompile_dspark_attention(q.shape[1], q.shape[2], kv_cache, q.shape[-1] ** -0.5)

    for batch in (1, 3, 8, 32):
        args = _make_inputs(31 + batch, batch=batch, cache_pages=40)
        q_b, main_b, block_b, cache_b, slots_b, pos_b, _, _, _ = args
        main_freqs_b = torch.zeros(batch, _ROPE_DIM // 2, 2, device=q.device)
        block_freqs_b = torch.zeros(batch * q_b.shape[1], _ROPE_DIM // 2, 2, device=q.device)
        main_freqs_b[..., 0] = 1
        block_freqs_b[..., 0] = 1
        slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_b.unsqueeze(1),
            weight,
            main_freqs_b,
            cache_b,
            slots_b.long(),
            pos_b.long(),
            1e-6,
        )
        draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(block_b, weight, block_freqs_b, 1e-6)
        torch.testing.assert_close(slots_i32, slots_b, rtol=0, atol=0)
        torch.testing.assert_close(cache_seqs, pos_b, rtol=0, atol=0)
        assert draft_block.shape == (batch, 8, 512)
        assert torch.count_nonzero(draft_block[:, q_b.shape[1] :]) == 0

    assert _compile_dspark_rmsnorm_rope_cache_write.cache_info().misses == 1
    assert _compile_dspark_rmsnorm_rope_draft_block.cache_info().misses == 1
