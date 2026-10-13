# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU correctness tests for the DSpark CuteDSL attention op."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
)
from tensorrt_llm._torch.models.modeling_dspark import (
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
    or get_sm_version() not in (100, 103, 107),
    reason="DSpark CuteDSL attention requires an SM100, SM103, or SM107 CUDA GPU",
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

    # A real DSpark stage window is a strided view of
    # [max_batch, num_stages, window, head_dim]. Exercise that contract here.
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
    # Per-request RoPE phases for the draft block positions, exactly as the
    # production path builds them (freqs_cis[start_pos + 1 + j]).
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


def _with_row_padding(x: torch.Tensor, padding: int = 128) -> torch.Tensor:
    storage = torch.empty((*x.shape[:-1], x.shape[-1] + padding), device=x.device, dtype=x.dtype)
    padded = storage[..., : x.shape[-1]]
    padded.copy_(x)
    return padded


def _reference(q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, blk_freqs):
    cache = kv_cache.clone()
    window = cache.shape[1]
    cache[slots.long(), (start_pos % window).long()] = main_kv
    kv_full = torch.cat([cache[slots.long()], block_kv], dim=1)
    topk = get_dspark_topk_idxs_batched(window, q.shape[1], start_pos.long(), valid_len)
    o = dspark_sparse_attn(q, kv_full, sink, topk, q.shape[-1] ** -0.5)
    return _rope_last_dims_batched(o, _ROPE_DIM, blk_freqs, inverse=True), cache


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
        cute_dsl_dspark_rmsnorm_rope_block_page,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs = (
        _make_inputs(3, start_pos_values=[257, 5, 390])
    )
    batch, block, dim = block_kv.shape
    scale = dim**-0.5
    main_x = _with_row_padding(main_kv.unsqueeze(1))
    block_x = _with_row_padding(block_kv)
    assert not main_x.is_contiguous()
    assert not block_x.is_contiguous()
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
        block_page = cute_dsl_dspark_rmsnorm_rope_block_page(block_x, weight, block_freqs, 1e-6)
        captured = cute_dsl_dspark_attention_prepared(
            q,
            block_page,
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
    torch.testing.assert_close(block_page[:, :block], expected_block, rtol=0, atol=0)
    torch.testing.assert_close(
        block_page[:, block:], torch.zeros_like(block_page[:, block:]), rtol=0, atol=0
    )


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="The fused DSpark FP8 output projection is a Rubin specialization",
)
def test_cute_dsl_dspark_attention_prepared_fp8_matches_separate_quant_and_replays_graph():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention_prepared,
        cute_dsl_dspark_attention_prepared_fp8,
        precompile_dspark_attention,
    )

    groups = 16
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, inverse_rope_freqs = _make_inputs(
        47, start_pos_values=[257, 390]
    )
    block = q.shape[1]
    block_page = torch.zeros(q.shape[0], 8, q.shape[-1], device=q.device, dtype=q.dtype)
    block_page[:, :block] = block_kv
    kv_cache[slots.long(), (start_pos % kv_cache.shape[1]).long()] = main_kv
    valid_len = torch.tensor([3, 5], device=q.device, dtype=torch.long)
    scale = q.shape[-1] ** -0.5

    precompile_dspark_attention(block, q.shape[2], kv_cache, scale, output_groups=groups)

    def separate_quantize():
        bf16 = cute_dsl_dspark_attention_prepared(
            q,
            block_page,
            kv_cache,
            slots,
            start_pos,
            valid_len,
            sink,
            inverse_rope_freqs,
            scale,
        )
        rows = q.shape[0] * block
        grouped = bf16.reshape(rows, groups, -1).permute(1, 0, 2).contiguous()
        quant_blocks = grouped.view(groups, rows, -1, 128)
        amax = quant_blocks.abs().amax(dim=-1).float().clamp_min(1e-10)
        quant_scale = 448.0 / amax
        quantized = (quant_blocks.float() * quant_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        scale_rows = (rows + 3) // 4 * 4
        scales = torch.zeros(
            groups,
            quant_blocks.shape[2],
            scale_rows,
            device=q.device,
            dtype=torch.float32,
        )
        scales[..., :rows] = (1.0 / quant_scale).transpose(1, 2)
        return quantized.view_as(grouped), scales

    def assert_quantized_close(actual_fp8, actual_scale, expected_fp8, expected_scale):
        rows = q.shape[0] * block
        torch.testing.assert_close(
            actual_scale[..., :rows], expected_scale[..., :rows], rtol=1e-6, atol=0
        )
        actual = actual_fp8.float()
        expected = expected_fp8.float()
        mismatch_rate = torch.count_nonzero(actual != expected) / actual.numel()
        assert mismatch_rate <= 1e-3
        # CuTe and Torch can choose neighboring E4M3 values at a rounding tie.
        torch.testing.assert_close(actual, expected, rtol=0.125, atol=0.015625)

    actual_fp8, actual_scale = cute_dsl_dspark_attention_prepared_fp8(
        q,
        block_page,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        inverse_rope_freqs,
        scale,
        groups,
    )
    expected_fp8, expected_scale = separate_quantize()
    assert_quantized_close(actual_fp8, actual_scale, expected_fp8, expected_scale)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_fp8, captured_scale = cute_dsl_dspark_attention_prepared_fp8(
            q,
            block_page,
            kv_cache,
            slots,
            start_pos,
            valid_len,
            sink,
            inverse_rope_freqs,
            scale,
            groups,
        )
    q.copy_(torch.randn_like(q))
    valid_len.copy_(torch.tensor([7, 2], device=q.device))
    graph.replay()
    expected_fp8, expected_scale = separate_quantize()
    assert_quantized_close(captured_fp8, captured_scale, expected_fp8, expected_scale)


def test_cute_dsl_dspark_attention_rejects_unsupported_shapes():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        is_cute_dsl_dspark_attention_supported,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _legacy_valid_len(start_pos)
    assert is_cute_dsl_dspark_attention_supported(
        q, main_kv, block_kv, kv_cache, slots, start_pos, valid_len, sink, freqs
    )
    # Draft lengths outside 5/6 and non-production head counts fall back to
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
@pytest.mark.parametrize("explicit_validity", (False, True))
def test_dspark_attention_forward_batched_matches_fallback(monkeypatch, persist, explicit_validity):
    import tensorrt_llm._torch.models.modeling_dspark as dspark_attention
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
    dispatch_valid_len = valid_len if explicit_validity else None
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
        "block_page": 0,
        "rmsnorm_rope": 0,
    }
    prepared_valid_lens = []
    op_attention = dspark_attention.cute_dsl_dspark_attention_prepared
    op_cache_write = dspark_attention.cute_dsl_dspark_rmsnorm_rope_cache_write
    op_block_page = dspark_attention.cute_dsl_dspark_rmsnorm_rope_block_page
    op_rmsnorm_rope = dspark_attention.cute_dsl_dspark_rmsnorm_rope

    def counted_attention(*args):
        calls["attention_prepared"] += 1
        prepared_valid_lens.append(args[5])
        return op_attention(*args)

    def counted_cache_write(*args):
        calls["cache_write"] += 1
        return op_cache_write(*args)

    def counted_block_page(*args):
        calls["block_page"] += 1
        return op_block_page(*args)

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
            "cute_dsl_dspark_rmsnorm_rope_block_page",
            counted_block_page,
        )
        patch.setattr(dspark_attention, "cute_dsl_dspark_rmsnorm_rope", counted_rmsnorm_rope)
        actual = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, op_cache, slots, dispatch_valid_len, **kwargs
        )

    # Both explicit physical-window validity (including the wrapped short
    # suffix above) and legacy synthesized validity use the prepared path.
    assert calls == {
        "attention_prepared": 1,
        "cache_write": 1,
        "block_page": 1,
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
            x, main_x, start_pos, fallback_cache, slots, dispatch_valid_len, **kwargs
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

    assert [_get_dspark_arch_str(sm) for sm in (100, 103, 107)] == [
        "sm_100",
        "sm_103",
        "sm_107",
    ]
    assert _get_dspark_arch_str(101) is None
    assert _get_dspark_arch_str(109) is None
    assert _get_dspark_arch_str() in ("sm_100", "sm_103", "sm_107")

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

    # Unsupported geometry silently no-ops rather than compiling during serving.
    precompile_dspark_attention(4, 128, kv_cache, scale)
    precompile_dspark_attention(q.shape[1], 24, kv_cache, scale)
    assert _compile_dspark_attention.cache_info().misses == 0

    precompile_dspark_attention(q.shape[1], 128, kv_cache, scale)
    info = _compile_dspark_attention.cache_info()
    assert info.currsize == 1
    assert info.misses == 1

    # Every runtime batch, page count, and page stride is a hot hit on the same
    # compiled object without compiler calls or runtime padding.
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
        _compile_dspark_rmsnorm_rope_block_page,
        _compile_dspark_rmsnorm_rope_cache_write,
        cute_dsl_dspark_rmsnorm_rope_block_page,
        cute_dsl_dspark_rmsnorm_rope_cache_write,
    )

    _compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    _compile_dspark_rmsnorm_rope_block_page.cache_clear()
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
        cute_dsl_dspark_rmsnorm_rope_block_page(block_kv, weight, block_freqs, 1e-6)
    torch.testing.assert_close(kv_cache, cache_before, rtol=0, atol=0)

    _compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    _compile_dspark_rmsnorm_rope_block_page.cache_clear()
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
        block_page = cute_dsl_dspark_rmsnorm_rope_block_page(block_b, weight, block_freqs_b, 1e-6)
        torch.testing.assert_close(slots_i32, slots_b, rtol=0, atol=0)
        torch.testing.assert_close(cache_seqs, pos_b, rtol=0, atol=0)
        assert block_page.shape == (batch, 8, 512)
        assert torch.count_nonzero(block_page[:, q_b.shape[1] :]) == 0

    assert _compile_dspark_rmsnorm_rope_cache_write.cache_info().misses == 1
    assert _compile_dspark_rmsnorm_rope_block_page.cache_info().misses == 1


@pytest.mark.skipif(
    get_sm_version() != 107 or not IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
    reason="The fused DSpark FP8 input producers require SM107 Rubin CuTe DSL",
)
def test_dspark_fp8_input_producers_match_reference_and_reuse_dynamic_m_compile():
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_dspark_concat_fp8_quant,
        _compile_dspark_rmsnorm_fp8_quant,
        cute_dsl_dspark_concat_fp8_quant,
        cute_dsl_dspark_rmsnorm_fp8_quant,
        precompile_dspark_fp8_input_fusions,
    )
    from tensorrt_llm._torch.models.modeling_dspark import _rmsnorm

    torch.manual_seed(53)
    hidden_dim = 1536
    eps = 1e-6
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.bfloat16)
    _compile_dspark_concat_fp8_quant.cache_clear()
    _compile_dspark_rmsnorm_fp8_quant.cache_clear()

    def active_scales(scale: torch.Tensor, rows: int) -> torch.Tensor:
        offsets = []
        quant_blocks = hidden_dim // 128
        for row in range(rows):
            for quant_block in range(quant_blocks):
                base = (
                    (row // 128) * quant_blocks * 512
                    + quant_block * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                )
                offsets.extend(range(base, base + 4))
        return scale[torch.tensor(offsets, device=scale.device, dtype=torch.long)]

    missing = torch.randn(2, hidden_dim, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not precompiled"):
        cute_dsl_dspark_concat_fp8_quant(missing[:1], missing[1:])
    with pytest.raises(RuntimeError, match="not precompiled"):
        cute_dsl_dspark_rmsnorm_fp8_quant(missing, weight, eps)

    _compile_dspark_concat_fp8_quant.cache_clear()
    _compile_dspark_rmsnorm_fp8_quant.cache_clear()
    precompile_dspark_fp8_input_fusions(hidden_dim, hidden_dim, eps)
    assert _compile_dspark_concat_fp8_quant.cache_info().misses == 1
    assert _compile_dspark_rmsnorm_fp8_quant.cache_info().misses == 1

    for first_rows, second_rows in ((1, 1), (3, 10), (64, 65)):
        first_storage = torch.randn(
            first_rows, hidden_dim + 128, device="cuda", dtype=torch.bfloat16
        )
        second_storage = torch.randn(
            second_rows, hidden_dim + 256, device="cuda", dtype=torch.bfloat16
        )
        first = first_storage[:, :hidden_dim]
        second = second_storage[:, :hidden_dim]
        if first_rows > 1:
            assert not first.is_contiguous()
        if second_rows > 1:
            assert not second.is_contiguous()

        expected_input = torch.cat((first, second), dim=0)
        expected_fp8, expected_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
            expected_input
        )
        actual_fp8, actual_scale = cute_dsl_dspark_concat_fp8_quant(first, second)
        torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            active_scales(actual_scale, first_rows + second_rows),
            active_scales(expected_scale, first_rows + second_rows),
            rtol=0,
            atol=0,
        )

        norm_storage = torch.randn(
            first_rows + second_rows,
            hidden_dim + 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        norm_input = norm_storage[:, :hidden_dim]
        expected_norm = _rmsnorm(norm_input, weight, eps)
        expected_fp8, expected_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
            expected_norm
        )
        actual_fp8, actual_scale = cute_dsl_dspark_rmsnorm_fp8_quant(norm_input, weight, eps)
        torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            active_scales(actual_scale, first_rows + second_rows),
            active_scales(expected_scale, first_rows + second_rows),
            rtol=0,
            atol=0,
        )

    # Runtime row counts and row strides reuse the two initialization artifacts.
    assert _compile_dspark_concat_fp8_quant.cache_info().misses == 1
    assert _compile_dspark_rmsnorm_fp8_quant.cache_info().misses == 1


@pytest.mark.skipif(
    get_sm_version() != 107 or not IS_CUTLASS_DSL_INTERNAL_AVAILABLE,
    reason="The fused DSpark output projection requires SM107 Rubin CuTe DSL",
)
def test_dspark_fused_fp8_output_projection_matches_separate_and_reuses_cache():
    from _torch.helpers import per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLFp8RubinBmmQuantRunner,
        CuteDSLMXFP8RubinLinear,
    )
    from tensorrt_llm._torch.models.modeling_dspark import _output_projection
    from tensorrt_llm.quantization.utils import fp8_utils

    torch.manual_seed(41)
    batch, block, groups = 2, 6, 8
    rank, group_dim, hidden = 128, 512, 128
    rows, flat_n = batch * block, groups * rank
    o = (
        torch.randn(
            batch,
            block,
            groups,
            group_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / group_dim
    )
    o_a = torch.randn(groups, rank, group_dim, device="cuda", dtype=torch.bfloat16) / group_dim
    o_a_fp8 = torch.empty_like(o_a, dtype=torch.float8_e4m3fn)
    o_a_scale = torch.empty(
        groups, rank // 128, group_dim // 128, device="cuda", dtype=torch.float32
    )
    for group in range(groups):
        o_a_fp8[group], o_a_scale[group] = per_block_cast_to_fp8(o_a[group])

    wo_b = torch.randn(hidden, flat_n, device="cuda", dtype=torch.bfloat16) / flat_n
    wo_b_fp8, wo_b_scale = per_block_cast_to_fp8_e8m0(wo_b)
    wo_b_scale = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(
        wo_b_scale, mn=hidden, k=flat_n
    )

    class OutputLinear:
        has_fp8_block_scales = True
        use_cute_dsl_blockscaling_mm = True
        disable_deep_gemm = False
        in_features = flat_n
        weight = wo_b_fp8
        weight_scale = wo_b_scale

        def __call__(self, input):
            if isinstance(input, tuple):
                activation, scale = input
            else:
                activation, scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(input)
            return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                activation, self.weight, scale, self.weight_scale
            )

    output_linear = OutputLinear()
    args = dict(
        wo_a=None,
        wo_b=output_linear,
        n_groups=groups,
        o_lora_rank=rank,
        o_a_proj=o_a_fp8,
        o_a_scale=o_a_scale,
        o_a_dequant=None,
        use_fused_output_projection=True,
    )
    with autotune(skip_dynamic_tuning_buckets=True):
        actual = _output_projection(o, **args)
    cache_sizes = (
        len(CuteDSLFp8RubinBmmQuantRunner.kernel_cache),
        len(CuteDSLMXFP8RubinLinear.kernel_cache),
    )
    replay = _output_projection(o, **args)
    assert cache_sizes[0] > 0 and cache_sizes[1] > 0
    assert cache_sizes == (
        len(CuteDSLFp8RubinBmmQuantRunner.kernel_cache),
        len(CuteDSLMXFP8RubinLinear.kernel_cache),
    )
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)

    grouped_o = o.reshape(rows, groups, group_dim)
    grouped_o_fp8, grouped_o_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
        grouped_o
    )
    assert grouped_o_fp8.shape == (groups, rows, group_dim)
    o_lora = torch.empty(groups, rows, rank, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(
        grouped_o_fp8, o_a_fp8, grouped_o_scale, o_a_scale, o_lora
    )
    expected = output_linear(o_lora.transpose(0, 1).reshape(rows, flat_n))
    expected = expected.view(batch, block, hidden)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
