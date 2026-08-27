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


def _legacy_valid_len(start_pos):
    return (start_pos.long() + 1).clamp(max=128)


def _prepare_attention_inputs(main_kv, block_kv, kv_cache, slots, start_pos):
    kv_cache[slots.long(), (start_pos % kv_cache.shape[1]).long()] = main_kv
    draft_block = block_kv.new_zeros((block_kv.shape[0], 8, block_kv.shape[2]))
    draft_block[:, : block_kv.shape[1]].copy_(block_kv)
    return draft_block, slots.to(torch.int32), start_pos.to(torch.int32)


def _with_row_padding(x: torch.Tensor, padding: int = 128) -> torch.Tensor:
    storage = torch.empty((*x.shape[:-1], x.shape[-1] + padding), device=x.device, dtype=x.dtype)
    padded = storage[..., : x.shape[-1]]
    padded.copy_(x)
    return padded


def _reference(
    q,
    main_kv,
    block_kv,
    kv_cache,
    slots,
    start_pos,
    valid_len,
    sink,
    blk_freqs,
    softmax_scale,
):
    cache = kv_cache.clone()
    window = cache.shape[1]
    cache[slots.long(), (start_pos % window).long()] = main_kv
    kv_full = torch.cat([cache[slots.long()], block_kv], dim=1)
    topk = get_dspark_topk_idxs_batched(window, q.shape[1], start_pos.long(), valid_len)
    o = dspark_sparse_attn(q, kv_full, sink, topk, softmax_scale)
    return _rope_last_dims_batched(o, _ROPE_DIM, blk_freqs, inverse=True), cache


@pytest.mark.parametrize(
    ("invalid_case", "reason"),
    [
        ("q_dtype", "q dtype must be BF16"),
        ("block_size", "draft block size must be 5 or 6"),
        ("valid_len_dtype", "valid_len must use INT64"),
        ("freqs_shape", "inverse_rope_freqs shape must be"),
    ],
)
def test_fused_dsv4_dspark_attention_support_gate_logs_rejection(monkeypatch, invalid_case, reason):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    q, _, _, kv_cache, _, start_pos, sink, _, freqs = _make_inputs()
    inputs = [q, kv_cache, _legacy_valid_len(start_pos), sink, freqs]
    if invalid_case == "q_dtype":
        inputs[0] = q.float()
    elif invalid_case == "block_size":
        inputs[0] = q[:, :4].contiguous()
        inputs[4] = freqs[:, :4].contiguous()
    elif invalid_case == "valid_len_dtype":
        inputs[2] = inputs[2].to(torch.int32)
    else:
        inputs[4] = freqs[:, :, :-1].contiguous()

    messages = []
    monkeypatch.setattr(
        dspark_attention_op.logger,
        "debug_once",
        lambda *message, key: messages.append((" ".join(map(str, message)), key)),
    )

    assert not dspark_attention_op.is_fused_dsv4_dspark_attention_supported(*inputs)
    assert len(messages) == 1
    assert reason in messages[0][0]
    assert messages[0][1][0] == "fused_dsv4_dspark_attention_unsupported"


@pytest.mark.parametrize("invalid_case", ("draft_block_shape", "draft_block_layout", "index_dtype"))
def test_fused_dsv4_dspark_attention_rejects_invalid_inputs_before_launch(
    monkeypatch, invalid_case
):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _legacy_valid_len(start_pos)
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )
    if invalid_case == "draft_block_shape":
        draft_block = draft_block[:, :-1].contiguous()
    elif invalid_case == "draft_block_layout":
        draft_block = _with_row_padding(draft_block)
        assert not draft_block.is_contiguous()
    else:
        slots_i32 = slots_i32.long()

    monkeypatch.setattr(
        dspark_attention_op,
        "_run_dspark_attention",
        lambda *args: pytest.fail("invalid inputs reached the kernel launch"),
    )
    with pytest.raises(ValueError, match="requires contiguous supported DSV4 DSpark tensors"):
        dspark_attention_op.fused_dsv4_dspark_attention(
            q,
            draft_block,
            kv_cache,
            slots_i32,
            cache_seqs,
            valid_len,
            sink,
            freqs,
            q.shape[-1] ** -0.5,
        )


@pytest.mark.parametrize("block", (5, 6))
@pytest.mark.parametrize(
    ("start_pos_values", "valid_len_values"),
    [
        # Full windows with slots != start_pos: catches the window-validity
        # mask being fed anything but the absolute decode position.
        ([257, 390], [128, 128]),
        # Partially filled windows: only rows 0..start_pos are attended.
        ([5, 100], [6, 101]),
        # Bootstrapped positions with short physical suffixes exercise wraparound.
        ([257, 390], [3, 5]),
    ],
    ids=["full_window", "partial_window", "wrapped_suffix"],
)
def test_fused_dsv4_dspark_attention_matches_reference(block, start_pos_values, valid_len_values):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs = (
        _make_inputs(29, block=block, start_pos_values=start_pos_values)
    )
    valid_len = torch.tensor(valid_len_values, device=q.device, dtype=torch.long)
    expected, expected_cache = _reference(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        blk_freqs,
        q.shape[-1] ** -0.5,
    )
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )

    actual = fused_dsv4_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        sink,
        inverse_rope_freqs,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_fused_dsv4_dspark_attention_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
        warmup_fused_dsv4_dspark_attention,
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

    # The named best-effort prewarm calls the same self-JIT ops used below,
    # so graph capture sees only hot compile-cache entries.
    warmup_fused_dsv4_dspark_attention(block, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots_i32, cache_seqs = cute_dsl_dspark_rmsnorm_rope_cache_write(
            main_x, weight, main_freqs, kv_cache, slots, start_pos, 1e-6
        )
        draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(block_x, weight, block_freqs, 1e-6)
        captured = fused_dsv4_dspark_attention(
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
        scale,
    )
    graph.replay()

    torch.testing.assert_close(captured, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(draft_block[:, :block], expected_block, rtol=0, atol=0)
    torch.testing.assert_close(
        draft_block[:, block:], torch.zeros_like(draft_block[:, block:]), rtol=0, atol=0
    )


@pytest.mark.parametrize("persist", (False, True))
def test_dspark_attention_forward_batched_matches_fallback(monkeypatch, persist):
    import tensorrt_llm._torch.models.dspark.attention as dspark_attention

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
    calls = {
        "fused_attention": 0,
        "cache_write": 0,
        "draft_block": 0,
        "rmsnorm_rope": 0,
    }
    op_attention = dspark_attention.fused_dsv4_dspark_attention
    op_cache_write = dspark_attention.cute_dsl_dspark_rmsnorm_rope_cache_write
    op_draft_block = dspark_attention.cute_dsl_dspark_rmsnorm_rope_draft_block
    op_rmsnorm_rope = dspark_attention.cute_dsl_dspark_rmsnorm_rope

    def counted_attention(*args):
        calls["fused_attention"] += 1
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
        patch.setattr(dspark_attention, "fused_dsv4_dspark_attention", counted_attention)
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
        "fused_attention": 1,
        "cache_write": 1,
        "draft_block": 1,
        "rmsnorm_rope": 2,
    }

    with monkeypatch.context() as patch:
        patch.setattr(
            dspark_attention,
            "is_fused_dsv4_dspark_attention_supported",
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


def test_warmup_is_best_effort(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    warnings = []
    monkeypatch.setattr(dspark_attention_op, "_get_dspark_arch_str", lambda: "sm_100")
    monkeypatch.setattr(
        dspark_attention_op,
        "cute_dsl_dspark_rmsnorm_rope_cache_write",
        lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic compile failure")),
    )
    monkeypatch.setattr(dspark_attention_op.logger, "warning", warnings.append)

    dspark_attention_op.warmup_fused_dsv4_dspark_attention(5, 1e-6)

    assert len(warnings) == 1
    assert "self-JIT on first use" in warnings[0]
    assert "synthetic compile failure" in warnings[0]


def test_self_jit_reuses_one_kernel_across_runtime_shapes_and_scales():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _dspark_attention_kernel_cache,
        _get_dspark_arch_str,
        fused_dsv4_dspark_attention,
        is_dsv4_dspark_attention_config_supported,
    )

    assert [_get_dspark_arch_str(sm) for sm in (100, 103)] == [
        "sm_100",
        "sm_103",
    ]
    assert _get_dspark_arch_str(101) is None
    assert _get_dspark_arch_str(109) is None
    assert _get_dspark_arch_str() in ("sm_100", "sm_103")
    assert is_dsv4_dspark_attention_config_supported(5, 128, 512, 128)
    assert not is_dsv4_dspark_attention_config_supported(4, 128, 512, 128)
    assert not is_dsv4_dspark_attention_config_supported(5, 24, 512, 128)

    _dspark_attention_kernel_cache.clear()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs(
        11, start_pos_values=[300, 4, 250], cache_pages=40
    )
    scale = q.shape[-1] ** -0.5
    valid_len = _legacy_valid_len(start_pos)

    expected, expected_cache = _reference(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        valid_len,
        sink,
        torch.view_as_complex(freqs),
        scale,
    )
    # A missing key self-JITs through the production op.
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )
    actual = fused_dsv4_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        sink,
        freqs,
        scale,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)
    assert len(_dspark_attention_kernel_cache) == 1

    # Every runtime batch, page count, page stride, and softmax scale is a hot
    # hit on the same compiled object without compiler calls or runtime padding.
    for batch in (1, 3, 32):
        values = [5 + (37 * i) % 386 for i in range(batch)]
        args = _make_inputs(17 + batch, start_pos_values=values, cache_pages=batch + 41)
        q_b, main_b, block_b, cache_b, slots_b, pos_b, sink_b, blk_freqs_b, freqs_b = args
        if batch == 3:
            cache_b = cache_b.clone()
            assert cache_b.is_contiguous()
        runtime_scale = scale if batch % 2 else scale * 0.5
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
            runtime_scale,
        )
        draft_block_b, slots_i32_b, cache_seqs_b = _prepare_attention_inputs(
            main_b, block_b, cache_b, slots_b, pos_b
        )
        actual = fused_dsv4_dspark_attention(
            q_b,
            draft_block_b,
            cache_b,
            slots_i32_b,
            cache_seqs_b,
            valid_len_b,
            sink_b,
            freqs_b,
            runtime_scale,
        )
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
        torch.testing.assert_close(cache_b, expected_cache, rtol=0, atol=0)
    assert len(_dspark_attention_kernel_cache) == 1


def test_compile_without_real_specimens_and_cached_wrapper_avoids_views(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    dspark_attention_op._dspark_attention_kernel_cache.clear()
    arch_str = dspark_attention_op._get_dspark_arch_str()
    assert arch_str is not None

    def unexpected_compile_op(*args, **kwargs):
        del args, kwargs
        pytest.fail("DSpark compilation used a real tensor specimen")

    # Compile before runtime inputs exist. Compile-only pointers and the fake
    # output anchor must not allocate or wrap a PyTorch tensor.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", unexpected_compile_op)
        patch.setattr(torch, "empty_like", unexpected_compile_op)
        patch.setattr(dspark_attention_op.cute.runtime, "from_dlpack", unexpected_compile_op)
        compiled = dspark_attention_op._compile_dspark_attention(5, arch_str)
    dspark_attention_op._dspark_attention_kernel_cache[(5, arch_str)] = compiled

    args = _make_inputs(41, block=5, start_pos_values=[257, 9])
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = args
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )
    valid_len = _legacy_valid_len(start_pos)
    scale = q.shape[-1] ** -0.5

    def unexpected_host_op(*args, **kwargs):
        del args, kwargs
        pytest.fail("cached DSpark host wrapper used a Python tensor conversion or view")

    monkeypatch.setattr(dspark_attention_op.cute.runtime, "from_dlpack", unexpected_host_op)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_host_op)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected_host_op)
    monkeypatch.setattr(torch.Tensor, "permute", unexpected_host_op)
    monkeypatch.setattr(torch.Tensor, "unsqueeze", unexpected_host_op)
    monkeypatch.setattr(torch.Tensor, "reshape", unexpected_host_op)

    output = dspark_attention_op._run_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        sink,
        freqs,
        scale,
    )
    assert output.shape == q.shape


def test_attention_cache_miss_rejects_cuda_graph_capture(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    dspark_attention_op._dspark_attention_kernel_cache.clear()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs(43, block=5)
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        dspark_attention_op,
        "_compile_dspark_attention",
        lambda *args: pytest.fail("compiler was called during CUDA graph capture"),
    )
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda *args, **kwargs: pytest.fail("output allocated before the capture guard"),
    )

    with pytest.raises(RuntimeError, match="must be warmed up before CUDA graph capture"):
        dspark_attention_op._run_dspark_attention(
            q,
            draft_block,
            kv_cache,
            slots_i32,
            cache_seqs,
            _legacy_valid_len(start_pos),
            sink,
            freqs,
            q.shape[-1] ** -0.5,
        )


def test_preparation_cache_misses_reject_cuda_graph_capture(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op as preparation_op

    preparation_op._compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        preparation_op.cute,
        "compile",
        lambda *args: pytest.fail("preparation compiler was called during CUDA graph capture"),
    )

    with pytest.raises(RuntimeError, match="cache-write must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_cache_write(1e-6)
    with pytest.raises(RuntimeError, match="draft-block must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_draft_block(5, 1e-6)


def test_preparation_self_jit_covers_dynamic_batches():
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

    for batch in (1, 3, 32):
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
