# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for the DSpark CuteDSL attention op."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.models.modeling_dspark import (
    _rope_last_dims_batched,
    dspark_attention_forward,
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
    page_size: int = 256,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    if start_pos_values is not None:
        batch = len(start_pos_values)
    heads, head_dim = 128, 512
    q = torch.randn(batch, block, heads, head_dim, device=device, dtype=torch.bfloat16)
    main_kv = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    block_kv = torch.randn(batch, block, head_dim, device=device, dtype=torch.bfloat16)

    if start_pos_values is None:
        start_pos_values = [199 * i + 1 for i in range(batch)]
    start_pos = torch.tensor(start_pos_values, device=device, dtype=torch.int64)
    width = (max(start_pos_values) + page_size - 1) // page_size
    required_pages = batch * width
    cache_pages = max(cache_pages or 0, required_pages + 1)
    cache_storage = torch.randn(
        cache_pages, 3, page_size, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = cache_storage[:, 1]
    assert not kv_cache.is_contiguous()
    # Physical pages differ from logical positions and from batch row indices.
    tables = torch.arange(required_pages, 0, -1, device=device, dtype=torch.int32)
    tables = tables.reshape(batch, width)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    # Per-request RoPE phases match the runtime indexing contract.
    table = precompute_dspark_freqs_cis(_ROPE_DIM, int(start_pos.max()) + block + 2, device=device)
    blk_freqs = table[start_pos.long().unsqueeze(1) + 1 + torch.arange(block, device=device)]
    inverse_rope_freqs = torch.view_as_real(blk_freqs).contiguous()
    return q, main_kv, block_kv, kv_cache, tables, start_pos, sink, blk_freqs, inverse_rope_freqs


def _valid_len(start_pos):
    return start_pos.clamp(max=128)


def _capacities(kv_cache, tables):
    return torch.full(
        (tables.shape[0],),
        tables.shape[1] * kv_cache.shape[1],
        dtype=torch.int64,
        device=kv_cache.device,
    )


def _prepare_attention_inputs(main_kv, block_kv, kv_cache, tables, start_pos):
    positions = start_pos - 1
    pages = tables.gather(1, (positions // kv_cache.shape[1]).unsqueeze(1)).squeeze(1)
    kv_cache[pages.long(), positions % kv_cache.shape[1]] = main_kv
    draft_block = block_kv.new_zeros((block_kv.shape[0], 8, block_kv.shape[2]))
    draft_block[:, : block_kv.shape[1]].copy_(block_kv)
    return draft_block, tables, start_pos


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
    positions = start_pos - 1
    pages = slots.gather(1, (positions // cache.shape[1]).unsqueeze(1)).squeeze(1)
    cache[pages.long(), positions % cache.shape[1]] = main_kv
    outputs = []
    for row in range(q.shape[0]):
        end = int(start_pos[row])
        length = min(128, end, int(valid_len[row]))
        logical = torch.arange(end - length, end, device=q.device)
        physical = slots[row, logical // cache.shape[1]].long()
        history = cache[physical, logical % cache.shape[1]]
        kv = torch.cat((history, block_kv[row])).float()
        scores = torch.einsum("mhd,nd->mhn", q[row].float(), kv) * softmax_scale
        scores = torch.cat((scores, sink[None, :, None].expand(q.shape[1], -1, 1)), dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
        outputs.append(torch.einsum("mhn,nd->mhd", probs, kv).to(q.dtype))
    output = torch.stack(outputs)
    return _rope_last_dims_batched(output, _ROPE_DIM, blk_freqs, inverse=True), cache


@pytest.mark.parametrize(
    ("invalid_case", "reason"),
    [
        ("q_dtype", "must use BF16"),
        ("block_size", "requires block 5/6"),
        ("valid_len_dtype", "contiguous INT64"),
        ("freqs_shape", "inverse RoPE frequencies"),
    ],
)
def test_fused_dsv4_dspark_attention_support_gate_logs_rejection(monkeypatch, invalid_case, reason):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    q, main, block, cache, tables, pos, sink, _, freqs = _make_inputs()
    draft, tables, pos = _prepare_attention_inputs(main, block, cache, tables, pos)
    inputs = [
        q,
        draft,
        cache,
        tables,
        pos,
        _valid_len(pos),
        _capacities(cache, tables),
        sink,
        freqs,
        512**-0.5,
    ]
    if invalid_case == "q_dtype":
        inputs[0] = q.float()
    elif invalid_case == "block_size":
        inputs[0] = q[:, :4].contiguous()
        inputs[8] = freqs[:, :4].contiguous()
    elif invalid_case == "valid_len_dtype":
        inputs[5] = inputs[5].int()
    else:
        inputs[8] = freqs[:, :, :-1].contiguous()
    monkeypatch.setattr(
        dspark_attention_op,
        "_compile_dspark_attention",
        lambda *args: pytest.fail("invalid inputs reached compilation"),
    )
    with pytest.raises(ValueError, match=reason):
        dspark_attention_op.fused_dsv4_dspark_attention(*inputs)


@pytest.mark.parametrize("invalid_case", ("draft_block_shape", "draft_block_layout", "index_dtype"))
def test_fused_dsv4_dspark_attention_rejects_invalid_inputs_before_launch(
    monkeypatch, invalid_case
):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs()
    valid_len = _valid_len(start_pos)
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
        "_compile_dspark_attention",
        lambda *args: pytest.fail("invalid inputs reached the kernel launch"),
    )
    with pytest.raises(ValueError):
        dspark_attention_op.fused_dsv4_dspark_attention(
            q,
            draft_block,
            kv_cache,
            slots_i32,
            cache_seqs,
            valid_len,
            _capacities(kv_cache, slots_i32),
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
        _make_inputs(
            29,
            block=block,
            start_pos_values=start_pos_values,
            page_size=(128 if block == 5 else 256)
            if valid_len_values[0] == 128
            else (16 if start_pos_values[0] == 5 else 32),
        )
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
        _capacities(kv_cache, slots_i32),
        sink,
        inverse_rope_freqs,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_fused_dsv4_dspark_attention_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rmsnorm_rope_draft_block,
        cute_dsl_dspark_rmsnorm_rope_page_write,
    )

    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, blk_freqs, inverse_rope_freqs = (
        _make_inputs(3, start_pos_values=[257, 5, 390], page_size=64)
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
    start_pos = start_pos.long()
    valid_len = _valid_len(start_pos)

    capacities = _capacities(kv_cache, slots)

    def run():
        cute_dsl_dspark_rmsnorm_rope_page_write(
            main_x, weight, main_freqs, kv_cache, slots, start_pos, capacities, 1e-6
        )
        draft = cute_dsl_dspark_rmsnorm_rope_draft_block(block_x, weight, block_freqs, 1e-6)
        output = fused_dsv4_dspark_attention(
            q,
            draft,
            kv_cache,
            slots,
            start_pos,
            valid_len,
            capacities,
            sink,
            inverse_rope_freqs,
            scale,
        )
        return output, draft

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, draft_block = run()

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


@pytest.mark.parametrize("page_size", (128, 256))
def test_dspark_attention_forward_batched_matches_fallback(monkeypatch, page_size):
    import tensorrt_llm._torch.models.modeling_dspark as model

    g = _make_attn_inputs(seed=17, device="cuda", page_size=page_size)
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        g[name].mul_(0.2)
    for name in ("x", "main_x", "kv_cache0"):
        g[name].mul_(0.1)
    before = g["kv_cache0"].clone()
    actual = _run(g)
    actual_cache = g["kv_cache0"].clone()
    g["kv_cache0"].copy_(before)

    def reference_attention(
        q, draft, pages, tables, positions, lengths, capacities, sink, freqs, scale
    ):
        main_positions = positions - 1
        main_pages = tables.gather(1, (main_positions // page_size).unsqueeze(1)).squeeze(1)
        main = pages[main_pages.long(), main_positions % page_size]
        output, _ = _reference(
            q,
            main,
            draft[:, : q.shape[1]],
            pages,
            tables,
            positions,
            lengths,
            sink,
            torch.view_as_complex(freqs),
            scale,
        )
        return output

    with monkeypatch.context() as patch:
        patch.setattr(torch.ops.trtllm, "fused_dsv4_dspark_attention", reference_attention)
        patch.setattr(model, "is_fused_dspark_rmsnorm_rope_supported", lambda *a: False)
        expected = _run(g)
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(actual_cache, g["kv_cache0"], rtol=2e-2, atol=2e-2)


def test_warmup_propagates_compile_failure(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as op

    op._dspark_attention_kernel_cache.clear()
    monkeypatch.setattr(
        op, "_compile_dspark_attention", Mock(side_effect=RuntimeError("synthetic compile failure"))
    )
    q, main, block, cache, tables, pos, sink, _, freqs = _make_inputs()
    draft, tables, pos = _prepare_attention_inputs(main, block, cache, tables, pos)
    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        op.fused_dsv4_dspark_attention(
            q,
            draft,
            cache,
            tables,
            pos,
            _valid_len(pos),
            _capacities(cache, tables),
            sink,
            freqs,
            512**-0.5,
        )


def test_self_jit_reuses_one_kernel_across_runtime_shapes_and_scales():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _dspark_attention_kernel_cache,
        _get_dspark_arch_str,
        fused_dsv4_dspark_attention,
    )

    assert [_get_dspark_arch_str(sm) for sm in (100, 103)] == [
        "sm_100",
        "sm_103",
    ]
    assert _get_dspark_arch_str(101) is None
    assert _get_dspark_arch_str(109) is None
    assert _get_dspark_arch_str() in ("sm_100", "sm_103")
    _dspark_attention_kernel_cache.clear()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = _make_inputs(
        11, start_pos_values=[300, 4, 250], cache_pages=40
    )
    scale = q.shape[-1] ** -0.5
    valid_len = _valid_len(start_pos)

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
        _capacities(kv_cache, slots_i32),
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
        valid_len_b = _valid_len(pos_b)
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
            _capacities(cache_b, slots_i32_b),
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
        compiled = dspark_attention_op._compile_dspark_attention(5, arch_str, 256)
    dspark_attention_op._dspark_attention_kernel_cache[(5, 256, arch_str)] = compiled

    args = _make_inputs(41, block=5, start_pos_values=[257, 9])
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink, _, freqs = args
    draft_block, slots_i32, cache_seqs = _prepare_attention_inputs(
        main_kv, block_kv, kv_cache, slots, start_pos
    )
    valid_len = _valid_len(start_pos)
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

    output = dspark_attention_op.fused_dsv4_dspark_attention(
        q,
        draft_block,
        kv_cache,
        slots_i32,
        cache_seqs,
        valid_len,
        _capacities(kv_cache, slots_i32),
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

    with pytest.raises(RuntimeError, match="must run eagerly before graph capture"):
        dspark_attention_op.fused_dsv4_dspark_attention(
            q,
            draft_block,
            kv_cache,
            slots_i32,
            cache_seqs,
            _valid_len(start_pos),
            _capacities(kv_cache, slots_i32),
            sink,
            freqs,
            q.shape[-1] ** -0.5,
        )


def test_preparation_cache_misses_reject_cuda_graph_capture(monkeypatch):
    import tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op as preparation_op

    preparation_op._compile_dspark_rmsnorm_rope_page_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        preparation_op.cute,
        "compile",
        lambda *args: pytest.fail("preparation compiler was called during CUDA graph capture"),
    )

    with pytest.raises(RuntimeError, match="page-write must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_page_write(256, 1e-6, 0)
    with pytest.raises(RuntimeError, match="draft-block must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_draft_block(5, 1e-6, 0)


def test_preparation_self_jit_covers_dynamic_batches():
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_dspark_rmsnorm_rope_draft_block,
        _compile_dspark_rmsnorm_rope_page_write,
        cute_dsl_dspark_rmsnorm_rope_draft_block,
        cute_dsl_dspark_rmsnorm_rope_page_write,
    )

    _compile_dspark_rmsnorm_rope_page_write.cache_clear()
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
        before = cache_b.clone()
        cute_dsl_dspark_rmsnorm_rope_page_write(
            main_b.unsqueeze(1),
            weight,
            main_freqs_b,
            cache_b,
            slots_b,
            pos_b,
            _capacities(cache_b, slots_b),
            1e-6,
        )
        draft_block = cute_dsl_dspark_rmsnorm_rope_draft_block(block_b, weight, block_freqs_b, 1e-6)
        expected_main = main_b.float() * torch.rsqrt(
            main_b.float().square().mean(-1, keepdim=True) + 1e-6
        )
        pos = pos_b - 1
        pages = slots_b.gather(1, (pos // cache_b.shape[1]).unsqueeze(1)).squeeze(1)
        before[pages.long(), pos % cache_b.shape[1]] = expected_main.to(cache_b.dtype)
        torch.testing.assert_close(cache_b, before, rtol=2e-2, atol=2e-2)
        assert draft_block.shape == (batch, 8, 512)
        assert torch.count_nonzero(draft_block[:, q_b.shape[1] :]) == 0

    assert _compile_dspark_rmsnorm_rope_page_write.cache_info().misses == 1
    assert _compile_dspark_rmsnorm_rope_draft_block.cache_info().misses == 1


def _attention_case(seed=0, start_positions=(40, 40), lengths=None, sink=None):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
    )

    q, main, block, pages, tables, positions, default_sink, _, _ = _make_inputs(
        seed,
        block=5,
        start_pos_values=start_positions,
    )
    lengths = torch.tensor(lengths or [min(128, p) for p in start_positions], device="cuda")
    sink = default_sink if sink is None else torch.full_like(default_sink, sink)
    draft, _, _ = _prepare_attention_inputs(main, block, pages, tables, positions)
    freqs = torch.zeros(q.shape[0], q.shape[1], 32, 2, device="cuda")
    freqs[..., 0] = 1
    output = fused_dsv4_dspark_attention(
        q,
        draft,
        pages,
        tables,
        positions,
        lengths,
        _capacities(pages, tables),
        sink,
        freqs,
        512**-0.5,
    )
    histories = []
    for row, (end, length) in enumerate(zip(start_positions, lengths.tolist())):
        logical = torch.arange(end - length, end, device="cuda")
        history = pages[tables[row, logical // pages.shape[1]].long(), logical % pages.shape[1]]
        histories.append(torch.cat((history, block[row])))
    return output, q, histories, sink


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_attn_matches_loop_reference(seed):
    output, q, histories, sink = _attention_case(seed)
    expected = torch.empty_like(output)
    for row, kv in enumerate(histories):
        for token in range(q.shape[1]):
            scores = q[row, token].float() @ kv.float().T * 512**-0.5
            weights = torch.softmax(torch.cat((scores, sink[:, None]), dim=-1), dim=-1)
            expected[row, token] = (weights[:, :-1] @ kv.float()).to(output.dtype)
    torch.testing.assert_close(output, expected, rtol=5e-2, atol=3e-2)


def test_sparse_attn_no_sink_matches_sdpa():
    output, q, histories, _ = _attention_case(sink=float("-inf"))
    kv = torch.stack(histories).unsqueeze(1).expand(-1, 128, -1, -1)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), kv, kv).transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=5e-2, atol=3e-2)


def test_sparse_attn_sink_reduces_mass():
    no_sink, *_ = _attention_case(start_positions=(4,), sink=float("-inf"))
    with_sink, *_ = _attention_case(start_positions=(4,), sink=0.0)
    assert with_sink.abs().sum() < no_sink.abs().sum()


def test_sparse_attn_masked_indices_excluded():
    masked, q, histories, _ = _attention_case(start_positions=(5,), lengths=[3], sink=float("-inf"))
    kv = histories[0][None, None].expand(1, 128, -1, -1)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), kv, kv).transpose(1, 2)
    torch.testing.assert_close(masked, expected, rtol=5e-2, atol=3e-2)
    full, *_ = _attention_case(start_positions=(5,), lengths=[4], sink=float("-inf"))
    assert not torch.allclose(full, masked, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "start_pos,page_size,block", [(1, 128, 5), (3, 64, 5), (10, 16, 5), (200, 256, 6)]
)
def test_page_mapping_matches_reference(start_pos, page_size, block):
    from tensorrt_llm._torch.attention.kernels.dspark import write_dspark_context

    q, values, _, pages, tables, positions, *_ = _make_inputs(
        block=block,
        start_pos_values=[start_pos] * 3,
        page_size=page_size,
    )
    before = pages.clone()
    write_dspark_context(
        values[:, None],
        positions[:, None],
        torch.ones(3, 1, device=q.device, dtype=torch.bool),
        pages,
        tables,
        _capacities(pages, tables),
    )
    for row in range(3):
        logical = start_pos - 1
        before[int(tables[row, logical // page_size]), logical % page_size] = values[row]
    torch.testing.assert_close(pages, before, rtol=0, atol=0)


def test_page_mapping_masks_non_generation_rows():
    from tensorrt_llm._torch.attention.kernels.dspark import write_dspark_context

    values = torch.randn(1, 1, 16, device="cuda")
    pool = torch.randn(1, 128, 16, device="cuda")
    before = pool.clone()
    write_dspark_context(
        values,
        torch.zeros(1, 1, device="cuda", dtype=torch.long),
        torch.ones(1, 1, device="cuda", dtype=torch.bool),
        pool,
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.long),
    )
    torch.testing.assert_close(pool, before, rtol=0, atol=0)


def _make_attn_inputs(seed=0, device="cuda", page_size=256):
    """Small synthetic DSpark attention inputs/weights (CPU bf16)."""
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 128, 512, 64
    q_lora, o_lora, n_groups = 64, 8, 8
    window, block, start_pos = 128, 5, 200
    b = 2
    g = dict(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rd,
        q_lora=q_lora,
        o_lora=o_lora,
        n_groups=n_groups,
        window=window,
        block=block,
        start_pos=start_pos,
        b=b,
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
    )
    bf = torch.bfloat16
    g["x"] = torch.randn(b, block, dim, dtype=bf)
    g["main_x"] = torch.randn(b, 1, dim, dtype=bf)
    width = (start_pos + page_size - 1) // page_size
    g["kv_cache0"] = torch.randn(b * width + 1, page_size, head_dim, dtype=bf)
    g["tables"] = torch.arange(b * width, 0, -1, dtype=torch.int32).reshape(b, width)
    g["capacities"] = torch.full((b,), width * page_size, dtype=torch.long)
    g["valid_len"] = torch.full((b,), min(start_pos, window), dtype=torch.long)
    g["wq_a"] = torch.randn(q_lora, dim, dtype=bf) * 0.1
    g["wq_b"] = torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1
    g["wkv"] = torch.randn(head_dim, dim, dtype=bf) * 0.1
    g["wo_a"] = torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1
    g["wo_b"] = torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1
    g["q_norm"] = torch.ones(q_lora, dtype=bf)
    g["kv_norm"] = torch.ones(head_dim, dtype=bf)
    g["attn_sink"] = torch.randn(n_heads)
    g["freqs"] = precompute_dspark_freqs_cis(rd, start_pos + 1 + block + 2)
    g["start_pos"] = torch.full((b,), start_pos, dtype=torch.long)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}


def _run(g):
    return dspark_attention_forward(
        g["x"],
        g["main_x"],
        g["start_pos"],
        g["kv_cache0"],
        g["tables"],
        g["capacities"],
        g["valid_len"],
        wq_a=g["wq_a"],
        q_norm_w=g["q_norm"],
        wq_b=g["wq_b"],
        wkv=g["wkv"],
        kv_norm_w=g["kv_norm"],
        wo_a=g["wo_a"],
        wo_b=g["wo_b"],
        attn_sink=g["attn_sink"],
        n_heads=g["n_heads"],
        head_dim=g["head_dim"],
        rope_head_dim=g["rope_head_dim"],
        n_groups=g["n_groups"],
        o_lora_rank=g["o_lora"],
        window_size=g["window"],
        eps=g["eps"],
        softmax_scale=g["softmax_scale"],
        freqs_cis=g["freqs"],
    )


def test_attention_forward_shape_and_determinism():
    g = _make_attn_inputs()
    o = _run(g)
    assert tuple(o.shape) == (g["b"], g["block"], g["dim"])
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o, _run(g))  # deterministic


def test_attention_forward_updates_only_mapped_kv():
    g = _make_attn_inputs()
    before = g["kv_cache0"].clone()
    _run(g)
    changed = (g["kv_cache0"] != before).any(dim=-1)
    expected = torch.zeros_like(changed)
    for row, position in enumerate(g["start_pos"].tolist()):
        logical = position - 1
        page_size = before.shape[1]
        expected[int(g["tables"][row, logical // page_size]), logical % page_size] = True
    assert torch.equal(changed, expected)


def _make_batched_inputs(seed=0, start_positions=(1, 3, 20)):
    g = _make_attn_inputs(seed)
    batch = len(start_positions)
    page_size = g["kv_cache0"].shape[1]
    width = (max(start_positions) + page_size - 1) // page_size
    g["x"] = g["x"][:1].repeat(batch, 1, 1)
    g["main_x"] = g["main_x"][:1].repeat(batch, 1, 1)
    g["kv_cache0"] = torch.randn(
        batch * width + 1, page_size, g["head_dim"], device="cuda", dtype=torch.bfloat16
    )
    g["tables"] = torch.arange(batch * width, 0, -1, device="cuda", dtype=torch.int32).reshape(
        batch, width
    )
    g["start_pos"] = torch.tensor(start_positions, device="cuda", dtype=torch.long)
    g["capacities"] = torch.full((batch,), width * page_size, device="cuda", dtype=torch.long)
    g["valid_len"] = g["start_pos"].clamp(max=128)
    g["b"] = batch
    return g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_attention_matches_scalar_per_request(seed):
    g = _make_batched_inputs(seed)
    before = g["kv_cache0"].clone()
    actual = _run(g)
    outputs = []
    for row in range(g["b"]):
        single = dict(g)
        for key in ("x", "main_x", "tables", "start_pos", "capacities", "valid_len"):
            single[key] = g[key][row : row + 1].contiguous()
        single["kv_cache0"] = before.clone()
        outputs.append(_run(single))
    torch.testing.assert_close(actual, torch.cat(outputs), rtol=2e-2, atol=2e-2)


def test_batched_attention_writes_through_pages():
    g = _make_batched_inputs(seed=3)
    before = g["kv_cache0"].clone()
    _run(g)
    changed = (g["kv_cache0"] != before).any(dim=-1)
    expected = torch.zeros_like(changed)
    for row, pos in enumerate(g["start_pos"].tolist()):
        page_size = before.shape[1]
        expected[int(g["tables"][row, (pos - 1) // page_size]), (pos - 1) % page_size] = True
    assert torch.equal(changed, expected)


def test_batched_attention_dummy_rows_keep_pages():
    g = _make_batched_inputs(seed=4)
    g["capacities"].zero_()
    before = g["kv_cache0"].clone()
    _run(g)
    torch.testing.assert_close(g["kv_cache0"], before)


@pytest.mark.parametrize("start_positions", [(1, 3, 20), (5, 5, 5), (2, 7, 200)])
def test_batched_page_writes_match_per_request(start_positions):
    from tensorrt_llm._torch.attention.kernels.dspark import write_dspark_context

    g = _make_batched_inputs(start_positions=start_positions)
    values = torch.randn(g["b"], 1, g["head_dim"], device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(g["b"], 1, device="cuda", dtype=torch.bool)
    expected = g["kv_cache0"].clone()
    for row in range(g["b"]):
        write_dspark_context(
            values[row : row + 1],
            g["start_pos"][row : row + 1, None],
            mask[row : row + 1],
            expected,
            g["tables"][row : row + 1],
            g["capacities"][row : row + 1],
        )
    write_dspark_context(
        values, g["start_pos"][:, None], mask, g["kv_cache0"], g["tables"], g["capacities"]
    )
    torch.testing.assert_close(g["kv_cache0"], expected, rtol=0, atol=0)


def test_batched_attention_respects_partial_window_valid_len():
    actual, q, histories, sink = _attention_case(
        start_positions=(3, 10, 200), lengths=[0, 3, 128], sink=0.0
    )
    for row, kv in enumerate(histories):
        scores = torch.einsum("mhd,nd->mhn", q[row].float(), kv.float()) * 512**-0.5
        scores = torch.cat((scores, sink[None, :, None].expand(q.shape[1], -1, 1)), dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
        expected = torch.einsum("mhn,nd->mhd", probs, kv.float()).to(q.dtype)
        torch.testing.assert_close(actual[row], expected, rtol=5e-2, atol=3e-2)


def test_batched_attention_cuda_graph_capture_replay():
    g = _make_batched_inputs(seed=0)
    eager = _run(g)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(g)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _run(g)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, eager, rtol=2e-2, atol=2e-2)
