# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for the DSpark CuteDSL attention op."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.custom_ops import dspark_attention_custom_op as attention_op
from tensorrt_llm._torch.custom_ops import dspark_rmsnorm_rope_custom_op as preparation_op
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.models import modeling_dspark as dspark
from tensorrt_llm._utils import get_sm_version

_ROPE_DIM = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="DSpark CuteDSL attention requires an SM100, SM103 or SM107 CUDA GPU",
)


def _make_inputs(
    seed=0,
    batch=2,
    block=6,
    heads=128,
    page_size=32,
    positions=None,
    valid_lengths=None,
    capacities=None,
    extra_pages=0,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    if positions is None:
        positions = [1 + (37 * i) % 386 for i in range(batch)]
    batch = len(positions)
    positions = torch.tensor(positions, device=device, dtype=torch.int64)
    lengths = (
        positions.clamp(max=128)
        if valid_lengths is None
        else torch.tensor(valid_lengths, device=device, dtype=torch.int64)
    )
    capacities = torch.tensor(
        [512] * batch if capacities is None else capacities, device=device, dtype=torch.int64
    )
    pages_per_request = 512 // page_size
    num_pages = batch * pages_per_request + extra_pages
    # Both the pool and table rows have padding, as in managed cache views.
    storage = torch.randn(num_pages, 3, page_size, 512, device=device, dtype=torch.bfloat16)
    table_storage = torch.full((batch, pages_per_request + 1), -1, device=device, dtype=torch.int32)
    tables = table_storage[:, :pages_per_request]
    tables.copy_(
        torch.randperm(num_pages, device=device, dtype=torch.int32)[
            : batch * pages_per_request
        ].reshape(batch, pages_per_request)
    )
    tables[capacities == 0] = -1
    freqs = dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device=device)
    block_freqs = freqs[positions[:, None] + 1 + torch.arange(block, device=device)]
    return dict(
        q=torch.randn(batch, block, heads, 512, device=device, dtype=torch.bfloat16),
        draft_block=torch.randn(batch, 8, 512, device=device, dtype=torch.bfloat16),
        kv_pages=storage[:, 1],
        block_tables=tables,
        positions=positions,
        valid_lengths=lengths,
        capacities=capacities,
        attn_sink=torch.randn(heads, device=device),
        inverse_rope_freqs=torch.view_as_real(block_freqs).contiguous(),
        softmax_scale=512**-0.5,
    )


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
    draft_block,
    kv_pages,
    block_tables,
    positions,
    valid_lengths,
    capacities,
    attn_sink,
    inverse_rope_freqs,
    softmax_scale,
):
    """Gather absolute context positions and compute sink attention in FP32."""
    tables = block_tables.cpu()
    outputs = []
    for row, (end, length, capacity) in enumerate(
        zip(positions.tolist(), valid_lengths.tolist(), capacities.tolist())
    ):
        tokens = list(range(max(0, end - min(128, length)), min(end, capacity)))
        if tokens:
            pages = [int(tables[row, token // kv_pages.shape[1]]) for token in tokens]
            offsets = [token % kv_pages.shape[1] for token in tokens]
            context = kv_pages[pages, offsets]
            kv = torch.cat((context, draft_block[row, : q.shape[1]]))
        else:
            kv = draft_block[row, : q.shape[1]]
        scores = torch.einsum("shd,td->sht", q[row].float(), kv.float()) * softmax_scale
        sink = attn_sink[None, :, None].expand(q.shape[1], -1, 1)
        probs = torch.cat((scores, sink), dim=-1).softmax(-1)[..., :-1]
        outputs.append(torch.einsum("sht,td->shd", probs, kv.float()).to(q.dtype))
    return dspark._rope_last_dims_batched(
        torch.stack(outputs),
        _ROPE_DIM,
        torch.view_as_complex(inverse_rope_freqs),
        inverse=True,
    )


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

    paged_inputs = _make_inputs()
    q = paged_inputs["q"]
    kv_cache = torch.empty(q.shape[0], 128, 512, dtype=q.dtype, device=q.device)
    start_pos = paged_inputs["positions"]
    sink = paged_inputs["attn_sink"]
    freqs = paged_inputs["inverse_rope_freqs"]
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


@pytest.mark.parametrize(
    "invalid_case,reason",
    [
        ("q_dtype", "must use BF16"),
        ("block_size", "block size must be one of"),
        ("heads", r"requires \(64, 128\) heads"),
        ("page_size", "page size must be one of"),
        ("page_stride", "positive 16-byte-aligned page stride"),
        ("draft_shape", "draft KV must have shape"),
        ("draft_layout", "must be contiguous"),
        ("table_dtype", "contiguous INT32 rows"),
        ("table_layout", "contiguous INT32 rows"),
        ("positions", "positions must be a contiguous INT64"),
        ("valid_lengths", "valid lengths must be a contiguous INT64"),
        ("capacities", "capacities must be a contiguous INT64"),
        ("sink_dtype", "sink must be FP32"),
        ("freqs_shape", "inverse RoPE frequencies must be FP32"),
    ],
)
def test_fused_dsv4_dspark_attention_rejects_invalid_inputs_before_launch(
    monkeypatch, invalid_case, reason
):
    inputs = _make_inputs()
    if invalid_case == "q_dtype":
        inputs["q"] = inputs["q"].float()
    elif invalid_case == "block_size":
        inputs["q"] = inputs["q"][:, :4].contiguous()
    elif invalid_case == "heads":
        inputs["q"] = inputs["q"][:, :, :24].contiguous()
    elif invalid_case == "page_size":
        inputs["kv_pages"] = inputs["kv_pages"][:, :24]
    elif invalid_case == "page_stride":
        inputs["kv_pages"] = inputs["kv_pages"][:1].expand_as(inputs["kv_pages"])
    elif invalid_case == "draft_shape":
        inputs["draft_block"] = inputs["draft_block"][:, :7].contiguous()
    elif invalid_case == "draft_layout":
        inputs["draft_block"] = F.pad(inputs["draft_block"], (0, 128))[..., :512]
    elif invalid_case == "table_dtype":
        inputs["block_tables"] = inputs["block_tables"].long()
    elif invalid_case == "table_layout":
        inputs["block_tables"] = inputs["block_tables"][:, ::2]
    elif invalid_case in ("positions", "valid_lengths", "capacities"):
        inputs[invalid_case] = inputs[invalid_case].int()
    elif invalid_case == "sink_dtype":
        inputs["attn_sink"] = inputs["attn_sink"].bfloat16()
    else:
        inputs["inverse_rope_freqs"] = inputs["inverse_rope_freqs"][:, :, :-1].contiguous()

    def unexpected_launch(*args):
        pytest.fail("invalid inputs reached compilation or a kernel launch")

    monkeypatch.setattr(attention_op, "_get_dspark_arch_str", lambda: "sm_100")
    key = (6, 128, 32, attention_op._get_dspark_arch_str())
    monkeypatch.setattr(attention_op, "_dspark_attention_kernel_cache", {key: unexpected_launch})
    monkeypatch.setattr(attention_op, "_compile_dspark_attention", unexpected_launch)
    with pytest.raises(ValueError, match=reason):
        attention_op.fused_dsv4_dspark_attention(**inputs)


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("block", (5, 6))
@pytest.mark.parametrize("page_size", (16, 32, 64, 128, 256))
def test_fused_dsv4_dspark_attention_matches_reference(heads, block, page_size):
    inputs = _make_inputs(
        29,
        heads=heads,
        block=block,
        page_size=page_size,
        positions=[0, 1, 17, 32, 33, 63, 64, 65, 127, 128, 129, 193, 257, 390, 99],
        valid_lengths=[0, 1, 1, 3, 32, 1, 1, 127, 1, 0, 1, 127, 32, 3, 128],
        capacities=[512] * 14 + [0],
    )
    original_cache = inputs["kv_pages"].clone()
    expected = _reference(**inputs)
    actual = attention_op.fused_dsv4_dspark_attention(**inputs)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(inputs["kv_pages"], original_cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    "heads,page_size,batch_repeats",
    [(64, 32, 1), (128, 64, 1), (128, 256, 1), (128, 64, 64)],
)
def test_fused_dsv4_dspark_attention_cuda_graph_replay(heads, page_size, batch_repeats):
    inputs = _make_inputs(
        3,
        heads=heads,
        page_size=page_size,
        positions=[257, 5, 390] * batch_repeats,
        capacities=[512, 512, 0] * batch_repeats,
    )
    batch = 3 * batch_repeats
    block_x = torch.randn(batch, 6, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    freqs = dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device="cuda")
    block_freqs = inputs["inverse_rope_freqs"].view(-1, 32, 2)

    def forward():
        draft = preparation_op.cute_dsl_dspark_rmsnorm_rope_draft_block(
            block_x, weight, block_freqs, 1e-6
        )
        return attention_op.fused_dsv4_dspark_attention(**(inputs | {"draft_block": draft})), draft

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, draft = forward()
    for end in (17, 65, 127, 128, 129, 193, 255):
        inputs["positions"][:2].copy_(torch.tensor([end, end + 1], device="cuda"))
        inputs["valid_lengths"].copy_(torch.tensor([127, 1, 128] * batch_repeats, device="cuda"))
        inputs["block_tables"][:2].copy_(inputs["block_tables"][:2].roll(1, dims=1))
        inputs["kv_pages"].normal_()
        block_x.normal_()
        block_phases = freqs[inputs["positions"][:, None] + 1 + torch.arange(6, device="cuda")]
        inputs["inverse_rope_freqs"].copy_(torch.view_as_real(block_phases))
        expected_block = _norm_rope(block_x, weight, block_phases)
        expected_draft = F.pad(expected_block, (0, 0, 0, 2))
        # The independent Torch oracle reads the current page table and page data.
        expected = _reference(**(inputs | {"draft_block": expected_draft}))
        expected_cache = inputs["kv_pages"].clone()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=5e-2, atol=3e-2)
        torch.testing.assert_close(inputs["kv_pages"], expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(draft, expected_draft, rtol=2e-2, atol=2e-2)
        assert torch.count_nonzero(draft[:, 6:]) == 0


@pytest.mark.parametrize("num_heads", [64, 128])
def test_warmup_is_best_effort(monkeypatch, num_heads):
    import tensorrt_llm._torch.custom_ops.dspark_attention_custom_op as dspark_attention_op

    warnings = []
    monkeypatch.setattr(dspark_attention_op, "_get_dspark_arch_str", lambda: "sm_100")
    monkeypatch.setattr(
        dspark_attention_op,
        "cute_dsl_dspark_rmsnorm_rope_cache_write",
        lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic compile failure")),
    )
    monkeypatch.setattr(dspark_attention_op.logger, "warning", warnings.append)

    dspark_attention_op.warmup_fused_dsv4_dspark_attention(5, 1e-6, num_heads)

    assert len(warnings) == 1
    assert "self-JIT on first use" in warnings[0]
    assert "synthetic compile failure" in warnings[0]


def test_self_jit_reuses_one_kernel_across_runtime_shapes_and_scales(monkeypatch):
    assert [attention_op._get_dspark_arch_str(sm) for sm in (100, 103, 107)] == [
        "sm_100",
        "sm_103",
        "sm_107",
    ]
    assert all(attention_op._get_dspark_arch_str(sm) is None for sm in (90, 101, 109))
    cache = {}
    monkeypatch.setattr(attention_op, "_dspark_attention_kernel_cache", cache)
    for batch in (1, 3, 32):
        inputs = _make_inputs(17 + batch, batch=batch, extra_pages=batch + 41)
        if batch == 3:
            inputs["kv_pages"] = inputs["kv_pages"].clone()
            inputs["block_tables"] = inputs["block_tables"].clone()
        inputs["softmax_scale"] *= 0.5 if batch % 2 == 0 else 1
        expected = _reference(**inputs)
        actual = attention_op.fused_dsv4_dspark_attention(**inputs)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
        assert len(cache) == 1
        if batch == 1:

            def unexpected_compile(*args):
                pytest.fail("runtime shape or scale change caused recompilation")

            monkeypatch.setattr(attention_op, "_compile_dspark_attention", unexpected_compile)


def test_compile_without_real_specimens_and_cached_wrapper_avoids_views(monkeypatch):
    arch_str = attention_op._get_dspark_arch_str()

    def unexpected_compile_op(*args, **kwargs):
        pytest.fail("DSpark compilation used a real tensor specimen")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", unexpected_compile_op)
        patch.setattr(torch, "empty_like", unexpected_compile_op)
        patch.setattr(attention_op.cute.runtime, "from_dlpack", unexpected_compile_op)
        compiled = attention_op._compile_dspark_attention(5, arch_str, 32, 128)
    monkeypatch.setattr(
        attention_op,
        "_dspark_attention_kernel_cache",
        {(5, 128, 32, arch_str): compiled},
    )
    inputs = _make_inputs(41, block=5, positions=[257, 9])

    def unexpected_host_op(*args, **kwargs):
        pytest.fail("cached DSpark host wrapper used a Python tensor conversion or view")

    monkeypatch.setattr(attention_op.cute.runtime, "from_dlpack", unexpected_host_op)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_host_op)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected_host_op)
    for method in ("permute", "unsqueeze", "reshape"):
        monkeypatch.setattr(torch.Tensor, method, unexpected_host_op)
    output = attention_op.fused_dsv4_dspark_attention(**inputs)
    assert output.shape == inputs["q"].shape


def test_attention_cache_miss_rejects_cuda_graph_capture(monkeypatch):
    inputs = _make_inputs(43, block=5)
    monkeypatch.setattr(attention_op, "_dspark_attention_kernel_cache", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("compiler or allocation was called before the capture guard")

    monkeypatch.setattr(attention_op, "_compile_dspark_attention", unexpected_compile)
    monkeypatch.setattr(torch, "empty_like", unexpected_compile)
    with pytest.raises(RuntimeError, match="must run eagerly before graph capture"):
        attention_op.fused_dsv4_dspark_attention(**inputs)


def test_preparation_cache_misses_reject_cuda_graph_capture(monkeypatch):
    preparation_op._compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("preparation compiler was called during CUDA graph capture")

    monkeypatch.setattr(preparation_op.cute, "compile", unexpected_compile)
    with pytest.raises(RuntimeError, match="cache-write must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_cache_write(1e-6)
    with pytest.raises(RuntimeError, match="draft-block must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_draft_block(5, 1e-6)


def test_preparation_self_jit_covers_dynamic_batches():
    preparation_op._compile_dspark_rmsnorm_rope_cache_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    for batch in (1, 3, 32):
        inputs = _make_inputs(31 + batch, batch=batch)
        main = torch.randn(batch, 1, 512, device="cuda", dtype=torch.bfloat16)
        block = inputs["draft_block"][:, :6].contiguous()
        freqs = dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device="cuda")
        start = inputs["positions"]
        main_freqs = freqs[start][:, None]
        block_freqs = torch.view_as_complex(inputs["inverse_rope_freqs"])
        slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int64)
        cache = torch.randn(batch + 1, 128, 512, device="cuda", dtype=torch.bfloat16)
        expected_cache = cache.clone()
        expected_main = _norm_rope(main, weight, main_freqs)
        expected_cache[slots, start % 128] = expected_main[:, 0]
        expected_block = _norm_rope(block, weight, block_freqs)
        actual_slots, cache_seqs = preparation_op.cute_dsl_dspark_rmsnorm_rope_cache_write(
            main,
            weight,
            torch.view_as_real(main_freqs).view(batch, 32, 2),
            cache,
            slots,
            start,
            1e-6,
        )
        draft = preparation_op.cute_dsl_dspark_rmsnorm_rope_draft_block(
            block, weight, inputs["inverse_rope_freqs"].view(-1, 32, 2), 1e-6
        )
        torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_slots, slots.to(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(cache_seqs, start.to(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(draft[:, :6], expected_block, rtol=2e-2, atol=2e-2)
        assert torch.count_nonzero(draft[:, 6:]) == 0
    assert preparation_op._compile_dspark_rmsnorm_rope_cache_write.cache_info().misses == 1
    assert preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_info().misses == 1


def _norm_rope(x, weight, freqs):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    normalized = (normalized * weight.float()).to(x.dtype)
    return dspark._rope_last_dims_batched(normalized, _ROPE_DIM, freqs)


def test_warmup_propagates_compile_failure(monkeypatch):
    inputs = _make_inputs(page_size=256)
    monkeypatch.setattr(attention_op, "_dspark_attention_kernel_cache", {})

    def fail_compile(*args):
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(attention_op, "_compile_dspark_attention", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        attention_op.fused_dsv4_dspark_attention(**inputs)


def _attention_case(seed=0, start_positions=(40, 40), lengths=None, sink=None):
    inputs = _make_inputs(
        seed, block=5, page_size=256, positions=start_positions, valid_lengths=lengths
    )
    if sink is not None:
        inputs["attn_sink"].fill_(sink)
    inputs["inverse_rope_freqs"].zero_()
    inputs["inverse_rope_freqs"][..., 0] = 1
    output = attention_op.fused_dsv4_dspark_attention(**inputs)
    histories = []
    pages, tables = inputs["kv_pages"], inputs["block_tables"]
    for row, (end, length) in enumerate(zip(start_positions, inputs["valid_lengths"].tolist())):
        logical = torch.arange(end - length, end, device="cuda")
        history = pages[tables[row, logical // pages.shape[1]].long(), logical % pages.shape[1]]
        histories.append(torch.cat((history, inputs["draft_block"][row, :5])))
    return output, inputs["q"], histories, inputs["attn_sink"]


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


@pytest.mark.skipif(
    get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
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
    get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The fused DSpark output projection requires SM107 Rubin CuTe DSL",
)
def test_dspark_fused_fp8_output_projection_matches_separate_and_replays_graph():
    from _torch.helpers import per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLFp8RubinBmmQuantRunner,
        CuteDSLMXFP8RubinLinear,
    )
    from tensorrt_llm._torch.models.modeling_dspark import _output_projection
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod
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
            return FP8BlockScalesLinearMethod().apply(self, (activation, scale), None)

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

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _output_projection(o, **args)
    o.mul_(1.125)
    graph.replay()
    expected = _output_projection(o, **args)
    torch.testing.assert_close(captured, expected, rtol=5e-2, atol=3e-2)


@pytest.mark.parametrize("num_heads", [64, 128])
def test_warmup_uses_configured_head_count(monkeypatch, num_heads):
    monkeypatch.setattr(
        attention_op,
        "cute_dsl_dspark_rmsnorm_rope_cache_write",
        lambda main, weight, freqs, cache, slots, pos, eps: (slots.int(), pos.int()),
    )
    monkeypatch.setattr(
        attention_op,
        "cute_dsl_dspark_rmsnorm_rope_draft_block",
        lambda block, *args: block.new_empty((1, 8, 512)),
    )
    calls = []

    def attention(q, draft, cache, tables, positions, lengths, capacities, sink, freqs, scale):
        assert q.shape == (1, 5, num_heads, 512)
        assert sink.shape == (num_heads,)
        calls.append(num_heads)

    monkeypatch.setattr(attention_op, "fused_dsv4_dspark_attention", attention)
    attention_op.warmup_fused_dsv4_dspark_attention(5, 1e-6, num_heads)
    assert calls == [num_heads]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("block", [5, 6])
def test_production_ring_fused_qkv_layout_numbers_and_graph(monkeypatch, batch, block):
    torch.manual_seed(220 + batch * 10 + block)
    heads, dim, hidden, rank, groups, o_rank, window = 64, 512, 128, 64, 8, 32, 128

    def weight(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02

    x, main = weight(batch, block, hidden), weight(batch, 1, hidden)
    qkv_weight = weight(rank + dim, hidden)

    def projection(inp):
        return F.linear(inp, qkv_weight)

    kwargs = dict(
        wq_a=None,
        wkv=None,
        q_norm_w=torch.ones(rank, device="cuda", dtype=x.dtype),
        wq_b=weight(heads * dim, rank),
        kv_norm_w=torch.ones(dim, device="cuda", dtype=x.dtype),
        wo_a=weight(groups * o_rank, heads * dim // groups),
        wo_b=weight(hidden, groups * o_rank),
        attn_sink=torch.randn(heads, device="cuda"),
        n_heads=heads,
        head_dim=dim,
        rope_head_dim=64,
        n_groups=groups,
        o_lora_rank=o_rank,
        window_size=window,
        eps=1e-6,
        softmax_scale=dim**-0.5,
        freqs_cis=dspark.precompute_dspark_freqs_cis(64, 1024, device="cuda"),
        persist=True,
        qkv_proj=projection,
    )
    slots = torch.tensor([2, 0][:batch], device="cuda", dtype=torch.long)
    start = torch.tensor([64, 192][:batch], device="cuda", dtype=torch.long)
    lengths = torch.tensor([1, 127][:batch], device="cuda", dtype=torch.long)
    initial = weight(3, window, dim)
    cache = initial.clone()
    _, block_view, main_view = dspark._qkv_projections(
        x, main, wq_a=None, wkv=None, q_norm_w=kwargs["q_norm_w"], qkv_proj=projection
    )
    assert block_view.stride(-2) == rank + dim and not block_view.is_contiguous()
    assert main_view.stride(-2) == rank + dim
    if batch > 1:
        assert not main_view.is_contiguous()

    def run(target):
        metadata = dspark._dspark_ring_attention_metadata(slots, start, lengths, window)
        return dspark.dspark_attention_forward_batched(
            x, main, start, target, slots, lengths, ring_attention_metadata=metadata, **kwargs
        )

    # The actual source fallback computes the oracle; only path selection changes.
    expected_cache = initial.clone()
    with monkeypatch.context() as patch:
        patch.setattr(dspark, "IS_CUTLASS_DSL_AVAILABLE", False)
        expected = run(expected_cache)
    real_cache_write = dspark.cute_dsl_dspark_rmsnorm_rope_cache_write
    real_draft = dspark.cute_dsl_dspark_rmsnorm_rope_draft_block
    observed = []

    def cache_write(inp, *rest):
        assert inp.is_contiguous()
        observed.append(("main", inp.stride()))
        return real_cache_write(inp, *rest)

    def draft(inp, *rest):
        assert inp.is_contiguous()
        observed.append(("block", inp.stride()))
        return real_draft(inp, *rest)

    cache_spy, draft_spy = Mock(wraps=cache_write), Mock(wraps=draft)
    with monkeypatch.context() as patch:
        patch.setattr(dspark, "cute_dsl_dspark_rmsnorm_rope_cache_write", cache_spy)
        patch.setattr(dspark, "cute_dsl_dspark_rmsnorm_rope_draft_block", draft_spy)
        actual = run(cache)
    cache_spy.assert_called_once()
    draft_spy.assert_called_once()
    assert [x[0] for x in observed] == ["main", "block"]
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache[1], initial[1], rtol=0, atol=0)

    # Capture the unchanged production path after all specializations are warm.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(cache)
    for position, valid in ((16, 1), (64, 127), (192, 127)):
        start.copy_(torch.tensor([position, position + 1][:batch], device="cuda"))
        lengths.fill_(valid)
        x.normal_(0, 0.02)
        main.normal_(0, 0.02)
        cache.copy_(initial)
        expected_cache = initial.clone()
        with monkeypatch.context() as patch:
            patch.setattr(dspark, "IS_CUTLASS_DSL_AVAILABLE", False)
            expected = run(expected_cache)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=8e-2, atol=1e-2)
        torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cache[1], initial[1], rtol=0, atol=0)
