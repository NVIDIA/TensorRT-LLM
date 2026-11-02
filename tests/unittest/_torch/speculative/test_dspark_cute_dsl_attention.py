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
        ("block_size", "requires block 5/6"),
        ("heads", "64/128 heads"),
        ("page_size", "16/32/64/128/256-token pages"),
        ("draft_shape", "draft KV must have shape"),
        ("draft_layout", "must be contiguous"),
        ("table_dtype", "contiguous INT32 rows"),
        ("table_layout", "contiguous INT32 rows"),
        ("positions", "contiguous INT64 batch vectors"),
        ("valid_lengths", "contiguous INT64 batch vectors"),
        ("capacities", "contiguous INT64 batch vectors"),
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

    monkeypatch.setattr(attention_op, "_get_dspark_arch_str", lambda: "sm_100a")
    key = (6, 128, 32, attention_op._get_dspark_arch_str(), None)
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
        positions=[0, 1, 32, 33, 127, 128, 129, 257, 390, 99],
        valid_lengths=[0, 1, 3, 32, 127, 128, 17, 128, 3, 128],
        capacities=[512] * 9 + [0],
    )
    original_cache = inputs["kv_pages"].clone()
    expected = _reference(**inputs)
    actual = attention_op.fused_dsv4_dspark_attention(**inputs)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(inputs["kv_pages"], original_cache, rtol=0, atol=0)


@pytest.mark.parametrize("heads,page_size", [(64, 32), (128, 64), (128, 256)])
def test_fused_dsv4_dspark_attention_cuda_graph_replay(heads, page_size):
    inputs = _make_inputs(
        3,
        heads=heads,
        page_size=page_size,
        positions=[257, 5, 390],
        capacities=[512, 512, 0],
    )
    main_x = torch.randn(3, 1, 512, device="cuda", dtype=torch.bfloat16)
    block_x = torch.randn(3, 6, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    freqs = dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device="cuda")
    main_freqs = torch.view_as_real(freqs[inputs["positions"]]).contiguous()
    block_freqs = inputs["inverse_rope_freqs"].view(-1, 32, 2)

    def forward():
        preparation_op.cute_dsl_dspark_rmsnorm_rope_page_write(
            main_x,
            weight,
            main_freqs,
            inputs["kv_pages"],
            inputs["block_tables"],
            inputs["positions"],
            inputs["capacities"],
            1e-6,
        )
        draft = preparation_op.cute_dsl_dspark_rmsnorm_rope_draft_block(
            block_x, weight, block_freqs, 1e-6
        )
        return attention_op.fused_dsv4_dspark_attention(**(inputs | {"draft_block": draft})), draft

    # Eager execution warms all preparation and attention specializations.
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, draft = forward()
    for end in (127, 128, 129, 255):
        inputs["positions"][:2].copy_(torch.tensor([end, end + 1], device="cuda"))
        inputs["valid_lengths"].copy_(torch.tensor([128, 3, 128], device="cuda"))
        inputs["block_tables"][:2].copy_(inputs["block_tables"][:2].roll(1, dims=1))
        main_x.normal_()
        block_x.normal_()
        main_freqs.copy_(torch.view_as_real(freqs[inputs["positions"]]))
        block_phases = freqs[inputs["positions"][:, None] + 1 + torch.arange(6, device="cuda")]
        inputs["inverse_rope_freqs"].copy_(torch.view_as_real(block_phases))
        expected_main = _norm_rope(main_x, weight, freqs[inputs["positions"]][:, None])
        expected_block = _norm_rope(block_x, weight, block_phases)
        expected_cache = _write_reference(inputs["kv_pages"], expected_main, inputs)
        expected_draft = F.pad(expected_block, (0, 0, 0, 2))
        expected = _reference(
            **(inputs | {"kv_pages": expected_cache, "draft_block": expected_draft})
        )
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=5e-2, atol=3e-2)
        torch.testing.assert_close(inputs["kv_pages"], expected_cache, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(draft, expected_draft, rtol=2e-2, atol=2e-2)
        assert torch.count_nonzero(draft[:, 6:]) == 0


@pytest.mark.parametrize("page_size", (128, 256))
def test_dspark_attention_forward_batched_matches_fallback(monkeypatch, page_size):
    g = _make_attn_inputs(seed=17, device="cuda", page_size=page_size)
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        g[name].mul_(0.2)
    for name in ("x", "main_x", "kv_cache0"):
        g[name].mul_(0.1)
    before = g["kv_cache0"].clone()
    actual = _run(g)
    actual_cache = g["kv_cache0"].clone()
    g["kv_cache0"].copy_(before)
    with monkeypatch.context() as patch:
        patch.setattr(torch.ops.trtllm, "fused_dsv4_dspark_attention", _reference)
        patch.setattr(dspark, "is_fused_dspark_rmsnorm_rope_supported", lambda *a: False)
        expected = _run(g)
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(actual_cache, g["kv_cache0"], rtol=2e-2, atol=2e-2)


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
        compiled = attention_op._compile_dspark_attention(5, arch_str, 32, None, 128)
    monkeypatch.setattr(
        attention_op,
        "_dspark_attention_kernel_cache",
        {(5, 128, 32, arch_str, None): compiled},
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
    preparation_op._compile_dspark_rmsnorm_rope_page_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    device = torch.cuda.current_device()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("preparation compiler was called during CUDA graph capture")

    monkeypatch.setattr(preparation_op.cute, "compile", unexpected_compile)
    with pytest.raises(RuntimeError, match="page-write must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_page_write(32, 1e-6, device)
    with pytest.raises(RuntimeError, match="draft-block must be warmed up"):
        preparation_op._compile_dspark_rmsnorm_rope_draft_block(5, 1e-6, device)


def test_preparation_self_jit_covers_dynamic_batches():
    preparation_op._compile_dspark_rmsnorm_rope_page_write.cache_clear()
    preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_clear()
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    for batch in (1, 3, 32):
        inputs = _make_inputs(31 + batch, batch=batch)
        main = torch.randn(batch, 1, 512, device="cuda", dtype=torch.bfloat16)
        block = inputs["draft_block"][:, :6].contiguous()
        freqs = dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device="cuda")
        main_freqs = freqs[inputs["positions"]][:, None]
        block_freqs = torch.view_as_complex(inputs["inverse_rope_freqs"])
        expected_cache = _write_reference(
            inputs["kv_pages"], _norm_rope(main, weight, main_freqs), inputs
        )
        expected_block = _norm_rope(block, weight, block_freqs)
        preparation_op.cute_dsl_dspark_rmsnorm_rope_page_write(
            main,
            weight,
            torch.view_as_real(main_freqs).view(batch, 32, 2),
            inputs["kv_pages"],
            inputs["block_tables"],
            inputs["positions"],
            inputs["capacities"],
            1e-6,
        )
        draft = preparation_op.cute_dsl_dspark_rmsnorm_rope_draft_block(
            block, weight, inputs["inverse_rope_freqs"].view(-1, 32, 2), 1e-6
        )
        torch.testing.assert_close(inputs["kv_pages"], expected_cache, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(draft[:, :6], expected_block, rtol=2e-2, atol=2e-2)
        assert torch.count_nonzero(draft[:, 6:]) == 0
    assert preparation_op._compile_dspark_rmsnorm_rope_page_write.cache_info().misses == 1
    assert preparation_op._compile_dspark_rmsnorm_rope_draft_block.cache_info().misses == 1


@pytest.fixture(params=[False, True], ids=["native", "unsupported_arch_fallback"])
def force_fallback(request, monkeypatch):
    if request.param:
        monkeypatch.setattr(preparation_op, "_get_dspark_arch_str", lambda: None)
        monkeypatch.setattr(preparation_op, "_is_dspark_rmsnorm_rope_sm_supported", lambda: False)
        monkeypatch.setattr(dspark, "_get_dspark_arch_str", lambda: None)

        def unsupported_kernel(*args, **kwargs):
            pytest.fail("Unsupported architecture entered a fused DSpark kernel")

        monkeypatch.setattr(dspark, "cute_dsl_dspark_rmsnorm_rope", unsupported_kernel)
        monkeypatch.setattr(dspark, "cute_dsl_dspark_rmsnorm_rope_page_write", unsupported_kernel)
        monkeypatch.setattr(dspark, "cute_dsl_dspark_rmsnorm_rope_draft_block", unsupported_kernel)
        monkeypatch.setattr(torch.ops.trtllm, "fused_dsv4_dspark_attention", unsupported_kernel)
        assert preparation_op._get_dspark_arch_str() is None
    return request.param


def _norm_rope(x, weight, freqs):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    normalized = (normalized * weight.float()).to(x.dtype)
    return dspark._rope_last_dims_batched(normalized, _ROPE_DIM, freqs)


def _write_reference(cache, main_kv, inputs):
    cache = cache.clone()
    for row, (end, capacity) in enumerate(
        zip(inputs["positions"].tolist(), inputs["capacities"].tolist())
    ):
        token = end - 1
        if 0 <= token < capacity:
            page = int(inputs["block_tables"][row, token // cache.shape[1]])
            cache[page, token % cache.shape[1]] = main_kv[row, 0]
    return cache


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("page_size", (32, 256))
@pytest.mark.parametrize("block", (3, 5, 6))
def test_dspark_attention_forward_matches_reference(
    heads, page_size, block, force_fallback, monkeypatch
):
    inputs = _make_inputs(
        17,
        block=block,
        heads=heads,
        page_size=page_size,
        positions=[1, 33, 129, 390, 0],
        valid_lengths=[1, 3, 128, 3, 0],
        capacities=[512, 512, 512, 512, 0],
    )
    batch, hidden, rank, groups, o_rank = 5, 64, 64, 8, 32

    def weight(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02

    x, main_x = weight(batch, block, hidden), weight(batch, 1, hidden)
    kwargs = dict(
        wq_a=weight(rank, hidden),
        q_norm_w=torch.ones(rank, device="cuda", dtype=x.dtype),
        wq_b=weight(heads * 512, rank),
        wkv=weight(512, hidden),
        kv_norm_w=torch.ones(512, device="cuda", dtype=x.dtype),
        wo_a=weight(groups * o_rank, heads * 512 // groups),
        wo_b=weight(hidden, groups * o_rank),
        attn_sink=inputs["attn_sink"],
        n_heads=heads,
        head_dim=512,
        rope_head_dim=_ROPE_DIM,
        n_groups=groups,
        o_lora_rank=o_rank,
        window_size=128,
        eps=1e-6,
        softmax_scale=inputs["softmax_scale"],
        freqs_cis=dspark.precompute_dspark_freqs_cis(_ROPE_DIM, 1024, device="cuda"),
    )
    positions = inputs["positions"]
    block_freqs = torch.view_as_complex(inputs["inverse_rope_freqs"])
    q = F.linear(
        dspark._rmsnorm(F.linear(x, kwargs["wq_a"]), kwargs["q_norm_w"], 1e-6), kwargs["wq_b"]
    ).unflatten(-1, (heads, 512))
    q = _norm_rope(q, torch.ones(512, device="cuda", dtype=x.dtype), block_freqs)
    main = _norm_rope(
        F.linear(main_x, kwargs["wkv"]),
        kwargs["kv_norm_w"],
        kwargs["freqs_cis"][positions][:, None],
    )
    draft = _norm_rope(F.linear(x, kwargs["wkv"]), kwargs["kv_norm_w"], block_freqs)
    expected_cache = _write_reference(inputs["kv_pages"], main, inputs)
    output = _reference(**(inputs | {"q": q, "draft_block": draft, "kv_pages": expected_cache}))
    output = torch.einsum(
        "bsgd,grd->bsgr",
        output.reshape(batch, block, groups, -1),
        kwargs["wo_a"].view(groups, o_rank, -1),
    )
    expected = F.linear(output.flatten(2), kwargs["wo_b"])
    fused_calls = []
    for module, name in (
        (dspark, "cute_dsl_dspark_rmsnorm_rope_page_write"),
        (dspark, "cute_dsl_dspark_rmsnorm_rope_draft_block"),
        (torch.ops.trtllm, "fused_dsv4_dspark_attention"),
    ):
        call = Mock(wraps=getattr(module, name))
        monkeypatch.setattr(module, name, call)
        fused_calls.append(call)
    actual = dspark.dspark_attention_forward(
        x,
        main_x,
        positions,
        inputs["kv_pages"],
        inputs["block_tables"],
        inputs["capacities"],
        inputs["valid_lengths"],
        **kwargs,
    )
    for call in fused_calls:
        if not force_fallback and block in (5, 6):
            call.assert_called_once()
        else:
            call.assert_not_called()
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(inputs["kv_pages"], expected_cache, rtol=2e-2, atol=2e-2)


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


@pytest.mark.parametrize(
    "start_pos,page_size,block", [(1, 128, 5), (3, 64, 5), (10, 16, 5), (200, 256, 6)]
)
def test_page_mapping_matches_reference(start_pos, page_size, block):
    from tensorrt_llm._torch.attention.backends.dspark import write_dspark_context

    inputs = _make_inputs(block=block, positions=[start_pos] * 3, page_size=page_size)
    values = inputs["draft_block"][:, :1].contiguous()
    expected = _write_reference(inputs["kv_pages"], values, inputs)
    write_dspark_context(
        values,
        inputs["positions"][:, None],
        torch.ones(3, 1, device="cuda", dtype=torch.bool),
        inputs["kv_pages"],
        inputs["block_tables"],
        inputs["capacities"],
    )
    torch.testing.assert_close(inputs["kv_pages"], expected, rtol=0, atol=0)


def test_page_mapping_masks_non_generation_rows():
    from tensorrt_llm._torch.attention.backends.dspark import write_dspark_context

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
    g["freqs"] = dspark.precompute_dspark_freqs_cis(rd, start_pos + 1 + block + 2)
    g["start_pos"] = torch.full((b,), start_pos, dtype=torch.long)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}


def _run(g):
    return dspark.dspark_attention_forward(
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
    from tensorrt_llm._torch.attention.backends.dspark import write_dspark_context

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


def test_batched_attention_cuda_graph_capture_replay(force_fallback):
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


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="The fused DSpark FP8 output projection is a Rubin specialization",
)
def test_fused_dsv4_dspark_attention_fp8_matches_separate_quant_and_replays_graph():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        fused_dsv4_dspark_attention,
        fused_dsv4_dspark_attention_fp8,
    )

    groups = 16
    inputs = _make_inputs(47, page_size=256, positions=[257, 390], valid_lengths=[3, 5])
    q = inputs["q"]
    block = q.shape[1]

    def separate_quantize():
        bf16 = fused_dsv4_dspark_attention(**inputs)
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

    actual_fp8, actual_scale = fused_dsv4_dspark_attention_fp8(**inputs, output_groups=groups)
    expected_fp8, expected_scale = separate_quantize()
    assert_quantized_close(actual_fp8, actual_scale, expected_fp8, expected_scale)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_fp8, captured_scale = fused_dsv4_dspark_attention_fp8(
            **inputs, output_groups=groups
        )
    q.copy_(torch.randn_like(q))
    inputs["valid_lengths"].copy_(torch.tensor([7, 2], device=q.device))
    graph.replay()
    expected_fp8, expected_scale = separate_quantize()
    assert_quantized_close(captured_fp8, captured_scale, expected_fp8, expected_scale)


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
