# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 packed formats, cache kernels, attention kernels and lazy import behavior."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
    packed_sparse_attention,
    quantize_scatter_rows,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    INDEX_PAGE_BYTES,
    INDEX_PAGE_ROWS,
    pack_rows,
    row_bytes,
    unpack_rows,
)


def _inputs(main_count=73, dtype=torch.int64):
    torch.manual_seed(7401)
    q = torch.randn(2, 32, 512, device="cuda", dtype=torch.bfloat16)
    swa = pack_rows(torch.randn(141, 512, device="cuda", dtype=torch.bfloat16), "swa")
    owner = torch.empty(103, 356, device="cuda", dtype=torch.uint8)
    main = owner[:, :288]
    main.copy_(pack_rows(torch.randn(103, 512, device="cuda", dtype=torch.bfloat16), "main"))
    si = torch.randint(0, 141, (2, 77), device="cuda", dtype=dtype)
    mi = torch.randint(0, 103, (2, main_count), device="cuda", dtype=dtype)
    si[0, 3:] = -1
    if main_count:
        mi[0] = -1
        mi[1, 0] = 103
        if dtype == torch.int64:
            mi[1, 1] = 1 << 40
    sink = torch.randn(32, device="cuda")
    return q, swa, main, si, mi, sink, 512**-0.5


def _reference(q, swa, main, si, mi, sink, scale):
    parts, masks = [], []
    for pool, indices, fmt in ((swa, si, "swa"), (main, mi, "main")):
        valid = (indices >= 0) & (indices < pool.shape[0])
        values = unpack_rows(pool[indices.clamp(0, pool.shape[0] - 1).long()], 512, fmt)
        parts.append(torch.where(valid[..., None], values, 0))
        masks.append(valid)
    kv = torch.cat(parts, 1).float()
    logits = torch.einsum("qhd,qkd->qhk", q.float(), kv) * scale
    logits.masked_fill_(~torch.cat(masks, 1)[:, None, :], float("-inf"))
    probabilities = torch.softmax(
        torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, -1)), -1), -1
    )[..., :-1]
    return torch.einsum("qhk,qkd->qhd", probabilities, kv).bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("main_count", [0, 73])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_attention_parity(main_count, dtype):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args = _inputs(main_count, dtype)
    actual = packed_sparse_attention(*args)
    torch.testing.assert_close(actual, _reference(*args), atol=0.016, rtol=0.016)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("queries", [1, 8, 129])
@pytest.mark.parametrize("main_count", [0, 73])
def test_mixed_cache_head_and_batch_geometries(heads, queries, main_count):
    """Preserve the merged mixed-cache coverage using CSA2's packed row layouts."""
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    _, swa, main, si, mi, _, scale = _inputs(main_count)
    q = torch.randn(queries, heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda")
    si = si[1:2].repeat(queries, 1)
    mi = mi[1:2].repeat(queries, 1)
    args = (q, swa, main, si, mi, sink, scale)
    torch.testing.assert_close(
        packed_sparse_attention(*args), _reference(*args), atol=0.016, rtol=0.016
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_attention_graph_refresh():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args = _inputs()
    output = torch.empty_like(args[0])
    packed_sparse_attention(*args, output=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed_sparse_attention(*args, output=output)
    args[3][0].fill_(-1)
    args[4][0].fill_(-1)
    args[4][1].fill_(0)
    graph.replay()
    torch.testing.assert_close(output, _reference(*args), atol=0.016, rtol=0.016)
    assert torch.count_nonzero(output[0]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_attention_multiple_tiles_per_split():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    q, swa, main, si, mi, sink, scale = _inputs(512)
    q = q.repeat(8, 2, 1)
    si = si.repeat(8, 1)
    mi = mi.repeat(8, 1)
    sink = sink.repeat(2)
    args = (q, swa, main, si, mi, sink, scale)
    torch.testing.assert_close(
        packed_sparse_attention(*args), _reference(*args), atol=0.016, rtol=0.016
    )


def _rope_inputs(heads):
    q, swa, main, si, mi, sink, scale = _inputs()
    if heads == 16:
        q, sink = q[:, :16].contiguous(), sink[:16].contiguous()
    else:
        q, sink = q.repeat(1, 2, 1), sink.repeat(2)
    angles = torch.randn(37, 32, device="cuda")
    cos_sin = torch.stack((angles.cos(), angles.sin()), dim=1).contiguous()
    positions = torch.tensor([3, 19], dtype=torch.int64, device="cuda")
    return (q, swa, main, si, mi, sink, scale), positions, cos_sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
def test_packed_inverse_rope_native_parity(heads):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(heads)
    attention = packed_sparse_attention(*args)
    expected = attention.clone()
    torch.ops.trtllm.mla_rope_inplace(
        expected, positions.int(), cos_sin, heads, 448, 64, True, False
    )
    actual = packed_sparse_attention(*args, position_ids=positions, rotary_cos_sin=cos_sin)
    torch.testing.assert_close(actual[..., :448], attention[..., :448], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_inverse_rope_graph_positions_and_bounds():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(16)
    attention = packed_sparse_attention(*args)
    output = torch.empty_like(attention)
    packed_sparse_attention(*args, output=output, position_ids=positions, rotary_cos_sin=cos_sin)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed_sparse_attention(
            *args, output=output, position_ids=positions, rotary_cos_sin=cos_sin
        )
    positions.copy_(torch.tensor([17, 2], device="cuda"))
    graph.replay()
    expected = attention.clone()
    torch.ops.trtllm.mla_rope_inplace(expected, positions.int(), cos_sin, 16, 448, 64, True, False)
    torch.testing.assert_close(output, expected, atol=0.002, rtol=0.008)
    # Standalone callers receive a nonfinite tail, never an out-of-bounds read.
    positions.copy_(torch.tensor([-1, cos_sin.shape[0]], device="cuda"))
    graph.replay()
    assert torch.isnan(output[..., 448:]).all()
    torch.testing.assert_close(output[..., :448], attention[..., :448], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_inverse_rope_rejects_invalid_geometry():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import supports_packed_attention

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(16)
    assert not supports_packed_attention(*args, position_ids=positions)
    assert not supports_packed_attention(*args, rotary_cos_sin=cos_sin)
    assert not supports_packed_attention(
        *args, position_ids=positions, rotary_cos_sin=cos_sin.bfloat16()
    )
    assert not supports_packed_attention(
        *args, position_ids=positions, rotary_cos_sin=cos_sin[:, :, :16]
    )


def _sm100():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Fused CSA2 store is enabled only after SM100-family validation")


def _values(rows, dim, cache_format):
    storage = torch.randn(rows, dim + 17, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :dim]
    pattern = torch.tensor(
        [6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x[0].copy_(pattern.repeat((dim + 15) // 16)[:dim])
    x[2].zero_()
    x[2, 1::2] = -0.0
    x[4].copy_(pattern.repeat((dim + 15) // 16)[:dim] * 2.0**-12)
    x[6].copy_(pattern.repeat((dim + 15) // 16)[:dim] * 2.0**-126)
    # Include BF16 subnormal input channels and positive/negative zero.
    x[8].fill_(2.0**-133)
    x[8, 1::2].neg_()
    x[10].fill_(6 * 1.0625)
    x[12].fill_(6 * 1.1875)
    x[14].fill_(6 * 1.5 * 2.0**-9)
    x[16].fill_(6 * 464)
    x[18].fill_(6 * 468)
    if cache_format == "swa":
        x[20].fill_(2.0**-30)
    x[22, 0] = float("nan")
    x[24, 0] = float("inf")
    x[26, 0] = -float("inf")
    x[28, 0] = -float("nan")
    return x


def _storage(rows, dim, cache_format):
    width = row_bytes(dim, cache_format)
    stride = 356 if cache_format in ("main", "index") else width + 13
    offset = 288 if cache_format == "index" else 0
    storage = torch.full((rows + 8, stride), 77, dtype=torch.uint8, device="cuda")
    return storage, storage[:, offset : offset + width]


def _expected(storage, pool, slots, x, cache_format):
    expected = storage.clone()
    offset = pool.storage_offset() - storage.storage_offset()
    target = expected[:, offset : offset + pool.shape[1]]
    valid = (slots >= 0) & (slots < pool.shape[0])
    target.index_copy_(0, slots[valid].long(), pack_rows(x, cache_format)[valid])
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cache_format,dim",
    [
        ("main", 512),
        ("index", 128),
        ("swa", 512),
        ("main", 48),
        ("index", 96),
        ("swa", 96),
    ],
)
@torch.inference_mode()
def test_fused_store_exact_bytes(cache_format, dim):
    _sm100()
    torch.manual_seed(781)
    rows = 33
    x = _values(rows, dim, cache_format)
    storage, pool = _storage(rows, dim, cache_format)
    slot_storage = torch.empty(rows * 2, dtype=torch.int64, device="cuda")
    slots = slot_storage[::2]
    slots.copy_((torch.arange(rows, device="cuda") * 7) % pool.shape[0])
    slots[1], slots[3], slots[5], slots[7] = 2**40, -1, pool.shape[0], -(2**50)
    expected = _expected(storage, pool, slots, x, cache_format)
    quantize_scatter_rows(pool, slots, x, cache_format)
    torch.testing.assert_close(storage, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_fused_store_jobs_match_separate_launches():
    """One four-job launch (SWA, main, index footer, index-query split) writes the same bytes."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
        footer_job,
        quantize_scatter_jobs,
        row_job,
        split_job,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import RowTransform

    _sm100()
    torch.manual_seed(97)
    swa, main, index, queries = 37, 19, 19, 45
    values = {
        "swa": torch.randn(swa, 512, dtype=torch.bfloat16, device="cuda") * 3,
        "main": torch.randn(main, 512, dtype=torch.bfloat16, device="cuda") * 3,
        "index": torch.randn(index, 128, dtype=torch.bfloat16, device="cuda") * 3,
        "query": torch.randn(queries, 128, dtype=torch.bfloat16, device="cuda") * 3,
    }
    slots = {
        "swa": (torch.arange(swa, device="cuda") * 5 % 41).long(),
        "main": (torch.arange(main, device="cuda") * 3 % 23).long(),
    }
    slots["main"][2] = -1
    table = torch.randn(64, 64, dtype=torch.float32, device="cuda")
    positions = (torch.arange(swa, device="cuda") % 64).int()
    transform = RowTransform(
        norm=(torch.rand(512, dtype=torch.bfloat16, device="cuda") + 0.5, 1e-6),
        rope=(positions, table, 64),
    )
    index_transform = RowTransform(
        norm=(torch.rand(128, dtype=torch.bfloat16, device="cuda") + 0.5, 1e-6),
        rope=(positions[:main], table, 64),
    )

    def run(fused):
        pools = {
            "swa": torch.full((41, 528), 7, dtype=torch.uint8, device="cuda"),
            "main": torch.full((23, 288), 7, dtype=torch.uint8, device="cuda"),
            "index": torch.zeros(INDEX_PAGE_BYTES, dtype=torch.uint8, device="cuda"),
            "data": torch.zeros(queries, 64, dtype=torch.uint8, device="cuda"),
            "scales": torch.zeros(queries, 4, dtype=torch.uint8, device="cuda"),
        }
        jobs = [
            row_job(
                pools["swa"], slots["swa"], values["swa"], "swa", transform.norm, transform.rope
            ),
            row_job(
                pools["main"], slots["main"], values["main"], "main", None, index_transform.rope
            ),
            footer_job(
                pools["index"],
                slots["main"],
                values["index"],
                INDEX_PAGE_ROWS,
                index_transform.norm,
                index_transform.rope,
            ),
            split_job(values["query"], pools["data"], pools["scales"]),
        ]
        if fused:
            quantize_scatter_jobs(jobs)
        else:
            for job in jobs:
                quantize_scatter_jobs([job])
        torch.cuda.synchronize()
        return pools

    separate, fused = run(False), run(True)
    for name in separate:
        torch.testing.assert_close(fused[name], separate[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cache_format,dim,norm", [("main", 512, False), ("index", 128, True), ("swa", 512, True)]
)
@torch.inference_mode()
def test_fused_store_norm_rope(cache_format, dim, norm):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import RowTransform
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        apply_row_transform,
        unpack_rows,
    )

    _sm100()
    torch.manual_seed(613)
    rows, rope_dim = 37, 64
    # Finite rows only: non-finite handling is covered by the exact-byte tests.
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda") * 3
    weight = (torch.rand(dim, device="cuda") + 0.5).bfloat16()
    positions = torch.randint(0, 4096, (rows,), dtype=torch.int32, device="cuda")
    angles = torch.rand(4096, rope_dim // 2, device="cuda") * 6.283
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
    transform = RowTransform(
        norm=(weight, 1e-6) if norm else None, rope=(positions, cos_sin, rope_dim)
    )
    storage, pool = _storage(rows, dim, cache_format)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots[5] = -1
    reference = apply_row_transform(x, transform)
    expected = _expected(storage.clone(), pool, slots, reference, cache_format)
    quantize_scatter_rows(pool, slots, x, cache_format, norm=transform.norm, rope=transform.rope)
    # The fused prologue uses the hardware rsqrt and FMA ordering; a bit may
    # differ at a rounding boundary, so compare dequantized rows and bound the
    # raw byte mismatch instead of demanding exact bytes.
    assert (storage != expected).float().mean().item() < 0.01
    torch.testing.assert_close(
        unpack_rows(pool[slots[slots >= 0]], dim, cache_format).float(),
        reference[slots >= 0].float(),
        atol=0.25 if cache_format == "swa" else 1.5,
        rtol=0.1,
    )
    if cache_format == "index":
        pages = torch.zeros(2 * 64 * 68, dtype=torch.uint8, device="cuda")
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            quantize_scatter_index_pages,
        )
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import read_index_rows

        quantize_scatter_index_pages(pages, slots, x, 64, norm=transform.norm, rope=transform.rope)
        torch.testing.assert_close(
            unpack_rows(read_index_rows(pages, slots[slots >= 0]), dim, "index").float(),
            reference[slots >= 0].float(),
            atol=1.5,
            rtol=0.1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
@torch.inference_mode()
def test_fused_store_graph_changes_padding(cache_format, dim):
    _sm100()
    torch.manual_seed(913)
    x = _values(33, dim, cache_format)
    storage, pool = _storage(33, dim, cache_format)
    slots = torch.arange(33, device="cuda", dtype=torch.int64)
    for _ in range(3):
        quantize_scatter_rows(pool, slots, x, cache_format)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_scatter_rows(pool, slots, x, cache_format)
    for valid_count in (33, 0, 1, 17, 33):
        slots.copy_(torch.arange(33, device="cuda"))
        slots[valid_count:] = 2**40
        x.neg_()
        storage.fill_(77)
        expected = _expected(storage, pool, slots, x, cache_format)
        graph.replay()
        torch.testing.assert_close(storage, expected, atol=0, rtol=0)


def _gather_reference(pool, slots, dim, cache_format):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    valid = (slots >= 0) & (slots < pool.shape[0])
    if pool.shape[0] == 0:
        return torch.zeros((*slots.shape, dim), dtype=torch.bfloat16, device=pool.device)
    selected = pool[torch.where(valid, slots, 0).long()]
    decoded = unpack_rows(selected, dim, cache_format)
    return torch.where(valid[..., None], decoded, 0)


def _assert_bf16_decode(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cache_format,dim,slot_dtype,rank",
    [
        ("main", 512, torch.int64, 2),
        ("index", 128, torch.int32, 1),
        ("swa", 512, torch.int64, 2),
        ("main", 48, torch.int32, 1),
        ("index", 96, torch.int64, 2),
        ("swa", 96, torch.int32, 1),
    ],
)
def test_fused_gather_strided_exact(cache_format, dim, slot_dtype, rank):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    torch.manual_seed(791)
    storage, pool = _storage(33, dim, cache_format)
    pool[:33].copy_(pack_rows(_values(33, dim, cache_format), cache_format))
    # Additional allocated rows are initialized, including padding ownership bytes.
    before = storage.clone()
    backing = torch.empty(66, dtype=slot_dtype, device="cuda")
    flat = backing[::2]
    flat.copy_(torch.arange(33, device="cuda") % 33)
    flat[1], flat[3] = -1, pool.shape[0]
    flat[5] = 2**40 if slot_dtype == torch.int64 else 2**30
    flat[7] = -(2**50) if slot_dtype == torch.int64 else -100
    slots = flat.reshape(3, 11) if rank == 2 else flat
    expected = _gather_reference(pool, slots, dim, cache_format)
    actual = gather_rows(pool, slots, dim, cache_format)
    _assert_bf16_decode(actual, expected)
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


def _fp8_python(code):
    import math

    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.copysign(float("nan"), sign)
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
def test_fused_gather_all_scale_bytes(cache_format, dim):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    _, pool = _storage(256, dim, cache_format)
    data_bytes = dim if cache_format == "swa" else dim // 2
    payload = torch.arange(data_bytes, dtype=torch.int64, device="cuda").to(torch.uint8)
    pool[:256, :data_bytes].copy_(payload)
    pool[:256, data_bytes:].copy_(torch.arange(256, device="cuda", dtype=torch.uint8)[:, None])
    slots = torch.arange(256, device="cuda", dtype=torch.int64)
    actual = gather_rows(pool, slots, dim, cache_format)
    _assert_bf16_decode(actual, _gather_reference(pool, slots, dim, cache_format))
    levels = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    expected_rows = []
    for scale_byte in range(256):
        scale = _fp8_python(scale_byte) if cache_format == "main" else 2.0 ** (scale_byte - 127)
        # Mirror the FP32 intermediate range before rounding into BF16.
        if cache_format != "main" and scale_byte == 255:
            scale = float("inf")
        values = []
        for channel in range(dim):
            if cache_format == "swa":
                value = _fp8_python(channel % 256)
            else:
                byte = (channel // 2) % 256
                value = levels[(byte >> (channel % 2 * 4)) & 15]
            values.append(value * scale)
        expected_rows.append(values)
    independent = torch.tensor(expected_rows, dtype=torch.float64).float().bfloat16().cuda()
    _assert_bf16_decode(actual, independent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
def test_fused_gather_graph_refresh(cache_format, dim):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    torch.manual_seed(792)
    _, pool = _storage(33, dim, cache_format)
    pool[:33].copy_(pack_rows(_values(33, dim, cache_format), cache_format))
    slots = torch.arange(33, device="cuda", dtype=torch.int64).reshape(3, 11)
    gather_rows(pool, slots, dim, cache_format)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_rows(pool, slots, dim, cache_format)
    for active in (0, 17, 33):
        flat = torch.arange(33, device="cuda", dtype=torch.int64)
        slots.copy_(torch.where(flat < active, flat, 2**40).reshape_as(slots))
        pool[:33, 0].bitwise_xor_(8)
        graph.replay()
        _assert_bf16_decode(actual, _gather_reference(pool, slots, dim, cache_format))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_gather_empty_and_fallback(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import quantization

    _sm100()
    pool = torch.empty((0, 288), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([[-1, 0, 2**40]], dtype=torch.int64, device="cuda")
    assert torch.count_nonzero(quantization.gather_rows(pool, slots, 512, "main")) == 0
    empty = slots[:, :0]
    assert quantization.gather_rows(pool, empty, 512, "main").shape == (1, 0, 512)
    _, pool = _storage(33, 512, "main")
    expected = _gather_reference(pool, slots, 512, "main")
    monkeypatch.setattr(quantization, "_fused_gather_supported", lambda _: False)
    _assert_bf16_decode(quantization.gather_rows(pool, slots, 512, "main"), expected)


def test_kernel_import_without_optional_cute_or_cuda_context():
    kernel = (
        Path(__file__).resolve().parents[6]
        / "tensorrt_llm/_torch/attention/backends/sparse/csa2/kernel.py"
    )
    program = """
import importlib.abc
import importlib.util
import sys
import torch
import triton

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'cutlass', 'cuda'}:
            raise AssertionError('Unexpected optional import: ' + fullname)

sys.meta_path.insert(0, RejectOptional())
assert not torch.cuda.is_initialized()
spec = importlib.util.spec_from_file_location('csa2_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not torch.cuda.is_initialized()
assert hasattr(torch.ops.trtllm, 'csa2_indexer_q_gemm_rope_fp4')
"""
    subprocess.run([sys.executable, "-c", program, str(kernel)], check=True, timeout=60)


@pytest.mark.parametrize(
    "cache_format,dim,expected_bytes", [("main", 512, 288), ("index", 128, 68), ("swa", 512, 528)]
)
def test_quantized_row_layout(cache_format, dim, expected_bytes):
    x = torch.zeros(2, dim, dtype=torch.bfloat16)
    x[1] = 6
    rows = pack_rows(x, cache_format)
    assert rows.shape == (2, expected_bytes)
    assert row_bytes(dim, cache_format) == expected_bytes
    torch.testing.assert_close(unpack_rows(rows, dim, cache_format), x, atol=0, rtol=0)
    assert torch.all(rows[0, dim if cache_format == "swa" else dim // 2 :] != 0)


def test_fp4_midpoint_rounding_and_rope_tail():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2)
    x[8:] = -x[8:]
    rows = pack_rows(x[None], "main")
    assert rows[0, -1].item() == 56  # E4M3 1.0
    codes = torch.stack((rows[0, :-1] & 15, rows[0, :-1] >> 4), dim=-1).flatten()
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]
    # No special BF16 tail: every channel, including the final RoPE group,
    # follows exactly the same packed quantization contract.
    torch.testing.assert_close(
        unpack_rows(rows, 16, "main", torch.float32),
        torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6, 0, -1, -1, -2, -2, -4, -4, -6.0]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantization_cuda_graph_changed_values():
    x = torch.ones(4, 128, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        pack_rows(x, "main")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = pack_rows(x, "main")
        restored = unpack_rows(packed, 128, "main")
    for value in (0.0, 3.0, -6.0):
        x.fill_(value)
        graph.replay()
        torch.testing.assert_close(restored, x, atol=0, rtol=0)


def _stage_fixture(queries, swa_width, main_width, *, empty_pool=False, extra_width=None):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

    torch.manual_seed(711)
    q = torch.empty(queries, 1, 512, dtype=torch.bfloat16, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(
        q, (main_width or 0) if extra_width is None else extra_width
    )
    rows = 0 if empty_pool else 23
    swa = pack_rows(torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16), "swa")
    main_storage = torch.empty(rows, 356, dtype=torch.uint8, device="cuda")
    main = main_storage[:, :288]
    main.copy_(pack_rows(torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16), "main"))
    if rows:
        # Finite edge payloads include negative zero and the smallest SWA scale.
        swa[0, :512] = 128
        swa[0, 512:] = 0
        swa[0, 0], swa[0, 1] = 56, 184  # E4M3 +1/-1 produce nonzero subnormals.
        main[0, :256] = 136
        main[0, 256:] = 1
    dtype = torch.int32 if main_width is None else torch.int64

    def slots(width):
        backing = torch.zeros(queries, width * 2, dtype=dtype, device="cuda")
        selected = backing[:, ::2]
        selected.copy_(torch.arange(width, device="cuda") % max(rows, 1))
        selected[:, 2::5] = -1
        selected[:, 3::11] = (1 << 40) if dtype == torch.int64 else 10000
        if queries > 1:
            selected[0].fill_(-1)
        return selected

    si = slots(swa_width)
    mi = None if main_width is None else slots(main_width)
    if queries > 1:
        si[1].fill_(-1)  # Main selections must fill the primary pool first.
    return metadata, CSA2BackendForwardArgs(
        swa_pool=swa, swa_indices=si, main_pool=main, topk_indices=mi
    )


def _stage_oracle(metadata, inputs):
    sources = [(inputs.swa_pool, inputs.swa_indices, "swa")]
    if inputs.topk_indices is not None:
        sources.append((inputs.main_pool, inputs.topk_indices, "main"))
    # Independent BF16 decode reference: never the CuTe dequant under test.
    rows = torch.cat([_gather_reference(p, s, 512, fmt) for p, s, fmt in sources], 1)
    valid = torch.cat([(s >= 0) & (s < p.shape[0]) for p, s, _ in sources], 1)
    positions = torch.arange(rows.shape[1], device=rows.device).expand(valid.shape)
    order = torch.where(valid, positions, rows.shape[1]).argsort(dim=1, stable=True)
    packed = rows.gather(1, order[..., None].expand(-1, -1, 512))
    counts = valid.sum(1, dtype=torch.int32)
    packed = torch.where((positions < counts[:, None])[..., None], packed, 0)
    primary, extra = torch.zeros_like(metadata.swa_pool), torch.zeros_like(metadata.extra_pool)
    primary[:, : min(rows.shape[1], 128)] = packed[:, :128]
    if rows.shape[1] > 128:
        extra[:, : rows.shape[1] - 128] = packed[:, 128:]
    lengths = counts.clamp_min(1)
    columns = torch.arange(metadata.num_sparse_topk, device=rows.device)[None, :]
    query = torch.arange(rows.shape[0], device=rows.device)[:, None]
    indices = torch.where(
        columns < 128,
        query * 128 + columns,
        query * (metadata.num_sparse_topk - 128) + columns - 128,
    )
    indices = torch.where(columns < lengths[:, None], indices, -1).int()
    return primary, extra, lengths, indices


def _check_stage(metadata, expected):
    for actual, reference in zip(
        (metadata.swa_pool, metadata.extra_pool, metadata.prepared_lens, metadata.prepared_indices),
        expected,
    ):
        if actual.dtype == torch.bfloat16:
            actual, reference = actual.view(torch.int16), reference.view(torch.int16)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert metadata.prepared_counter.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "queries,swa_width,main_width,empty_pool,extra_width",
    [
        (1, 128, None, False, None),
        (8, 17, 17, False, None),
        (128, 128, 512, False, None),
        (1, 0, 17, False, None),
        (1, 17, 17, True, None),
        (1, 17, 17, False, 4096),
    ],
)
def test_native_staging_exact(queries, swa_width, main_width, empty_pool, extra_width, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    _sm100()
    calls = []
    original = kernel.stage_selected_rows

    def stage(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "stage_selected_rows", stage)
    metadata, inputs = _stage_fixture(
        queries, swa_width, main_width, empty_pool=empty_pool, extra_width=extra_width
    )
    expected = _stage_oracle(metadata, inputs)
    metadata.swa_pool.fill_(13)
    metadata.extra_pool.fill_(17)
    metadata.prepared_counter.fill_(19)
    metadata.stage_selected(inputs)
    _check_stage(metadata, expected)
    assert bool(calls) == (metadata.num_sparse_topk <= 4096)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staging_group_reuse_validates_each_layers_own_slots():
    """A reused compaction still decodes slots the compacting layer never resolved.

    Staging groups key on the KV owner and index source, so they span layers whose
    SWA pool and SWA slots are private to each layer. A column the group's first
    layer resolved can therefore be empty in a later one.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_selected_rows

    _sm100()
    queries, width = 4, 17
    metadata, inputs = _stage_fixture(queries, width, None)
    rows, row_bytes = inputs.swa_pool.shape
    resolved = (
        torch.arange(width, device="cuda", dtype=torch.int64)
        .remainder(rows)[None, :]
        .expand(queries, width)
        .contiguous()
    )
    scratch = {
        "tags": torch.empty(queries, metadata.num_sparse_topk, dtype=torch.int32, device="cuda"),
        "counts": torch.empty(queries, dtype=torch.int32, device="cuda"),
    }
    # The first layer of the group resolves every column, so it tags all of them.
    stage_selected_rows(
        inputs.swa_pool, resolved, inputs.swa_pool, None, metadata, scratch, compact=True
    )
    assert scratch["counts"].tolist() == [width] * queries
    # The later layer's private pool starts one row into a sentinel buffer, so the
    # row an unvalidated negative slot reaches decodes to +1.0 instead of zero.
    sentinel = torch.empty(rows + 1, row_bytes, dtype=torch.uint8, device="cuda")
    sentinel[:, :512], sentinel[:, 512:] = 56, 127  # E4M3 +1.0 with a 2**0 scale
    metadata.swa_pool.fill_(13)
    metadata.extra_pool.fill_(17)
    stage_selected_rows(
        sentinel[1:],
        torch.full((queries, width), -1, device="cuda", dtype=torch.int64),
        sentinel[1:],
        None,
        metadata,
        scratch,
        compact=False,
    )
    # Every tagged column is empty for this layer, so every staged row stays zero.
    assert torch.count_nonzero(metadata.swa_pool) == 0
    assert torch.count_nonzero(metadata.extra_pool) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_staging_graph_refresh():
    _sm100()
    metadata, inputs = _stage_fixture(8, 128, 17)
    metadata.stage_selected(inputs)
    tensor_names = ("swa_pool", "extra_pool", "prepared_indices", "prepared_lens")
    pointers = [getattr(metadata, name).data_ptr() for name in tensor_names]
    retained = metadata.get_workspace_bytes()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.stage_selected(inputs)
    for selection, payload in ((0, 56), (-1, 128), (2, 32), (2, 184)):
        inputs.swa_indices.fill_(selection)
        inputs.topk_indices.fill_(selection)
        inputs.swa_pool[2, :512].fill_(payload)
        inputs.main_pool[2, :256].fill_(payload)
        expected = _stage_oracle(metadata, inputs)
        metadata.swa_pool.fill_(13)
        metadata.extra_pool.fill_(17)
        metadata.prepared_counter.fill_(19)
        graph.replay()
        _check_stage(metadata, expected)
    assert pointers == [getattr(metadata, name).data_ptr() for name in tensor_names]
    assert retained == metadata.get_workspace_bytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_staging_validation_and_fallback(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    _sm100()
    metadata, inputs = _stage_fixture(1, 17, None)
    # An unused malformed main pool is ignored; a supplied empty selection is validated.
    inputs.main_pool = torch.zeros(1, device="cuda")
    metadata.stage_selected(inputs)
    inputs.topk_indices = inputs.swa_indices[:, :0]
    with pytest.raises(ValueError, match="wrong row shape or dtype"):
        metadata.stage_selected(inputs)
    inputs.topk_indices = None
    original_slots = inputs.swa_indices
    backing = torch.empty(inputs.swa_pool.shape[0], 1056, dtype=torch.uint8, device="cuda")
    backing[:, ::2].copy_(inputs.swa_pool)
    inputs.swa_pool = backing[:, ::2]
    expected = _stage_oracle(metadata, inputs)
    monkeypatch.setattr(
        kernel, "stage_selected_rows", lambda *args: pytest.fail("Expected fallback")
    )
    metadata.stage_selected(inputs)
    _check_stage(metadata, expected)
    # The unsupported zero-query direct invocation still resets the counter.
    metadata._num_tokens = 0
    inputs.swa_indices = original_slots[:0]
    metadata.prepared_counter.fill_(19)
    metadata.stage_selected(inputs)
    assert metadata.prepared_counter.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_format,dim", [("index", 128), ("swa", 512)])
@torch.inference_mode()
def test_fused_store_changed_row_counts(cache_format, dim):
    """One fixed cache/stride signature serves changing launch grids exactly."""
    _sm100()
    torch.manual_seed(814)
    maximum = 8384
    storage, pool = _storage(maximum, dim, cache_format)
    values = _values(maximum, dim, cache_format)
    slot_storage = torch.empty(maximum * 2, dtype=torch.int64, device="cuda")
    slots = slot_storage[::2]
    for rows in (0, 1, 4097, 8384):
        slots.copy_(torch.arange(maximum, device="cuda"))
        selected = slots[:rows]
        if cache_format == "swa" and rows > 128:
            selected[:-128] = -1  # Real context cache writes retain only the final window.
        selected[1::31] = -1
        selected[3::37] = 1 << 40
        storage.fill_(77)
        expected = _expected(storage, pool, selected, values[:rows], cache_format)
        quantize_scatter_rows(pool, selected, values[:rows], cache_format)
        torch.testing.assert_close(storage, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dequant,q_dequant", [(1.0, 1.0), (3.0, 2.0)])
@torch.inference_mode()
def test_native_fp8_staging_exact_bytes(dequant, q_dequant, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    control, inputs = _stage_fixture(3, 128, 17)
    # All E4M3 codes, both zero signs, subnormals and NaNs; scale255 also
    # yields infinities after decode. Other rows retain random finite values.
    inputs.swa_pool[2, :512] = torch.arange(512, device="cuda").to(torch.uint8)
    inputs.swa_pool[2, 512:] = torch.tensor(
        [127, 126, 128, 136, 255, 0, 127, 127] * 2, device="cuda", dtype=torch.uint8
    )
    inputs.swa_indices[2, :4] = torch.tensor([2, 2, 0, 1], device="cuda")
    # Main payload/scales include FP4 midpoint reconstruction and saturated
    # E4M3 outputs while retaining the physical356-byte row stride.
    inputs.main_pool[2, :256] = torch.arange(256, device="cuda").to(torch.uint8)
    inputs.main_pool[2, 256:] = torch.tensor(
        [57, 59, 120, 126] * 8, device="cuda", dtype=torch.uint8
    )
    inputs.topk_indices[2, :2] = 2
    expected = _stage_oracle(control, inputs)
    q = torch.zeros(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    dq = torch.tensor([dequant], device="cuda", dtype=torch.float32)
    q_dq = torch.tensor([q_dequant], device="cuda", dtype=torch.float32)
    multiplier = dq.reciprocal()
    calls = []
    original = kernel.stage_selected_rows

    def stage(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "stage_selected_rows", stage)
    metadata.prepared_counter.fill_(19)
    metadata.mla_bmm1_scale.fill_(float("nan"))
    metadata.mla_bmm2_scale.fill_(float("nan"))
    metadata.stage_selected(
        inputs,
        kv_scale_orig_quant=multiplier,
        kv_scale_quant_orig=dq,
        q_scale_quant_orig=q_dq,
        softmax_scale=512**-0.5,
    )
    assert calls == [True]
    for actual, decoded in zip((metadata.swa_pool, metadata.extra_pool), expected[:2]):
        quantized, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded, dq)
        torch.testing.assert_close(
            actual.view(torch.uint8), quantized.view(torch.uint8), atol=0, rtol=0
        )
    torch.testing.assert_close(metadata.prepared_lens, expected[2], atol=0, rtol=0)
    torch.testing.assert_close(metadata.prepared_indices, expected[3], atol=0, rtol=0)
    assert metadata.prepared_counter.item() == 0
    # BMM1 folds the two independent dequant scales, not one shared scale.
    scale = q_dequant * dequant * 512**-0.5
    torch.testing.assert_close(
        metadata.mla_bmm1_scale,
        torch.tensor([scale, scale * 1.4426950408889634], device="cuda"),
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(metadata.mla_bmm2_scale, dq, atol=0, rtol=0)


def _stage_scale_reference(inputs):
    """Independent ceiling on the magnitudes the selected rows can decode to.

    Both persistent encoders clamp their payload -- ``main`` codes at the E2M1
    maximum of 6, ``swa`` values at +-448 -- so a row's group scale alone bounds
    it, whichever way the encoder rounded that scale.
    """
    ceiling = 0.0
    sources = [(inputs.swa_pool, inputs.swa_indices, "swa")]
    if inputs.topk_indices is not None:
        sources.append((inputs.main_pool, inputs.topk_indices, "main"))
    for pool, slots, cache_format in sources:
        groups = 16 if cache_format == "swa" else 32
        valid = (slots >= 0) & (slots < pool.shape[0])
        if not valid.any():
            continue
        observed = pool[torch.where(valid, slots, 0).long()][valid][:, -groups:].flatten()
        if cache_format == "swa":
            # Byte 255 is the infinite scale, which no finite scale can bound.
            finite = observed[observed != 255].int()
            bound = 448.0 * torch.ldexp(torch.ones_like(finite, dtype=torch.float64), finite - 127)
        else:
            # 0x7F is E4M3 NaN; every byte above it encodes a negative scale.
            finite = observed[observed < 127]
            bound = 6.0 * finite.view(torch.float8_e4m3fn).double()
        if bound.numel():
            ceiling = max(ceiling, bound.max().item())
    return ceiling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", ["random", "edges", "absent", "huge"])
@pytest.mark.parametrize("minimum", [-126, 0])
@torch.inference_mode()
def test_stage_kv_scale_derivation(case, minimum):
    """The derived staging scale covers every selected row without saturating."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    _, inputs = _stage_fixture(3, 128, 17)
    if case == "edges":
        # E4M3 NaN and the infinite SWA scale cannot be bounded and must not
        # drive the reduction; the largest finite scales must.
        inputs.swa_pool[2, 512:] = torch.tensor(
            [127, 126, 128, 136, 255, 0, 127, 127] * 2, device="cuda", dtype=torch.uint8
        )
        inputs.main_pool[2, 256:] = torch.tensor(
            [57, 59, 120, 126] * 8, device="cuda", dtype=torch.uint8
        )
        inputs.swa_indices[2, :4] = 2
        inputs.topk_indices[2, :2] = 2
    elif case == "absent":
        inputs.swa_indices.fill_(-1)
        inputs.topk_indices.fill_(-1)
    elif case == "huge":
        inputs.swa_pool[2, 512:].fill_(254)  # 2**127, past every clamp
        inputs.swa_indices[2, 0] = 2
    q = torch.zeros(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    quant, dequant = metadata.derive_stage_kv_scales(inputs, minimum=minimum, maximum=126)
    assert (quant, dequant) == metadata.stage_kv_scales
    scale = dequant.item()
    # A power of two costs no relative precision: it only moves the exponent.
    assert scale > 0.0 and math.log2(scale).is_integer()
    assert quant.item() == 1.0 / scale
    ceiling = _stage_scale_reference(inputs)
    lowest, highest = 2.0**minimum, 2.0**126
    if ceiling == 0.0:
        assert scale == 1.0  # Staging nothing keeps the historical unit scale.
    elif ceiling > 448.0 * highest:
        assert scale == highest
    elif ceiling <= 448.0 * lowest:
        assert scale == lowest
    else:
        # The smallest power of two whose E4M3 range covers the whole ceiling.
        assert 448.0 * scale >= ceiling > 224.0 * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "amax,expected",
    [
        (4.0, 64.0),  # 448 / 4 = 112, so six binades of headroom
        (1.75, 256.0),  # exactly 2**8 of headroom, the boundary mantissa
        (1.8, 128.0),  # just past it, one binade tighter
        (500.0, 1.0),  # Q already saturates: leave the historical unit scale
        (0.0, 2.0**126),  # a zero Q cannot saturate and bounds nothing
    ],
)
@torch.inference_mode()
def test_stage_kv_scale_respects_q_headroom(amax, expected):
    """Q's own range caps the staging scale it has to take the reciprocal of.

    Q carries ``1 / scale`` so that BMM1's dequant product stays one, which the
    closed cubins behind the other two ``MultiCtasKvMode`` values still assume, so
    its codes land at ``scale * amax(|Q|)``: a scale wide enough to rescue the KV
    rows must still leave Q inside the E4M3 range.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    _, inputs = _stage_fixture(3, 128, 17)
    # A group scale of 2**127 outruns every clamp, so only Q can bound this.
    inputs.swa_pool[2, 512:].fill_(254)
    inputs.swa_indices[2, 0] = 2
    q = torch.full((3, 64, 512), amax, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    quant, dequant = metadata.derive_stage_kv_scales(inputs, q=q)
    assert dequant.item() == expected
    assert quant.item() == 1.0 / expected
    if 0.0 < amax <= 448.0:  # A Q that saturates on its own bounds nothing useful.
        assert expected * amax <= 448.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stage_kv_scale_rescues_saturated_rows():
    """Rows decoding past 448 survive staging instead of clamping at the E4M3 max."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    control, inputs = _stage_fixture(4, 128, 17)
    torch.manual_seed(357)
    rows = inputs.swa_pool.shape[0]
    # Both encodings carry their own block scales, so packing large magnitudes
    # is lossless in range; only the unit staging scale used to lose them.
    inputs.swa_pool.copy_(
        pack_rows(torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) * 900, "swa")
    )
    inputs.main_pool.copy_(
        pack_rows(torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) * 900, "main")
    )
    expected = _stage_oracle(control, inputs)
    q = torch.zeros(4, 64, 512, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    unit = metadata.native_unit_scale
    metadata.stage_selected(inputs, kv_scale_orig_quant=unit, kv_scale_quant_orig=unit)
    clamped = [pool.float().clone() for pool in (metadata.swa_pool, metadata.extra_pool)]
    quant, dequant = metadata.derive_stage_kv_scales(inputs)
    assert dequant.item() > 1.0
    metadata.stage_selected(inputs, kv_scale_orig_quant=quant, kv_scale_quant_orig=dequant)
    saturated = 0
    for pool, before, reference in zip((metadata.swa_pool, metadata.extra_pool), clamped, expected):
        over = reference.float().abs() > 448.0
        saturated += int(over.sum())
        # A unit scale clamped these rows at 448, losing every bit of magnitude.
        assert before[over].abs().eq(448.0).all()
        staged = pool.float() * dequant
        # Half an E4M3 step is 2**-4 relative, and the BF16 decode feeding the
        # conversion is finer still, so the range is all that was ever missing.
        torch.testing.assert_close(staged[over], reference.float()[over], atol=0, rtol=0.07)
    assert saturated > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "amax,expected",
    [
        (4.0, 4.0),  # floor(log2(4)) = 2, so Q affords two binades
        (1.0, 1.0),  # already at unit magnitude: nothing left to divide
        (0.5, 1.0),  # below it; the bound goes negative and the unit scale stands
        (2.0**40, 2.0**40),  # a wide Q affords every binade the rows ask for
        (0.0, 2.0**126),  # a zero Q loses nothing and bounds nothing
    ],
)
@torch.inference_mode()
def test_stage_kv_scale_respects_q_resolution(amax, expected):
    """A shared dequant tensor caps the scale by Q's resolution, not its headroom.

    When Q cannot take the reciprocal scale it is divided by the staging scale
    instead, so it can never saturate -- it runs out of resolution underneath. The
    bound therefore holds Q's own amax at unit magnitude or above, which leaves it
    the whole E4M3 normal range below one plus the subnormals under that.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    _, inputs = _stage_fixture(3, 128, 17)
    # A group scale of 2**127 outruns every clamp, so only Q can bound this.
    inputs.swa_pool[2, 512:].fill_(254)
    inputs.swa_indices[2, 0] = 2
    q = torch.full((3, 64, 512), amax, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    quant, dequant = metadata.derive_stage_kv_scales(inputs, q=q, q_shared=True)
    assert dequant.item() == expected
    assert quant.item() == 1.0 / expected
    # A zero Q has no magnitude to keep; otherwise Q stays at unit magnitude, or
    # the historical unit scale was the floor all along.
    assert amax == 0.0 or expected == 1.0 or amax / expected >= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stage_kv_scale_rescues_context_rows():
    """Sharing one dequant tensor still de-saturates the main pool in context.

    The saturation that keeps native FP8 staging opt-in was first observed on image
    tokens at a chunked-prefill boundary, which runs this path. Context cannot give
    Q the reciprocal scale, so Q is divided by the staging scale instead: that costs
    Q resolution underneath but cannot saturate it, so the exponent is bounded by
    Q's amax and by the binades the main pool can need, not pinned at zero.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        _CONTEXT_STAGE_SCALE_MAX_EXPONENT,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    control, inputs = _stage_fixture(4, 128, 17)
    torch.manual_seed(911)
    rows = inputs.main_pool.shape[0]
    # The main pool carries its own block scales, so packing these magnitudes is
    # lossless in range; only the unit staging scale ever lost them. The SWA rows
    # stay as the fixture packed them, well inside E4M3, so the main pool alone
    # drives the exponent here.
    inputs.main_pool.copy_(
        pack_rows(torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) * 900, "main")
    )
    # Pin the ceiling at the main pool's structural maximum: its E4M3 group scale
    # saturates at 448 and its E2M1 codes at 6, so row 0 now decodes to exactly
    # 6 * 448 = 2688 and no row of any main pool can ever ask for more.
    inputs.main_pool[0, 256:].fill_(126)
    expected = _stage_oracle(control, inputs)
    q = torch.full((4, 64, 512), 64.0, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17, staging_dtype=torch.float8_e4m3fn)
    unit = metadata.native_unit_scale
    metadata.stage_selected(inputs, kv_scale_orig_quant=unit, kv_scale_quant_orig=unit)
    clamped = [pool.float().clone() for pool in (metadata.swa_pool, metadata.extra_pool)]
    quant, dequant = metadata.derive_stage_kv_scales(
        inputs, q=q, q_shared=True, maximum=_CONTEXT_STAGE_SCALE_MAX_EXPONENT
    )
    # 2688 mapped onto 448 is three binades, which is exactly the cap: context
    # never has to clip what the main pool asks for.
    assert dequant.item() == 8.0 == 2.0**_CONTEXT_STAGE_SCALE_MAX_EXPONENT
    assert 64.0 / dequant.item() >= 1.0  # Q gave up resolution, not magnitude.
    metadata.stage_selected(inputs, kv_scale_orig_quant=quant, kv_scale_quant_orig=dequant)
    saturated = 0
    for pool, before, reference in zip((metadata.swa_pool, metadata.extra_pool), clamped, expected):
        over = reference.float().abs() > 448.0
        saturated += int(over.sum())
        # A unit scale clamped these rows at 448, losing every bit of magnitude.
        assert before[over].abs().eq(448.0).all()
        staged = pool.float() * dequant
        # Half an E4M3 step is 2**-4 relative, and the BF16 decode feeding the
        # conversion is finer still, so the range is all that was ever missing.
        torch.testing.assert_close(staged[over], reference.float()[over], atol=0, rtol=0.07)
    assert saturated > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_unfused_fp8_staging_matches_fused_bytes(monkeypatch):
    """The eager staging fallback reproduces the fused route byte for byte."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _sm100()
    calls = []
    original = kernel.stage_selected_rows

    def stage(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "stage_selected_rows", stage)
    staged = []
    for extra_width, fused in ((17, True), (4096, False)):
        _, inputs = _stage_fixture(2, 128, 17, extra_width=extra_width)
        # A selected main row at its format's own ceiling, 6 * 448, makes the
        # derivation widen; a unit scale would not tell the routes apart.
        inputs.main_pool[1, 256:].fill_(126)
        calls.clear()
        q = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q, extra_width, staging_dtype=torch.float8_e4m3fn
        )
        quant, dequant = metadata.derive_stage_kv_scales(inputs)
        assert dequant.item() == 8.0  # 2688 mapped onto 448.
        metadata.mla_bmm1_scale.fill_(float("nan"))
        metadata.mla_bmm2_scale.fill_(float("nan"))
        metadata.stage_selected(
            inputs,
            kv_scale_orig_quant=quant,
            kv_scale_quant_orig=dequant,
            q_scale_quant_orig=metadata.native_unit_scale,
            softmax_scale=512**-0.5,
        )
        assert bool(calls) == fused
        # The unfused route must publish the same folded BMM scales.
        scale = dequant * 512**-0.5
        torch.testing.assert_close(
            metadata.mla_bmm1_scale,
            torch.cat((scale, scale * 1.4426950408889634)),
            atol=0,
            rtol=1e-6,
        )
        torch.testing.assert_close(metadata.mla_bmm2_scale, dequant, atol=0, rtol=0)
        staged.append(
            (
                metadata.swa_pool.view(torch.uint8).clone(),
                metadata.extra_pool[:, :17].view(torch.uint8).clone(),
                metadata.prepared_lens.clone(),
            )
        )
    for fused_tensor, unfused in zip(*staged):
        torch.testing.assert_close(unfused, fused_tensor, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_shared_active_rows_preserve_compaction_and_decode_once(dtype):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_shared_rows

    torch.manual_seed(946)
    swa = pack_rows(torch.randn(64, 512, device="cuda", dtype=torch.bfloat16), "swa")
    records = torch.empty(32, 356, device="cuda", dtype=torch.uint8)
    main = records[:, :288]
    main.copy_(pack_rows(torch.randn(32, 512, device="cuda", dtype=torch.bfloat16), "main"))
    swa_values = [[20, -1, 22, 23], [21, 22, 23, 24], [-1] * 4, [37, 38, 39, 40], [20] * 4]
    logical_values = [[0, 0, 2, -1], [1, 3, 4, (1 << 63) - 1], [-1] * 4, [1, 3, 0, -1], [0] * 4]
    main_values = [[8, 8, 10, -1], [9, 11, 12, 9], [-1] * 4, [17, 19, 1 << 40, -1], [8] * 4]

    def strided(values):
        storage = torch.full((len(values), 8), -1, device="cuda", dtype=torch.int64)
        storage[:, ::2].copy_(torch.tensor(values, device="cuda"))
        return storage[:, ::2]

    swa_slots, main_slots, logical = map(strided, (swa_values, main_values, logical_values))
    plan = SimpleNamespace(
        query_requests=torch.tensor([0, 0, 1, 1, -1], device="cuda", dtype=torch.int64),
        query_positions=torch.tensor([7, 8, 23, 24, 7], device="cuda", dtype=torch.int32),
        swa_starts=torch.tensor([4, 20], device="cuda", dtype=torch.int64),
        swa_offsets=torch.tensor([1, 9, 17], device="cuda", dtype=torch.int64),
        main_offsets=torch.tensor([17, 21, 25], device="cuda", dtype=torch.int64),
        swa_pages=torch.tensor([[2, 3, -1, -1], [-1, -1, 4, 5]], device="cuda", dtype=torch.int32),
        main_pages=torch.tensor([[1], [2]], device="cuda", dtype=torch.int32),
        swa_page_size=8,
        main_page_size=8,
        num_requests=2,
        swa_rows=16,
        main_rows=8,
    )
    bank = torch.full((25, 512), 7, device="cuda", dtype=dtype)
    metadata = SimpleNamespace(
        num_tokens=5,
        num_sparse_topk=128,
        shared_pool=bank,
        _csa2_shared_selected=torch.empty(25, device="cuda", dtype=torch.int32),
        prepared_indices=torch.empty(5, 128, device="cuda", dtype=torch.int32),
        prepared_lens=torch.empty(5, device="cuda", dtype=torch.int32),
        prepared_counter=torch.empty(1, device="cuda", dtype=torch.uint32),
        mla_bmm1_scale=torch.empty(2, device="cuda"),
        mla_bmm2_scale=torch.empty(1, device="cuda"),
    )
    unit = torch.ones(1, device="cuda")
    stage_shared_rows(
        swa,
        swa_slots,
        main,
        main_slots,
        logical,
        metadata,
        plan,
        kv_scale_orig_quant=unit,
        kv_scale_quant_orig=unit,
        q_scale_quant_orig=unit,
    )
    expected_ids = [[1, 3, 4, 17, 17, 19], [2, 3, 4, 5, 18, 20], [0], [10, 11, 12, 13, 22, 24], [0]]
    expected_indices = torch.full_like(metadata.prepared_indices, -1)
    for row, ids in enumerate(expected_ids):
        expected_indices[row, : len(ids)] = torch.tensor(ids, device="cuda")
    torch.testing.assert_close(metadata.prepared_indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(
        metadata.prepared_lens,
        torch.tensor([len(ids) for ids in expected_ids], device="cuda", dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    selected_rows = sorted({index for ids in expected_ids for index in ids if index})
    expected_mask = torch.zeros_like(metadata._csa2_shared_selected)
    expected_mask[selected_rows] = 1
    torch.testing.assert_close(metadata._csa2_shared_selected, expected_mask, atol=0, rtol=0)
    expected_bank = torch.full((25, 512), 7, device="cuda", dtype=torch.bfloat16)
    expected_bank[0].zero_()
    decoded_swa, decoded_main = unpack_rows(swa, 512, "swa"), unpack_rows(main, 512, "main")
    for row in selected_rows:
        if row < 17:
            request = int(row >= 9)
            position = [4, 20][request] + row - [1, 9][request]
            page = [[2, 3, -1, -1], [-1, -1, 4, 5]][request][position // 8]
            expected_bank[row].copy_(decoded_swa[page * 8 + position % 8])
        else:
            request = int(row >= 21)
            slot = [8, 16][request] + row - [17, 21][request]
            expected_bank[row].copy_(decoded_main[slot])
    if dtype == torch.float8_e4m3fn:
        expected_bank = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(expected_bank, unit)[
            0
        ]
    torch.testing.assert_close(
        bank.view(torch.uint8), expected_bank.view(torch.uint8), atol=0, rtol=0
    )
    # A second, empty publication clears live indices/masks and the dummy row,
    # without reading or rewriting any unselected source row.
    previous = bank.clone()
    swa_slots.fill_(-1)
    main_slots.fill_(-1)
    stage_shared_rows(
        swa,
        swa_slots,
        main,
        main_slots,
        logical,
        metadata,
        plan,
        kv_scale_orig_quant=unit,
        kv_scale_quant_orig=unit,
        q_scale_quant_orig=unit,
    )
    assert not metadata._csa2_shared_selected.any()
    assert (metadata.prepared_lens == 1).all()
    assert (metadata.prepared_indices[:, 0] == 0).all()
    assert (metadata.prepared_indices[:, 1:] == -1).all()
    torch.testing.assert_close(bank.view(torch.uint8), previous.view(torch.uint8), atol=0, rtol=0)
    # Empty request domains and no MAIN pool use the same offset upper bound.
    plan.swa_offsets = torch.tensor([1, 1, 9], device="cuda", dtype=torch.int64)
    plan.main_offsets = torch.tensor([9, 9, 9], device="cuda", dtype=torch.int64)
    plan.swa_rows, plan.main_rows, plan.main_pages = 8, 0, None
    metadata.shared_pool = torch.full((9, 512), 7, device="cuda", dtype=dtype)
    metadata._csa2_shared_selected = torch.empty(9, device="cuda", dtype=torch.int32)
    swa_slots.copy_(torch.tensor(swa_values, device="cuda"))
    stage_shared_rows(
        swa,
        swa_slots,
        None,
        None,
        None,
        metadata,
        plan,
        kv_scale_orig_quant=unit,
        kv_scale_quant_orig=unit,
        q_scale_quant_orig=unit,
    )
    expected_indices.fill_(-1)
    expected_indices[:, 0] = 0
    expected_indices[3, :4] = torch.tensor([2, 3, 4, 5], device="cuda")
    torch.testing.assert_close(metadata.prepared_indices, expected_indices, atol=0, rtol=0)
    assert metadata._csa2_shared_selected.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert metadata.prepared_lens.tolist() == [1, 1, 1, 4, 1]
    expected_bank = torch.full((9, 512), 7, device="cuda", dtype=torch.bfloat16)
    expected_bank[0].zero_()
    expected_bank[2:6].copy_(decoded_swa[37:41])
    if dtype == torch.float8_e4m3fn:
        expected_bank = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(expected_bank, unit)[
            0
        ]
    torch.testing.assert_close(
        metadata.shared_pool.view(torch.uint8), expected_bank.view(torch.uint8), atol=0, rtol=0
    )


# ----------------------------------------------------------------------------
# Decode glue kernels against their PyTorch references in metadata.py/indexer.py
# ----------------------------------------------------------------------------


def _glue():
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    if not kernel.dsl_available():
        pytest.skip("CSA2 decode glue kernels require the CuTe DSL")
    return kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_owner_refresh_kernel_matches_reference():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    kernel = _glue()
    torch.manual_seed(1201)
    request_count = 3
    lengths = torch.tensor([3, 2, 2], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([140, 300, 905], dtype=torch.int32, device="cuda")
    query_base = torch.tensor([137, 298, 903], dtype=torch.int32, device="cuda")
    # One launch serves a ratio-2 owner with compression batch descriptors and
    # a ratio-4 owner without them, each with its own sources and capacity.
    owners = []
    for ratio, source_base, source_lengths, capacity, with_batch in (
        (2, [137, 298, 900], [3, 0, 5], 8, True),
        (4, [100, 290, 880], [40, 10, 25], 20, False),
        # No output rows: only the compression batch descriptors are written.
        (2, [130, 290, 880], [0, 0, 0], 0, True),
    ):
        batch = (
            [torch.empty(request_count, dtype=torch.int32, device="cuda") for _ in range(2)]
            + [torch.empty(request_count + 1, dtype=torch.int32, device="cuda")]
            if with_batch
            else [None] * 3
        )
        owners.append(
            (
                torch.tensor(source_base, dtype=torch.int32, device="cuda"),
                torch.tensor(source_lengths, dtype=torch.int32, device="cuda"),
                torch.randint(-1, 9, (request_count, 8), dtype=torch.int32, device="cuda"),
                torch.empty(capacity, dtype=torch.int64, device="cuda"),
                torch.empty(capacity, dtype=torch.int32, device="cuda"),
                ratio,
                128 // ratio,
                *batch,
            )
        )
    descriptors = torch.tensor(
        [kernel.owner_slot_descriptor(*owner) for owner in owners],
        dtype=torch.int64,
        device="cuda",
    )
    kernel.refresh_owner_slots(kv_lens, lengths, query_base, descriptors, request_count, 20)
    delta = kv_lens[:request_count] - lengths - query_base
    for source_base, source_lengths, pages, slots, compressed, ratio, page_size, *batch in owners:
        outputs = [t for t in (slots, compressed, *batch) if t is not None]
        expected = [torch.empty_like(t) for t in outputs]
        CSA2TrtllmMetadata._refresh_owner_slots(
            source_base,
            source_lengths,
            delta,
            ratio,
            page_size,
            pages,
            *expected[:2],
            *(expected[2:] or [None] * 3),
        )
        for actual, reference_value in zip(outputs, expected):
            torch.testing.assert_close(actual, reference_value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("top_k", [4, 512])
@torch.inference_mode()
def test_selection_glue_kernels_match_reference(top_k):
    kernel = _glue()
    torch.manual_seed(1202)
    count, width, cwidth = 3, 700, 96
    logits = torch.randn(count, width, device="cuda")
    valid = torch.rand(count, width + 8, device="cuda") > 0.2
    visible = torch.tensor([700, 65, 0], dtype=torch.int32, device="cuda")
    # Masking unbacked positions in place.
    expected = logits.masked_fill(~valid[:, :width], -torch.inf)
    kernel.mask_logits_(logits, valid)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    # Candidate-ordered scores: out-of-range, invisible and unbacked candidates are -inf / -1.
    candidates = torch.randint(-1, width + 3, (count, cwidth), dtype=torch.int64, device="cuda")
    scores = torch.empty(count, cwidth, device="cuda")
    positions = torch.empty(count, cwidth, dtype=torch.int32, device="cuda")
    kernel.gather_candidate_scores(logits, valid, candidates, visible, scores, positions)
    ok = (candidates >= 0) & (candidates < width) & (candidates < visible[:, None].long())
    safe = candidates.clamp(0, width - 1)
    ok &= valid[:, :width].gather(1, safe)
    torch.testing.assert_close(scores, torch.where(ok, logits.gather(1, safe), -torch.inf))
    torch.testing.assert_close(positions, torch.where(ok, candidates, -1).int(), atol=0, rtol=0)
    # Finalize: paged mode validates offsets against ``valid``; mapped mode maps
    # offsets through candidate positions and drops -inf scores. Both sort with -1 last.
    indices = torch.randint(-1, width + 5, (count, top_k), dtype=torch.int32, device="cuda")
    paged = indices.clone()
    kernel.finalize_selection_(paged, width, valid=valid)
    keys = (
        torch.where(
            (indices >= 0)
            & (indices < width)
            & valid[:, :width].gather(1, indices.clamp(0, width - 1)),
            indices.long(),
            torch.iinfo(torch.int64).max,
        )
        .sort(-1)
        .values
    )
    torch.testing.assert_close(
        paged, torch.where(keys == torch.iinfo(torch.int64).max, -1, keys).int(), atol=0, rtol=0
    )
    mapped = torch.randint(-1, cwidth + 5, (count, top_k), dtype=torch.int32, device="cuda")
    original = mapped.clone()
    kernel.finalize_selection_(mapped, cwidth, scores=scores, positions=positions)
    safe = original.clamp(0, cwidth - 1)
    ok = (original >= 0) & (original < cwidth) & (scores.gather(1, safe) > -torch.inf)
    keys = (
        torch.where(ok, positions.gather(1, safe).long(), torch.iinfo(torch.int64).max)
        .sort(-1)
        .values
    )
    torch.testing.assert_close(
        mapped, torch.where(keys == torch.iinfo(torch.int64).max, -1, keys).int(), atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_shared_active_rows_heterogeneous_request_domains(dtype: torch.dtype) -> None:
    """Check request ownership across empty domains, page holes and strided tables."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_shared_rows

    torch.manual_seed(947)
    requests, page_size, page_columns, swa_width = 16, 8, 6, 4
    swa_counts = [0, 0, 3, 9, 1, 0, 17, 5, 0, 0, 12, 2, 7, 15, 0, 0]
    main_counts = [0, 4, 0, 11, 1, 0, 0, 19, 8, 0, 2, 13, 0, 5, 0, 0]
    starts = [(request * 3) % 17 for request in range(requests)]
    swa_offsets = [1]
    for count in swa_counts:
        swa_offsets.append(swa_offsets[-1] + count)
    main_offsets = [swa_offsets[-1]]
    for count in main_counts:
        main_offsets.append(main_offsets[-1] + count)
    pages = [
        [
            ((request * page_columns + column) * 7) % (requests * page_columns)
            for column in range(page_columns)
        ]
        for request in range(requests)
    ]
    # Valid selected physical slots may still point into a missing plan page;
    # those shared rows must decode to zero instead of touching an invalid page.
    pages[3][1] = pages[6][1] = pages[7][1] = -1
    pool_rows = requests * page_columns * page_size
    swa = pack_rows(torch.randn(pool_rows, 512, device="cuda", dtype=torch.bfloat16), "swa")
    records = torch.empty(pool_rows, 356, device="cuda", dtype=torch.uint8)
    main = records[:, :288]
    main.copy_(pack_rows(torch.randn(pool_rows, 512, device="cuda", dtype=torch.bfloat16), "main"))
    query_requests, positions, swa_slots, main_slots, logical_rows, expected_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for request in reversed(range(requests)):
        for displacement in (0, swa_counts[request] - 1, swa_counts[request] // 2):
            position = starts[request] + displacement
            query_requests.append(request)
            positions.append(position)
            selected_swa = []
            ids = []
            for column in range(swa_width):
                logical = position - swa_width + 1 + column
                page = pages[request][logical // page_size] if logical >= 0 else -1
                slot = page * page_size + logical % page_size if page >= 0 else 0
                if column == 1 and request % 3 == 0:
                    slot = 1 << 40
                selected_swa.append(slot)
                if (
                    0 <= slot < pool_rows
                    and starts[request] <= logical < starts[request] + swa_counts[request]
                ):
                    ids.append(swa_offsets[request] + logical - starts[request])
            logical = [
                0,
                0,
                main_counts[request] - 1,
                main_counts[request] // 2,
                -1,
                main_counts[request],
                (1 << 63) - 1,
            ]
            selected_main = []
            for column, row in enumerate(logical):
                page = (
                    pages[request][row // page_size] if 0 <= row < page_columns * page_size else -1
                )
                slot = page * page_size + row % page_size if page >= 0 else 0
                if column == 3 and request % 4 == 0:
                    slot = -1
                selected_main.append(slot)
                if 0 <= slot < pool_rows and 0 <= row < main_counts[request]:
                    ids.append(main_offsets[request] + row)
            swa_slots.append(selected_swa)
            main_slots.append(selected_main)
            logical_rows.append(logical)
            expected_ids.append(ids or [0])

    def strided(values: list[list[int]], dtype: torch.dtype = torch.int64) -> torch.Tensor:
        storage = torch.full((len(values), len(values[0]) * 2), -1, device="cuda", dtype=dtype)
        storage[:, ::2].copy_(torch.tensor(values, device="cuda", dtype=dtype))
        return storage[:, ::2]

    def strided_pages(values: list[list[int]]) -> torch.Tensor:
        storage = torch.full(
            (len(values) * 2, len(values[0])), -1, device="cuda", dtype=torch.int32
        )
        storage[::2].copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        return storage[::2]

    plan = SimpleNamespace(
        query_requests=torch.tensor(query_requests, device="cuda", dtype=torch.int64),
        query_positions=torch.tensor(positions, device="cuda", dtype=torch.int32),
        swa_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
        swa_offsets=torch.tensor(swa_offsets, device="cuda", dtype=torch.int64),
        main_offsets=torch.tensor(main_offsets, device="cuda", dtype=torch.int64),
        swa_pages=strided_pages(pages),
        main_pages=strided_pages(pages),
        swa_page_size=page_size,
        main_page_size=page_size,
        num_requests=requests,
        swa_rows=sum(swa_counts),
        main_rows=sum(main_counts),
    )
    bank = torch.full((main_offsets[-1], 512), 7, device="cuda", dtype=dtype)
    metadata = SimpleNamespace(
        num_tokens=len(query_requests),
        num_sparse_topk=128,
        shared_pool=bank,
        _csa2_shared_selected=torch.empty(bank.shape[0], device="cuda", dtype=torch.int32),
        prepared_indices=torch.empty(len(query_requests), 128, device="cuda", dtype=torch.int32),
        prepared_lens=torch.empty(len(query_requests), device="cuda", dtype=torch.int32),
        prepared_counter=torch.empty(1, device="cuda", dtype=torch.uint32),
        mla_bmm1_scale=torch.empty(2, device="cuda"),
        mla_bmm2_scale=torch.empty(1, device="cuda"),
    )
    dequant_scale = torch.tensor([0.3], device="cuda", dtype=torch.float32)
    stage_shared_rows(
        swa,
        strided(swa_slots),
        main,
        strided(main_slots),
        strided(logical_rows),
        metadata,
        plan,
        kv_scale_orig_quant=dequant_scale.reciprocal(),
        kv_scale_quant_orig=dequant_scale,
        q_scale_quant_orig=dequant_scale,
    )
    expected_indices = torch.full_like(metadata.prepared_indices, -1)
    for query, ids in enumerate(expected_ids):
        expected_indices[query, : len(ids)] = torch.tensor(ids, device="cuda")
    torch.testing.assert_close(metadata.prepared_indices, expected_indices, atol=0, rtol=0)
    assert metadata.prepared_lens.tolist() == [len(ids) for ids in expected_ids]
    selected_rows = {row for ids in expected_ids for row in ids if row}
    expected_selected = torch.zeros_like(metadata._csa2_shared_selected)
    expected_selected[list(selected_rows)] = 1
    torch.testing.assert_close(metadata._csa2_shared_selected, expected_selected, atol=0, rtol=0)
    # Build the numerical oracle request by request, without a flat-row owner lookup.
    expected_bank = torch.full_like(bank, 7)
    expected_bank[0].zero_()
    for packed, offsets, counts, fmt in (
        (swa, swa_offsets, swa_counts, "swa"),
        (main, main_offsets, main_counts, "main"),
    ):
        decoded = unpack_rows(packed, 512, fmt)
        for request, count in enumerate(counts):
            for logical in range(count):
                bank_row = offsets[request] + logical
                if bank_row not in selected_rows:
                    continue
                position = logical + (starts[request] if fmt == "swa" else 0)
                page = pages[request][position // page_size]
                value = (
                    decoded[page * page_size + position % page_size].clone()
                    if page >= 0
                    else torch.zeros(512, device="cuda", dtype=torch.bfloat16)
                )
                if dtype == torch.float8_e4m3fn:
                    value = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                        value.unsqueeze(0), dequant_scale
                    )[0][0]
                expected_bank[bank_row].copy_(value)
    torch.testing.assert_close(
        bank.view(torch.uint8), expected_bank.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_shared_active_rows_reuse_request_count_specialization(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing active request counts preserves results and reuses compiled kernels."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    # Isolate the CuTe compilation cache so this test proves that request
    # count changes reuse the same compiled staging functions.
    monkeypatch.setattr(kernel, "_COMPILED", {})
    compiled_staging = None
    torch.manual_seed(953)
    max_requests, domain_rows, page_size, queries = 32, 96, 8, 5
    page_columns = domain_rows // page_size
    pool_rows = max_requests * domain_rows
    swa = pack_rows(torch.randn(pool_rows, 512, device="cuda", dtype=torch.bfloat16), "swa")
    records = torch.empty(pool_rows, 356, device="cuda", dtype=torch.uint8)
    main = records[:, :288]
    main.copy_(pack_rows(torch.randn(pool_rows, 512, device="cuda", dtype=torch.bfloat16), "main"))
    decoded_swa, decoded_main = unpack_rows(swa, 512, "swa"), unpack_rows(main, 512, "main")
    pages = torch.arange(max_requests * page_columns, device="cuda", dtype=torch.int32).reshape(
        max_requests, page_columns
    )
    plan = SimpleNamespace(
        query_requests=torch.empty(queries, device="cuda", dtype=torch.int64),
        query_positions=torch.empty(queries, device="cuda", dtype=torch.int32),
        swa_starts=torch.zeros(max_requests, device="cuda", dtype=torch.int64),
        swa_offsets=torch.empty(max_requests + 1, device="cuda", dtype=torch.int64),
        main_offsets=torch.empty(max_requests + 1, device="cuda", dtype=torch.int64),
        swa_pages=pages,
        main_pages=pages,
        swa_page_size=page_size,
        main_page_size=page_size,
        num_requests=1,
        swa_rows=domain_rows,
        main_rows=domain_rows,
    )
    bank = torch.empty(1 + 2 * domain_rows, 512, device="cuda", dtype=dtype)
    metadata = SimpleNamespace(
        num_tokens=queries,
        num_sparse_topk=128,
        shared_pool=bank,
        _csa2_shared_selected=torch.empty(bank.shape[0], device="cuda", dtype=torch.int32),
        prepared_indices=torch.empty(queries, 128, device="cuda", dtype=torch.int32),
        prepared_lens=torch.empty(queries, device="cuda", dtype=torch.int32),
        prepared_counter=torch.empty(1, device="cuda", dtype=torch.uint32),
        mla_bmm1_scale=torch.empty(2, device="cuda"),
        mla_bmm2_scale=torch.empty(1, device="cuda"),
    )
    swa_slots = torch.empty(queries, 1, device="cuda", dtype=torch.int64)
    main_slots = torch.empty(queries, 2, device="cuda", dtype=torch.int64)
    main_logical = torch.empty_like(main_slots)
    dequant_scale = torch.tensor([0.3], device="cuda", dtype=torch.float32)
    quant_scale = dequant_scale.reciprocal()
    # Fixed pools, table strides, pointers and bank sizes isolate specialization
    # by count, including singleton and larger active request sets.
    for request_count in (1, 3, 6, 8, 16, 32):
        width = domain_rows // request_count
        owners = [0, request_count - 1, request_count // 2, -1, request_count]
        positions = [0, width - 1, width // 2, 0, 0]
        plan.num_requests = request_count
        plan.query_requests.copy_(torch.tensor(owners, device="cuda"))
        plan.query_positions.copy_(torch.tensor(positions, device="cuda"))
        offsets = [min(request, request_count) * width for request in range(max_requests + 1)]
        plan.swa_offsets.copy_(torch.tensor([1 + offset for offset in offsets], device="cuda"))
        plan.main_offsets.copy_(
            torch.tensor([1 + domain_rows + offset for offset in offsets], device="cuda")
        )
        swa_slots.zero_()
        main_slots.zero_()
        main_logical[:, 0].zero_()
        main_logical[:, 1].fill_(width - 1)
        expected_ids = []
        expected_rows = {}
        for query, (owner, position) in enumerate(zip(owners, positions)):
            if 0 <= owner < request_count:
                swa_slot = owner * domain_rows + position
                main_first, main_last = owner * domain_rows, owner * domain_rows + width - 1
                swa_slots[query, 0] = swa_slot
                main_slots[query, 0], main_slots[query, 1] = main_first, main_last
                ids = [
                    1 + owner * width + position,
                    1 + domain_rows + owner * width,
                    1 + domain_rows + owner * width + width - 1,
                ]
                expected_rows[ids[0]] = decoded_swa[swa_slot]
                expected_rows[ids[1]] = decoded_main[main_first]
                expected_rows[ids[2]] = decoded_main[main_last]
                expected_ids.append(ids)
            else:
                expected_ids.append([0])
        bank.fill_(7)
        kernel.stage_shared_rows(
            swa,
            swa_slots,
            main,
            main_slots,
            main_logical,
            metadata,
            plan,
            kv_scale_orig_quant=quant_scale,
            kv_scale_quant_orig=dequant_scale,
            q_scale_quant_orig=dequant_scale,
        )
        staging = {
            key: value
            for key, value in kernel._COMPILED.items()
            if key[0][0] in ("shared_compact", "shared_decode")
        }
        assert sum(key[0][0] == "shared_compact" for key in staging) == 1
        assert sum(key[0][0] == "shared_decode" for key in staging) == 2  # SWA and MAIN.
        if compiled_staging is None:
            compiled_staging = staging
        else:
            assert staging.keys() == compiled_staging.keys()
            assert all(value is compiled_staging[key] for key, value in staging.items())
        expected_indices = torch.full_like(metadata.prepared_indices, -1)
        for query, ids in enumerate(expected_ids):
            expected_indices[query, : len(ids)] = torch.tensor(ids, device="cuda")
        torch.testing.assert_close(metadata.prepared_indices, expected_indices, atol=0, rtol=0)
        assert metadata.prepared_lens.tolist() == [len(ids) for ids in expected_ids]
        selected_rows = sorted(expected_rows)
        expected_selected = torch.zeros_like(metadata._csa2_shared_selected)
        expected_selected[selected_rows] = 1
        torch.testing.assert_close(
            metadata._csa2_shared_selected, expected_selected, atol=0, rtol=0
        )
        expected_bank = torch.full_like(bank, 7)
        expected_bank[0].zero_()
        values = torch.stack([expected_rows[row] for row in selected_rows])
        if dtype == torch.float8_e4m3fn:
            values = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(values, dequant_scale)[
                0
            ]
        expected_bank.view(torch.uint8)[selected_rows] = values.view(torch.uint8)
        torch.testing.assert_close(
            bank.view(torch.uint8), expected_bank.view(torch.uint8), atol=0, rtol=0
        )
