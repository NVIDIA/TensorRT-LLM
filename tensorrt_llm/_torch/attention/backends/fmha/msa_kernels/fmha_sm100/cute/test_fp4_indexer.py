# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import argparse
import contextlib
import io
import statistics

import pytest
import torch

from fp4_indexer_interface import (
    fp4_indexer_block_scores,
    fp4_indexer_mma_scale_shape,
    fp4_indexer_mma_scale_storage_shape,
    fp4_indexer_mma_scale_storage_stride,
    fp4_indexer_mma_scale_stride,
    fp4_indexer_reorder_scales_for_mma_cute,
)
from src.sm100.fp4_indexer import normalize_fp4_format


def _has_sm100_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


_FP4_E2M1_LUT = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)


def _ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _exclusive_prefix(values: list[int]) -> list[int]:
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + int(value))
    return prefix


def _repeat_lengths(long_len: int, long_count: int, short_len: int, short_count: int) -> list[int]:
    return [int(long_len)] * int(long_count) + [int(short_len)] * int(short_count)


_CUSTOM_BENCHMARK_DEFAULT_SQ = 4096
_CUSTOM_BENCHMARK_DEFAULT_SKV = 4096
_BENCHMARK_HEAD_KV = 4
_BENCHMARK_QHEAD_PER_KV = 16
_BENCHMARK_HEAD_DIM = 128
_BENCHMARK_BLK_KV = 128
_DECODE_BATCH = 30
_DECODE_Q_LEN = 8
_SCORE_ATOL = 1.0e-1
_SCORE_RTOL = 1.0e-2
_DEFAULT_BENCHMARK_CASES = (
    {
        "name": "prefill_q8k_k8k",
        "q_lengths": [8192],
        "k_lengths": [8192],
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "prefill_q8k_k64k",
        "q_lengths": [8192],
        "k_lengths": [65536],
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "decode_uniform",
        "q_lengths": [_DECODE_Q_LEN] * _DECODE_BATCH,
        "k_lengths": [67584] * _DECODE_BATCH,
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "decode_1x2x",
        "q_lengths": [_DECODE_Q_LEN] * _DECODE_BATCH,
        "k_lengths": _repeat_lengths(135168, 1, 65253, 29),
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "decode_5x2x",
        "q_lengths": [_DECODE_Q_LEN] * _DECODE_BATCH,
        "k_lengths": _repeat_lengths(135168, 5, 54067, 25),
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "decode_1x3x",
        "q_lengths": [_DECODE_Q_LEN] * _DECODE_BATCH,
        "k_lengths": _repeat_lengths(202752, 1, 62923, 29),
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
    {
        "name": "decode_1x4x",
        "q_lengths": [_DECODE_Q_LEN] * _DECODE_BATCH,
        "k_lengths": _repeat_lengths(270336, 1, 60592, 29),
        "causal": True,
        "head_kv": _BENCHMARK_HEAD_KV,
        "qhead_per_kv": _BENCHMARK_QHEAD_PER_KV,
    },
)


def _length_summary(values: list[int]) -> str:
    if not values:
        return "[]"
    parts: list[str] = []
    current = int(values[0])
    count = 1
    for value in values[1:]:
        value = int(value)
        if value == current:
            count += 1
        else:
            parts.append(f"{count}x{current}" if count > 1 else str(current))
            current = value
            count = 1
    parts.append(f"{count}x{current}" if count > 1 else str(current))
    return " + ".join(parts)


def _decode_fp4x2_to_f32(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8).cpu().to(torch.int64)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    lo_f32 = _FP4_E2M1_LUT[lo]
    hi_f32 = _FP4_E2M1_LUT[hi]
    return torch.stack((lo_f32, hi_f32), dim=-1).reshape(*packed.shape[:-1], 128)


def _dequantize_public_fp4(packed: torch.Tensor, scale: torch.Tensor, *, fmt: str) -> torch.Tensor:
    spec = normalize_fp4_format(fmt)
    values = _decode_fp4x2_to_f32(packed)
    scale_f32 = scale.cpu().to(torch.float32)
    scale_f32 = torch.repeat_interleave(scale_f32, spec.sf_vec_size, dim=-1)[..., :128]
    return values * scale_f32


def _random_lengths(batch: int, max_seqlen: int, generator: torch.Generator) -> list[int]:
    min_len = max(1, max_seqlen // 2)
    lengths = torch.randint(min_len, max_seqlen + 1, (batch,), generator=generator)
    lengths[0] = max_seqlen
    if batch > 1:
        lengths[-1] = max(1, max_seqlen - max(1, max_seqlen // 3))
    return [int(v) for v in lengths.tolist()]


def _scale_raw_choices(fmt: str, *, device: torch.device | None = None) -> torch.Tensor:
    spec = normalize_fp4_format(fmt)
    if spec.name == "mxfp4":
        return torch.arange(122, 133, dtype=torch.uint8, device=device)
    return torch.arange(16, 89, dtype=torch.uint8, device=device)


def _random_scale(
    shape: tuple[int, ...],
    *,
    fmt: str,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    spec = normalize_fp4_format(fmt)
    raw_choices = _scale_raw_choices(spec.name)
    choice_idx = torch.randint(len(raw_choices), shape, generator=generator)
    raw = raw_choices[choice_idx].contiguous().to(device=device)
    return raw.view(spec.torch_scale_dtype)


def _random_scale_cuda(
    shape: tuple[int, ...],
    *,
    fmt: str,
    device: torch.device,
) -> torch.Tensor:
    spec = normalize_fp4_format(fmt)
    raw_choices = _scale_raw_choices(spec.name, device=device)
    choice_idx = torch.randint(len(raw_choices), shape, dtype=torch.int64, device=device)
    raw = raw_choices[choice_idx].contiguous()
    return raw.view(spec.torch_scale_dtype)


def _make_random_score_case(
    *,
    fmt: str,
    batch: int,
    max_seqlen: int,
    heads_q: int,
    heads_k: int,
    seed: int,
) -> dict[str, torch.Tensor | int]:
    if heads_q % heads_k != 0:
        raise ValueError("heads_q must be divisible by heads_k")
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(seed)
    q_lengths = _random_lengths(batch, max_seqlen, generator)
    k_lengths = _random_lengths(batch, max_seqlen, generator)
    total_q = sum(q_lengths)
    pages_per_batch = [_ceil_div(k_len, 128) for k_len in k_lengths]
    page_count = sum(pages_per_batch)

    q = torch.randint(
        0,
        256,
        (total_q, heads_q, 64),
        dtype=torch.uint8,
        generator=generator,
    ).to(device=device)
    k = torch.randint(
        0,
        256,
        (page_count, heads_k, 128, 64),
        dtype=torch.uint8,
        generator=generator,
    ).to(device=device)
    q_scale = _random_scale(
        (total_q, heads_q, normalize_fp4_format(fmt).scale_groups),
        fmt=fmt,
        device=device,
        generator=generator,
    )
    k_scale = _random_scale(
        (page_count, heads_k, 128, normalize_fp4_format(fmt).scale_groups),
        fmt=fmt,
        device=device,
        generator=generator,
    )

    page_order = torch.randperm(page_count, generator=generator, dtype=torch.int32)
    return {
        "q": q,
        "k": k,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "cu_seqlens_q": torch.tensor(_exclusive_prefix(q_lengths), dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.tensor(_exclusive_prefix(k_lengths), dtype=torch.int32, device=device),
        "cu_page_offsets": torch.tensor(_exclusive_prefix(pages_per_batch), dtype=torch.int32, device=device),
        "kv_indices": page_order.to(device=device),
        "max_seqlen": max_seqlen,
    }


def _reference_block_scores(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    fmt: str,
    kv_indices: torch.Tensor,
    causal: bool = False,
    qo_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    if qo_offset is not None and not causal:
        raise ValueError("qo_offset is only valid when causal=True")
    q = _dequantize_public_fp4(q_fp4, q_scale, fmt=fmt)
    k = _dequantize_public_fp4(k_fp4, k_scale, fmt=fmt)
    q_prefix = cu_seqlens_q.cpu()
    k_prefix = cu_seqlens_k.cpu()
    page_prefix = cu_page_offsets.cpu()
    batch = int(q_prefix.shape[0]) - 1
    total_q, heads_q, _ = q.shape
    page_count, heads_k, page_size, _ = k.shape
    assert page_size == 128
    qhead_per_kv = heads_q // heads_k
    max_k_tiles = max(_ceil_div(int(k_prefix[b + 1].item() - k_prefix[b].item()), 128) for b in range(batch))
    kv_indices_cpu = kv_indices.cpu()
    qo_offset_cpu = qo_offset.cpu() if qo_offset is not None else None
    scores = torch.full((heads_q, max_k_tiles, total_q), float("-inf"), dtype=torch.float32)

    for b in range(batch):
        q_begin = int(q_prefix[b].item())
        q_len = int(q_prefix[b + 1].item() - q_prefix[b].item())
        k_len = int(k_prefix[b + 1].item() - k_prefix[b].item())
        page_cursor = int(page_prefix[b].item())
        offset = int(qo_offset_cpu[b].item()) if qo_offset_cpu is not None else k_len - q_len
        for hq in range(heads_q):
            hk = hq // qhead_per_kv
            for q_local in range(q_len):
                q_abs = q_begin + q_local
                for ktile in range(_ceil_div(k_len, 128)):
                    k_start = ktile * 128
                    k_end = min(k_start + 128, k_len)
                    physical_page = int(kv_indices_cpu[page_cursor + ktile].item())
                    assert 0 <= physical_page < page_count
                    logits = k[physical_page, hk, : k_end - k_start] @ q[q_abs, hq]
                    if causal:
                        k_local = torch.arange(k_start, k_end)
                        visible = q_local + offset >= k_local
                        if not bool(visible.any()):
                            continue
                        logits = logits.masked_fill(~visible, float("-inf"))
                    scores[hq, ktile, q_abs] = logits.max()
    return scores


def _fp8_byte(value: torch.Tensor) -> int:
    return int(value.reshape(()).view(torch.uint8).item())


def _mma_scale_view_to_storage(scale: torch.Tensor) -> torch.Tensor:
    return scale.permute(5, 2, 4, 0, 1, 3)


@pytest.mark.skipif(not _has_sm100_cuda(), reason="SM100-class CUDA device required")
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_reorder_scales_for_mma_matches_public_layout(fmt):
    device = torch.device("cuda")
    spec = normalize_fp4_format(fmt)
    total_q = 257
    heads_q = 3
    page_count = 5
    heads_k = 2
    q_scale = _random_scale_cuda((total_q, heads_q, spec.scale_groups), fmt=fmt, device=device)
    k_scale = _random_scale_cuda((page_count, heads_k, 128, spec.scale_groups), fmt=fmt, device=device)

    q_mma, k_mma = fp4_indexer_reorder_scales_for_mma_cute(q_scale, k_scale, fp4_format=fmt)
    torch.cuda.synchronize()

    assert tuple(q_mma.shape) == fp4_indexer_mma_scale_shape(total_q, heads_q, fp4_format=fmt)
    assert tuple(k_mma.shape) == fp4_indexer_mma_scale_shape(128, page_count * heads_k, fp4_format=fmt)
    assert tuple(q_mma.stride()) == fp4_indexer_mma_scale_stride(total_q, heads_q, fp4_format=fmt)
    assert tuple(k_mma.stride()) == fp4_indexer_mma_scale_stride(128, page_count * heads_k, fp4_format=fmt)
    q_storage = _mma_scale_view_to_storage(q_mma)
    k_storage = _mma_scale_view_to_storage(k_mma)
    assert tuple(q_storage.shape) == fp4_indexer_mma_scale_storage_shape(total_q, heads_q, fp4_format=fmt)
    assert tuple(k_storage.shape) == fp4_indexer_mma_scale_storage_shape(128, page_count * heads_k, fp4_format=fmt)
    assert tuple(q_storage.stride()) == fp4_indexer_mma_scale_storage_stride(total_q, heads_q, fp4_format=fmt)
    assert tuple(k_storage.stride()) == fp4_indexer_mma_scale_storage_stride(128, page_count * heads_k, fp4_format=fmt)
    assert q_storage.is_contiguous()
    assert k_storage.is_contiguous()

    q_scale_cpu = q_scale.cpu()
    k_scale_cpu = k_scale.cpu()
    q_mma_cpu = q_mma.cpu()
    k_mma_cpu = k_mma.cpu()
    for row in range(total_q):
        row_atom = row % 32
        row_major = (row // 32) % 4
        row_block = row // 128
        for head in range(heads_q):
            for group in range(spec.scale_groups):
                assert _fp8_byte(q_mma_cpu[row_atom, row_major, row_block, group % 4, group // 4, head]) == _fp8_byte(
                    q_scale_cpu[row, head, group]
                )
    for page in range(page_count):
        for head in range(heads_k):
            scale_l = page * heads_k + head
            for row in range(128):
                row_atom = row % 32
                row_major = (row // 32) % 4
                for group in range(spec.scale_groups):
                    assert _fp8_byte(k_mma_cpu[row_atom, row_major, 0, group % 4, group // 4, scale_l]) == _fp8_byte(
                        k_scale_cpu[page, head, row, group]
                    )


@pytest.mark.skipif(not _has_sm100_cuda(), reason="SM100-class CUDA device required")
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("max_seqlen", [130, 257])
@pytest.mark.parametrize(
    ("heads_q", "heads_k"),
    [
        pytest.param(2, 1, id="Hq2_Hk1"),
        pytest.param(4, 2, id="Hq4_Hk2"),
    ],
)
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale_layout", ["public", "preordered_mma"])
def test_cute_block_scores_random_fp4_matches_reference(
    fmt,
    batch,
    max_seqlen,
    heads_q,
    heads_k,
    seed,
    causal,
    scale_layout,
):
    case = _make_random_score_case(
        fmt=fmt,
        batch=batch,
        max_seqlen=max_seqlen,
        heads_q=heads_q,
        heads_k=heads_k,
        seed=seed,
    )

    ref = _reference_block_scores(
        case["q"].cpu(),
        case["k"].cpu(),
        case["q_scale"].cpu(),
        case["k_scale"].cpu(),
        case["cu_seqlens_q"].cpu(),
        case["cu_seqlens_k"].cpu(),
        case["cu_page_offsets"].cpu(),
        kv_indices=case["kv_indices"].cpu(),
        fmt=fmt,
        causal=causal,
    )
    q_scale_for_score = case["q_scale"]
    k_scale_for_score = case["k_scale"]
    if scale_layout == "preordered_mma":
        q_mma, k_mma = fp4_indexer_reorder_scales_for_mma_cute(
            case["q_scale"],
            case["k_scale"],
            fp4_format=fmt,
        )
        torch.cuda.synchronize()
        q_scale_for_score = _mma_scale_view_to_storage(q_mma)
        k_scale_for_score = _mma_scale_view_to_storage(k_mma)

    out = fp4_indexer_block_scores(
        case["q"],
        case["k"],
        q_scale_for_score,
        k_scale_for_score,
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["cu_page_offsets"],
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        kv_indices=case["kv_indices"],
        fp4_format=fmt,
        causal=causal,
        scale_layout=scale_layout,
    )
    torch.cuda.synchronize()

    assert tuple(out.shape) == tuple(ref.shape)
    assert torch.allclose(out.cpu(), ref, atol=_SCORE_ATOL, rtol=_SCORE_RTOL)


@pytest.mark.skipif(not _has_sm100_cuda(), reason="SM100-class CUDA device required")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seqlen_q", [1, 8])
@pytest.mark.parametrize("seqlen_k", [128, 257])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize(
    ("heads_q", "heads_k"),
    [
        pytest.param(16, 1, id="Hq16_Hk1"),
        pytest.param(32, 2, id="Hq32_Hk2"),
        pytest.param(64, 4, id="Hq64_Hk4"),
    ],
)
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale_layout", ["public", "preordered_mma"])
def test_cute_decode_packed_q_matches_reference(
    batch,
    seqlen_q,
    seqlen_k,
    seed,
    heads_q,
    heads_k,
    fmt,
    causal,
    scale_layout,
):
    length_generator = torch.Generator().manual_seed(seed + 1000)
    q_lengths = _random_lengths(batch, seqlen_q, length_generator)
    k_lengths = _random_lengths(batch, seqlen_k, length_generator)
    case = _make_benchmark_case(
        fmt=fmt,
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        head_kv=heads_k,
        qhead_per_kv=heads_q // heads_k,
        seed=seed,
        shuffle_pages=True,
        causal=causal,
    )
    ref = _reference_block_scores(
        case["q"].cpu(),
        case["k"].cpu(),
        case["q_scale"].cpu(),
        case["k_scale"].cpu(),
        case["cu_seqlens_q"].cpu(),
        case["cu_seqlens_k"].cpu(),
        case["cu_page_offsets"].cpu(),
        kv_indices=case["kv_indices"].cpu(),
        fmt=fmt,
        causal=causal,
    )
    q_scale_for_score = case["q_scale"]
    k_scale_for_score = case["k_scale"]
    if scale_layout == "preordered_mma":
        q_mma, k_mma = fp4_indexer_reorder_scales_for_mma_cute(
            case["q_scale"],
            case["k_scale"],
            fp4_format=fmt,
        )
        torch.cuda.synchronize()
        q_scale_for_score = _mma_scale_view_to_storage(q_mma)
        k_scale_for_score = _mma_scale_view_to_storage(k_mma)

    out = fp4_indexer_block_scores(
        case["q"],
        case["k"],
        q_scale_for_score,
        k_scale_for_score,
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["cu_page_offsets"],
        max_seqlen_q=int(case["seqlen_q"]),
        max_seqlen_k=int(case["seqlen_k"]),
        kv_indices=case["kv_indices"],
        fp4_format=fmt,
        causal=causal,
        scale_layout=scale_layout,
    )
    torch.cuda.synchronize()

    assert tuple(out.shape) == tuple(ref.shape)
    assert torch.allclose(out.cpu(), ref, atol=_SCORE_ATOL, rtol=_SCORE_RTOL)


def _make_benchmark_case(
    *,
    fmt: str,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    q_lengths: list[int] | None = None,
    k_lengths: list[int] | None = None,
    head_kv: int,
    qhead_per_kv: int,
    seed: int,
    shuffle_pages: bool,
    causal: bool,
) -> dict[str, torch.Tensor | int | str | list[int]]:
    if not _has_sm100_cuda():
        raise RuntimeError("SM100-class CUDA device required")
    if q_lengths is None:
        q_lengths = [seqlen_q] * batch
    if k_lengths is None:
        k_lengths = [seqlen_k] * batch
    if len(q_lengths) != len(k_lengths):
        raise ValueError("q_lengths and k_lengths must have the same length")
    batch = len(q_lengths)
    if batch <= 0:
        raise ValueError("batch must be positive")
    q_lengths = [int(v) for v in q_lengths]
    k_lengths = [int(v) for v in k_lengths]
    if any(v <= 0 for v in q_lengths) or any(v <= 0 for v in k_lengths):
        raise ValueError("all q_lengths and k_lengths must be positive")
    seqlen_q = max(q_lengths)
    seqlen_k = max(k_lengths)
    if head_kv <= 0 or qhead_per_kv <= 0:
        raise ValueError("head_kv and qhead_per_kv must be positive")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    spec = normalize_fp4_format(fmt)
    head_q = head_kv * qhead_per_kv
    pages_per_batch = [_ceil_div(length, 128) for length in k_lengths]
    total_q = sum(q_lengths)
    page_count = sum(pages_per_batch)

    q = torch.randint(0, 256, (total_q, head_q, 64), dtype=torch.uint8, device=device)
    k = torch.randint(0, 256, (page_count, head_kv, 128, 64), dtype=torch.uint8, device=device)
    q_scale = _random_scale_cuda((total_q, head_q, spec.scale_groups), fmt=fmt, device=device)
    k_scale = _random_scale_cuda((page_count, head_kv, 128, spec.scale_groups), fmt=fmt, device=device)
    cu_seqlens_q = torch.tensor(_exclusive_prefix(q_lengths), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(_exclusive_prefix(k_lengths), dtype=torch.int32, device=device)
    cu_page_offsets = torch.tensor(_exclusive_prefix(pages_per_batch), dtype=torch.int32, device=device)
    if shuffle_pages:
        kv_indices = torch.randperm(page_count, dtype=torch.int64, device=device).to(torch.int32)
    else:
        kv_indices = torch.arange(page_count, dtype=torch.int32, device=device)

    return {
        "fmt": fmt,
        "q": q,
        "k": k,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "cu_page_offsets": cu_page_offsets,
        "kv_indices": kv_indices,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "q_lengths": q_lengths,
        "k_lengths": k_lengths,
        "head_q": head_q,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "pages_per_batch": max(pages_per_batch),
        "pages_by_batch": pages_per_batch,
        "seed": seed,
        "causal": causal,
    }


def _run_score_benchmark_once(
    case: dict[str, torch.Tensor | int | str | list[int]],
    *,
    scale_layout: str,
):
    q_scale = case["q_scale"]
    k_scale = case["k_scale"]
    if scale_layout == "preordered_mma":
        q_scale = case["q_scale_preordered_mma"]
        k_scale = case["k_scale_preordered_mma"]
    return fp4_indexer_block_scores(
        case["q"],
        case["k"],
        q_scale,
        k_scale,
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["cu_page_offsets"],
        max_seqlen_q=int(case["seqlen_q"]),
        max_seqlen_k=int(case["seqlen_k"]),
        kv_indices=case["kv_indices"],
        fp4_format=str(case["fmt"]),
        causal=bool(case["causal"]),
        scale_layout=scale_layout,
    )


def _cuda_timed_loop_ms(fn, *, repeat: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _dense_score_flops(case: dict[str, torch.Tensor | int | str | list[int]]) -> int:
    q_lengths = case["q_lengths"]
    k_lengths = case["k_lengths"]
    qk = sum(int(q_len) * int(k_len) for q_len, k_len in zip(q_lengths, k_lengths))
    return 2 * int(case["head_q"]) * qk * _BENCHMARK_HEAD_DIM


def _causal_visible_qk_pairs(q_len: int, k_len: int) -> int:
    offset = int(k_len) - int(q_len)
    visible = 0
    for q_idx in range(int(q_len)):
        visible_k = q_idx + offset + 1
        visible_k = max(0, min(int(k_len), visible_k))
        visible += visible_k
    return visible


def _causal_effective_score_flops(case: dict[str, torch.Tensor | int | str | list[int]]) -> int:
    q_lengths = case["q_lengths"]
    k_lengths = case["k_lengths"]
    qk = sum(_causal_visible_qk_pairs(int(q_len), int(k_len)) for q_len, k_len in zip(q_lengths, k_lengths))
    return 2 * int(case["head_q"]) * qk * _BENCHMARK_HEAD_DIM


def _effective_score_flops(case: dict[str, torch.Tensor | int | str | list[int]]) -> int:
    if bool(case["causal"]):
        return _causal_effective_score_flops(case)
    return _dense_score_flops(case)


def _shape_summary_for_table(case: dict[str, torch.Tensor | int | str | list[int]]) -> str:
    q_lengths = case["q_lengths"]
    k_lengths = case["k_lengths"]
    return (
        f"B={int(case['batch'])}, q={_length_summary(q_lengths)}, "
        f"k={_length_summary(k_lengths)}, Hq={int(case['head_q'])}, "
        f"Hkv={int(case['head_kv'])}, "
        f"D={_BENCHMARK_HEAD_DIM}, blk_kv={_BENCHMARK_BLK_KV}, causal={bool(case['causal'])}"
    )


def _print_benchmark_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    headers = ("Case", "Format", "Scale Layout", "Shape", "Time ms", "Eff TFLOPS")
    table_rows = [
        (
            str(row["case"]),
            str(row["format"]).upper(),
            str(row["scale_layout"]),
            str(row["shape"]),
            f"{float(row['avg_ms']):.4f}",
            f"{float(row['effective_tflops']):.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]

    def fmt_row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(value.ljust(widths[col]) for col, value in enumerate(values)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in table_rows:
        print(fmt_row(row))


def _run_fp4_indexer_benchmark(
    *,
    case_name: str,
    fmt: str,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    q_lengths: list[int] | None,
    k_lengths: list[int] | None,
    head_kv: int,
    qhead_per_kv: int,
    warmup: int,
    iters: int,
    repeats: int,
    seed: int,
    shuffle_pages: bool,
    causal: bool,
    scale_layout: str,
) -> dict[str, object]:
    case = _make_benchmark_case(
        fmt=fmt,
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        seed=seed,
        shuffle_pages=shuffle_pages,
        causal=causal,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if scale_layout == "preordered_mma":
            q_mma, k_mma = fp4_indexer_reorder_scales_for_mma_cute(
                case["q_scale"],
                case["k_scale"],
                fp4_format=fmt,
            )
            torch.cuda.synchronize()
            case = {
                **case,
                "q_scale_preordered_mma": _mma_scale_view_to_storage(q_mma),
                "k_scale_preordered_mma": _mma_scale_view_to_storage(k_mma),
            }

        _run_score_benchmark_once(case, scale_layout=scale_layout)
        torch.cuda.synchronize()
        for _ in range(warmup):
            _run_score_benchmark_once(case, scale_layout=scale_layout)
        torch.cuda.synchronize()

        times = [
            _cuda_timed_loop_ms(
                lambda: _run_score_benchmark_once(
                    case,
                    scale_layout=scale_layout,
                ),
                repeat=iters,
            )
            for _ in range(repeats)
        ]
    avg_ms = statistics.mean(times)
    effective_tflops = _effective_score_flops(case) / (avg_ms * 1.0e-3) / 1.0e12

    return {
        "case": case_name,
        "format": fmt,
        "scale_layout": scale_layout,
        "shape": _shape_summary_for_table(case),
        "avg_ms": avg_ms,
        "effective_tflops": effective_tflops,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="FP4 indexer correctness tests and score-kernel benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run FP4 indexer score-kernel benchmark")
    bench.add_argument("--format", choices=["mxfp4", "nvfp4", "both"], default="nvfp4")
    bench.add_argument("--b", type=int, default=1, help="Batch size for custom --sq/--skv cases")
    bench.add_argument(
        "--sq",
        type=int,
        default=None,
        help="Run one custom uniform case with this Q length instead of the default benchmark suite",
    )
    bench.add_argument(
        "--skv",
        type=int,
        default=None,
        help="Run one custom uniform case with this K length instead of the default benchmark suite",
    )
    bench.add_argument("--head-kv", type=int, default=_BENCHMARK_HEAD_KV)
    bench.add_argument("--qhead-per-kv", type=int, default=_BENCHMARK_QHEAD_PER_KV)
    bench.add_argument("--dim", type=int, default=128)
    bench.add_argument("--blk-kv", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iters", type=int, default=100)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--shuffle-pages", action="store_true")
    bench.add_argument("--causal", action="store_true", help="Benchmark causal masking for custom --sq/--skv cases")
    bench.add_argument("--scale-layout", choices=["public", "preordered_mma", "both", "all"], default="preordered_mma")
    bench.add_argument(
        "--case",
        choices=["all", *[str(case["name"]) for case in _DEFAULT_BENCHMARK_CASES]],
        default="all",
        help="Default benchmark suite case to run",
    )
    bench.add_argument("--list-cases", action="store_true", help="List default benchmark suite cases and exit")
    bench.add_argument("--profile", action="store_true", help="Use warmup=0, iters=1, repeats=1")

    args = parser.parse_args()
    if args.command == "benchmark":
        if args.dim != _BENCHMARK_HEAD_DIM:
            raise ValueError("FP4 indexer benchmark only supports --dim 128")
        if args.blk_kv != _BENCHMARK_BLK_KV:
            raise ValueError("FP4 indexer benchmark only supports --blk-kv 128")
        warmup = 0 if args.profile else args.warmup
        iters = 1 if args.profile else args.iters
        repeats = 1 if args.profile else args.repeats
        formats = ["mxfp4", "nvfp4"] if args.format == "both" else [args.format]
        if args.scale_layout == "both":
            scale_layouts = ["public", "preordered_mma"]
        elif args.scale_layout == "all":
            scale_layouts = ["public", "preordered_mma"]
        else:
            scale_layouts = [args.scale_layout]
        if args.list_cases:
            for case in _DEFAULT_BENCHMARK_CASES:
                q_lengths = case["q_lengths"]
                k_lengths = case["k_lengths"]
                head_kv = int(case.get("head_kv", args.head_kv))
                qhead_per_kv = int(case.get("qhead_per_kv", args.qhead_per_kv))
                print(
                    f"{case['name']}: causal={bool(case['causal'])} "
                    f"batch={len(q_lengths)} q={_length_summary(q_lengths)} "
                    f"k={_length_summary(k_lengths)} "
                    f"head_q={head_kv * qhead_per_kv} head_kv={head_kv} "
                    f"qhead_per_kv={qhead_per_kv}"
                )
            return

        custom_shape = args.sq is not None or args.skv is not None
        if custom_shape:
            seqlen_q = _CUSTOM_BENCHMARK_DEFAULT_SQ if args.sq is None else args.sq
            seqlen_k = _CUSTOM_BENCHMARK_DEFAULT_SKV if args.skv is None else args.skv
            cases = (
                {
                    "name": "custom",
                    "q_lengths": [seqlen_q] * args.b,
                    "k_lengths": [seqlen_k] * args.b,
                    "causal": args.causal,
                },
            )
        elif args.case == "all":
            cases = _DEFAULT_BENCHMARK_CASES
        else:
            cases = tuple(case for case in _DEFAULT_BENCHMARK_CASES if case["name"] == args.case)

        rows: list[dict[str, object]] = []
        for case in cases:
            q_lengths = case["q_lengths"]
            k_lengths = case["k_lengths"]
            case_head_kv = int(case.get("head_kv", args.head_kv))
            case_qhead_per_kv = int(case.get("qhead_per_kv", args.qhead_per_kv))
            for fmt in formats:
                for scale_layout in scale_layouts:
                    rows.append(
                        _run_fp4_indexer_benchmark(
                            case_name=str(case["name"]),
                            fmt=fmt,
                            batch=len(q_lengths),
                            seqlen_q=max(q_lengths),
                            seqlen_k=max(k_lengths),
                            q_lengths=q_lengths,
                            k_lengths=k_lengths,
                            head_kv=case_head_kv,
                            qhead_per_kv=case_qhead_per_kv,
                            warmup=warmup,
                            iters=iters,
                            repeats=repeats,
                            seed=args.seed,
                            shuffle_pages=args.shuffle_pages,
                            causal=bool(case["causal"]),
                            scale_layout=scale_layout,
                        )
                    )
        _print_benchmark_table(rows)


if __name__ == "__main__":
    _main()
