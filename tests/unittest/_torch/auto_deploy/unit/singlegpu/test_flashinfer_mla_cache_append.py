# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.mla.flashinfer_mla import (
    _append_paged_mla_kv_cache_fallback,
)

try:
    import flashinfer
except ImportError:  # pragma: no cover - exercised only in FlashInfer-enabled envs.
    flashinfer = None


def _make_append_metadata(device: torch.device):
    page_size = 4
    batch_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 4, 0, 2], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([3, 1, 5], dtype=torch.int32, device=device)
    return page_size, batch_indices, positions, kv_indices, kv_indptr


def test_fallback_append_writes_expected_pages_for_smaller_mla_dims():
    device = torch.device("cpu")
    page_size, batch_indices, positions, kv_indices, kv_indptr = _make_append_metadata(device)

    append_ckv = torch.arange(5 * 256, dtype=torch.float32, device=device).view(5, 256)
    append_kpe = torch.arange(5 * 64, dtype=torch.float32, device=device).view(5, 64)

    ckv_cache = torch.zeros(8, page_size, 256, dtype=torch.float32, device=device)
    kpe_cache = torch.zeros(8, page_size, 64, dtype=torch.float32, device=device)

    _append_paged_mla_kv_cache_fallback(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
    )

    for token_idx, (batch_idx, position) in enumerate(
        zip(batch_indices.tolist(), positions.tolist())
    ):
        page_offset = kv_indptr[batch_idx].item() + position // page_size
        page_idx = kv_indices[page_offset].item()
        entry_idx = position % page_size
        torch.testing.assert_close(ckv_cache[page_idx, entry_idx], append_ckv[token_idx])
        torch.testing.assert_close(kpe_cache[page_idx, entry_idx], append_kpe[token_idx])


@pytest.mark.skipif(
    flashinfer is None or not torch.cuda.is_available(), reason="requires CUDA FlashInfer"
)
def test_fallback_append_matches_flashinfer_native_append_for_supported_dims():
    device = torch.device("cuda")
    page_size, batch_indices, positions, kv_indices, kv_indptr = _make_append_metadata(device)
    kv_last_page_len = torch.tensor([1, 3], dtype=torch.int32, device=device)

    append_ckv = torch.randn(5, 512, dtype=torch.bfloat16, device=device)
    append_kpe = torch.randn(5, 64, dtype=torch.bfloat16, device=device)

    ckv_cache_native = torch.zeros(8, page_size, 512, dtype=torch.bfloat16, device=device)
    kpe_cache_native = torch.zeros(8, page_size, 64, dtype=torch.bfloat16, device=device)
    ckv_cache_fallback = torch.zeros_like(ckv_cache_native)
    kpe_cache_fallback = torch.zeros_like(kpe_cache_native)

    flashinfer.page.append_paged_mla_kv_cache(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache_native,
        kpe_cache_native,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )
    _append_paged_mla_kv_cache_fallback(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache_fallback,
        kpe_cache_fallback,
        kv_indices,
        kv_indptr,
    )

    torch.testing.assert_close(ckv_cache_fallback, ckv_cache_native)
    torch.testing.assert_close(kpe_cache_fallback, kpe_cache_native)
