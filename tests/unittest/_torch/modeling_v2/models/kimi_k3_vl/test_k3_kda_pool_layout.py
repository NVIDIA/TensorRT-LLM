# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch

source = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kernels/k3_kda_verify/_pool_layout.py"
)
spec = importlib.util.spec_from_file_location("_k3_kda_pool_layout_test", source)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)
pytestmark = pytest.mark.cpu_only


def _pools(dtype: torch.dtype, coalesced: bool) -> tuple[torch.Tensor, ...]:
    slots, layers, heads, width, drafts = 5, 2 if coalesced else 1, 6, 128, 7
    channels = heads * width
    caches = tuple(
        torch.zeros(slots, layers, 3 + drafts, channels, dtype=dtype)[:, -1].transpose(1, 2)
        for _ in range(3)
    )
    records = torch.zeros(slots, layers, 3, drafts, heads, width)[:, -1]
    counts = torch.zeros(slots, 4 if coalesced else 1, dtype=torch.int32)[:, 0]
    return *caches, records, counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("coalesced", [False, True])
def test_pool_views_preserve_real_storage_and_slot_stride(
    dtype: torch.dtype, coalesced: bool
) -> None:
    caches = _pools(dtype, coalesced)
    assert op.replay_pools_supported(*caches, 6, 7)
    for cache in caches[:3]:
        flat = op.slots_view(cache.transpose(1, 2))
        assert flat.data_ptr() == cache.data_ptr()
        assert flat.stride(0) == cache.stride(0)
        flat[3, 2 * 768 + 17] = 9
        assert cache[3, 17, 2].item() == 9
        assert cache[2, 17, 2].item() == 0
    records = caches[3]
    flat_records = op.slots_view(records)
    flat_records[4, 2 * 7 * 768 + 3 * 768 + 129] = 11
    assert records[4, 2, 3, 1, 1].item() == 11
    assert records[3, 2, 3, 1, 1].item() == 0
    counts = caches[4]
    counts[2] = 5
    assert counts[1].item() == 0
    assert counts[2].item() == 5


@pytest.mark.parametrize(
    "malformed", ["conv-inner-stride", "conv-dtype", "record-inner-stride", "counts-rank"]
)
def test_rejects_pool_views_the_kernel_cannot_address(malformed: str) -> None:
    pools = list(_pools(torch.bfloat16, True))
    if malformed == "conv-inner-stride":
        pools[0] = pools[0].contiguous()
    elif malformed == "conv-dtype":
        pools[1] = pools[1].float()
    elif malformed == "record-inner-stride":
        pools[3] = pools[3].transpose(-1, -2)
    else:
        pools[4] = pools[4].unsqueeze(0)
    assert not op.replay_pools_supported(*pools, 6, 7)
