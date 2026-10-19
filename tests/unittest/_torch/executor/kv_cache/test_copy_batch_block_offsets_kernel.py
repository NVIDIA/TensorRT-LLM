# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``copy_batch_block_offsets_to_device`` against a torch reference.

The kernel splits each (pool, sequence) row over several CTAs; the shapes cover
rows shorter than one CTA tile, rows ending inside a tile, and the 32K-block
rows of a long-context pool.
"""

import pytest
import torch

from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)

BAD_PAGE_INDEX = -1
SENTINEL = -7

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _pinned(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory()


@pytest.mark.parametrize("num_blocks", [4, 132, 1028, 2052, 4100, 32772])
@pytest.mark.parametrize("num_pools", [1, 2])
@pytest.mark.parametrize("num_seqs", [1, 3, 16])
def test_copy_batch_block_offsets_matches_reference(num_blocks, num_pools, num_seqs):
    gen = torch.Generator().manual_seed(num_blocks * 31 + num_pools * 7 + num_seqs)
    src_seqs = num_seqs + 4
    dst_seqs = num_seqs + 2
    src = torch.randint(
        0, 1 << 20, (num_pools, src_seqs, 2, num_blocks), generator=gen, dtype=torch.int32
    )
    holes = torch.rand(src.shape, generator=gen) < 0.2
    src[holes] = BAD_PAGE_INDEX
    src[:, :, :, num_blocks - num_blocks // 3 :] = BAD_PAGE_INDEX
    copy_index = torch.randperm(src_seqs, generator=gen)[:num_seqs].to(torch.int32)
    scales = torch.randint(1, 4, (num_pools,), generator=gen, dtype=torch.int32)
    kv_offset = torch.randint(0, 5, (num_pools,), generator=gen, dtype=torch.int32)

    dst = torch.full(
        (num_pools, dst_seqs, 2, num_blocks), SENTINEL, dtype=torch.int32, device="cuda"
    )
    copy_batch_block_offsets_to_device(
        _pinned(src),
        dst,
        _pinned(copy_index),
        _pinned(scales),
        _pinned(kv_offset),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    rows = src[:, copy_index.long(), 0].long()
    bad = rows == BAD_PAGE_INDEX
    scaled = rows * scales.long().view(-1, 1, 1)
    expected = torch.full(dst.shape, SENTINEL, dtype=torch.int32)
    expected[:, :num_seqs, 0] = torch.where(bad, 0, scaled).to(torch.int32)
    expected[:, :num_seqs, 1] = torch.where(bad, 0, scaled + kv_offset.long().view(-1, 1, 1)).to(
        torch.int32
    )
    assert torch.equal(dst.cpu(), expected)
