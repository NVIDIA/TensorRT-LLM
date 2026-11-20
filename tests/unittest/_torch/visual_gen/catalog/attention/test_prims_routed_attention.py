# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.prims_routed_attention import (
    PrimsSageQuantization,
    prims_routed_attention,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS needs Blackwell",
)
@pytest.mark.parametrize("cell", prims_routed_attention.CELLS, ids=lambda c: c.why)
def test_prims_routed_attention(cell):
    rng = torch.Generator(device="cuda").manual_seed(1824)

    def sample(length):
        return torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=rng)

    q, k, v = sample(128), sample(160), sample(160)
    mode = cell.spec["route"]
    kwargs = dict(plans={})
    if mode == "bsr":
        kwargs.update(
            block_indptr=torch.tensor([[[0, 1, 2], [2, 3, 4]]], dtype=torch.int32, device="cuda"),
            block_indices=torch.tensor([2, 0, 1, 2], dtype=torch.int32, device="cuda"),
            max_blocks_per_row=1,
            q_block_size=64,
            kv_block_size=64,
            kv_valid_bits=torch.full((1, 5), 0x55555555, dtype=torch.uint32, device="cuda"),
        )
        v[:, 1::2].fill_(6)
    elif mode == "proxy":
        kwargs.update(
            q_block_size=64,
            kv_block_size=64,
            exact_block_bits=torch.tensor(
                [[[[1], [4]], [[2], [1]]]], dtype=torch.uint32, device="cuda"
            ),
            k_summary=sample(3),
            v_summary=sample(3),
        )
    elif mode == "sage":
        kwargs["quantization"] = PrimsSageQuantization(torch.float8_e4m3fn, 1, 16, True, True)
    prims_routed_attention.is_valid(q, k, v, **kwargs)
    actual = prims_routed_attention(q, k, v, **kwargs)
    expected = prims_routed_attention.reference(q, k, v, **kwargs)
    assert actual.dtype == q.dtype
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.12 if mode == "sage" else 0.02,
        atol=0.045 if mode == "sage" else 0.02,
    )


@pytest.mark.parametrize(
    "offsets,indices,capacity",
    [
        ([0, 1, 3], [0, 1], 2),
        ([0, 1, 2], [0, 3], 1),
        ([0, 2, 2], [1, 1], 2),
        ([0, 2, 2], [0, 1], 1),
    ],
    ids=["unbounded offset", "absent KV block", "duplicate block", "over capacity"],
)
def test_invalid_bsr_row(offsets, indices, capacity):
    q = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 192, 1, 128, dtype=q.dtype)
    with pytest.raises(AssertionError):
        prims_routed_attention.is_valid(
            q,
            k,
            k,
            plans={},
            q_block_size=64,
            kv_block_size=64,
            max_blocks_per_row=capacity,
            block_indptr=torch.tensor([[offsets]], dtype=torch.int32),
            block_indices=torch.tensor(indices, dtype=torch.int32),
        )


def test_bitmask_capacity_counts_structural_blocks():
    q = torch.empty(1, 64, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 160, 1, 128, dtype=q.dtype)
    bits = torch.full((1, 1, 1, 1), 0xFFFFFFF9, dtype=torch.uint32)
    route = dict(
        plans={}, q_block_size=64, kv_block_size=64, max_blocks_per_row=1, exact_block_bits=bits
    )
    prims_routed_attention.is_valid(q, k, k, **route)
    bits.fill_(0xFFFFFFFB)
    with pytest.raises(AssertionError):
        prims_routed_attention.is_valid(q, k, k, **route)


def test_kv_valid_bits_cover_all_token_words():
    q = torch.empty(1, 64, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 65, 1, 128, dtype=q.dtype)
    with pytest.raises(AssertionError):
        prims_routed_attention.is_valid(
            q,
            k,
            k,
            plans={},
            q_block_size=64,
            kv_block_size=64,
            exact_block_bits=torch.zeros((1, 1, 1, 1), dtype=torch.uint32),
            kv_valid_bits=torch.zeros((1, 2), dtype=torch.uint32),
        )


def test_bsr_and_exact_bitmask_are_exclusive():
    q = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        prims_routed_attention.is_valid(
            q,
            q,
            q,
            plans={},
            block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
            block_indices=torch.tensor([0], dtype=torch.int32),
            exact_block_bits=torch.tensor([[[[1]]]], dtype=torch.uint32),
        )
