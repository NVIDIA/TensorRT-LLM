# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check gathered MLA history ordering; CP32 here is synthetic on one GPU."""

import pytest
import torch

import tensorrt_llm  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cp_size", [1, 2, 4, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lora_size", [448, 512])
@pytest.mark.parametrize("empty", [False, True])
def test_mla_helix_history_reorder(cp_size, dtype, lora_size, empty):
    page, rope_size = 64, 64
    starts = [0, 63, 64, 65, 127, 128, 129, 2047]
    lengths = [1, 0, 2, 64, 65, 129, 17, 3] if not empty else [0] * len(starts)
    tokens = [
        (req, g)
        for req, (start, length) in enumerate(zip(starts, lengths))
        for g in range(start, start + length)
    ]
    generator = torch.Generator(device="cuda").manual_seed(187)
    expected = torch.randn(
        len(tokens), lora_size + rope_size, device="cuda", dtype=dtype, generator=generator
    )
    source_ids, local_cu, local_starts, rank_offsets = [], [], [], [0]
    for rank in range(cp_size):
        owned = [
            i for i, (_, position) in enumerate(tokens) if (position // page) % cp_size == rank
        ]
        source_ids.extend(owned)
        rank_offsets.append(len(source_ids))
        cu, offsets = [0], []
        for start, length in zip(starts, lengths):
            offsets.append(sum((g // page) % cp_size == rank for g in range(start)))
            cu.append(
                cu[-1] + sum((g // page) % cp_size == rank for g in range(start, start + length))
            )
        local_cu.append(cu)
        local_starts.append(offsets)
    gathered = expected[torch.tensor(source_ids, device="cuda", dtype=torch.long)]
    cu_global = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0)
    compressed, positional = torch.ops.trtllm.reorder_helix_mla_kv(
        gathered,
        cu_global,
        torch.tensor(starts, device="cuda", dtype=torch.int64),
        torch.tensor(local_cu, device="cuda", dtype=torch.int64),
        torch.tensor(local_starts, device="cuda", dtype=torch.int64),
        torch.tensor(rank_offsets, device="cuda", dtype=torch.int64),
        page,
        lora_size,
        rope_size,
    )
    assert compressed.is_contiguous() and positional.is_contiguous()
    torch.testing.assert_close(compressed, expected[:, :lora_size], atol=0, rtol=0)
    torch.testing.assert_close(positional, expected[:, lora_size:], atol=0, rtol=0)
