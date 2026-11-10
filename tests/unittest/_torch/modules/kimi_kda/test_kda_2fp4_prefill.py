# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed 2FP4 KDA prefill: BT16 reads and writes V2 checkpoints in place.

The reference is the QDQ flow on the same route: materialize FP32 rows, run
the FP32-state prefill, then store the checkpoints. Without initial states the
two flows are bitwise identical. With initial states the fused dequantization
uses one FMA contraction order, which the Triton materialization does not, so
outputs and states agree to rounding.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda import kda_2fp4 as ops
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
    is_intree_prefill_available,
    is_kda_optimized_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_kda_optimized_supported()
    or not is_intree_prefill_available(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)

STATE_BYTES = ops.STATE_BYTES_PER_HEAD
SLOT_PADDING = 64


def _make_case(heads: int, lengths: list[int], fresh: list[bool], seed: int = 0) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens, batch = sum(lengths), len(lengths)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=generator) * scale

    slots_total = batch + 5
    # Noncontiguous slots: per-slot padding after the dense head payloads must survive.
    backing = torch.full(
        (slots_total, heads * STATE_BYTES + SLOT_PADDING), 0xA5, dtype=torch.uint8, device="cuda"
    )
    pool = backing[:, : heads * STATE_BYTES].view(slots_total, heads, 1, STATE_BYTES)
    ops.store_checkpoint(
        pool,
        torch.arange(slots_total, dtype=torch.int32, device="cuda"),
        randn(slots_total, heads, 128, 128, scale=0.05),
    )
    offsets = torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()
    return {
        "q": randn(1, tokens, heads, 128).bfloat16(),
        "k": randn(1, tokens, heads, 128).bfloat16(),
        "v": randn(1, tokens, heads, 128, scale=0.1).bfloat16(),
        "g": (randn(1, tokens, heads, 128) - 4).bfloat16(),
        "beta": randn(1, tokens, heads).bfloat16(),
        "A_log": randn(heads, scale=0.15),
        "dt_bias": randn(heads * 128, scale=0.2),
        "cu_seqlens": torch.tensor([0, *offsets], dtype=torch.int64, device="cuda"),
        "slots": torch.randperm(slots_total, generator=generator, device="cuda")[:batch].int(),
        "has_init": torch.tensor([not f for f in fresh], dtype=torch.bool, device="cuda"),
        "backing": backing,
        "pool": pool,
    }


def _prefill(case: dict, state_pool: torch.Tensor, state_indices: torch.Tensor, mode: str):
    return torch.ops.trtllm.kda_prefill(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        state_pool,
        state_indices,
        case["cu_seqlens"],
        case["A_log"],
        case["dt_bias"],
        128**-0.5,
        -5.0,
        mode,
    )


@pytest.mark.parametrize("mode", ["engine", "decomp"])
@pytest.mark.parametrize(
    ("heads", "lengths", "fresh"),
    [
        (96, [512, 512], [True, True]),
        (24, [1, 15, 16, 17, 100, 300, 31, 64], [True] * 8),
        (96, [0, 48, 129], [False, False, True]),
        (96, [256] * 4, [False, True, False, True]),
        (12, [100, 1, 31, 64, 17, 300, 16, 15], [False] * 8),
    ],
)
def test_packed_prefill_matches_qdq(
    mode: str, heads: int, lengths: list[int], fresh: list[bool]
) -> None:
    case = _make_case(heads, lengths, fresh)
    slots, has_init = case["slots"], case["has_init"]
    reference_backing = case["backing"].clone()
    reference_pool = reference_backing[:, : heads * STATE_BYTES].view(case["pool"].shape)

    staged = ops.materialize_checkpoint(reference_pool, slots, has_init)
    rows = torch.arange(slots.numel(), dtype=torch.int32, device="cuda")
    reference = _prefill(case, staged, rows, mode)
    ops.store_checkpoint(reference_pool, slots, staged)

    encoded = torch.where(has_init, slots, torch.bitwise_not(slots))
    actual = _prefill(case, case["pool"], encoded, mode)
    torch.cuda.synchronize()

    backing = case["backing"]
    untouched = torch.ones(backing.shape[0], dtype=torch.bool, device="cuda")
    untouched[slots.long()] = False
    torch.testing.assert_close(backing[untouched], reference_backing[untouched], rtol=0, atol=0)
    torch.testing.assert_close(
        backing[:, heads * STATE_BYTES :],
        reference_backing[:, heads * STATE_BYTES :],
        rtol=0,
        atol=0,
    )
    if all(fresh):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(backing, reference_backing, rtol=0, atol=0)
        return
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-4)
    every_row = torch.ones_like(has_init)
    decoded = ops.materialize_checkpoint(case["pool"], slots, every_row).double()
    expected = ops.materialize_checkpoint(reference_pool, slots, every_row).double()
    error = (decoded - expected).flatten(2).norm(dim=-1)
    assert (error <= 2e-3 * expected.flatten(2).norm(dim=-1)).all()


def test_packed_prefill_rejects_unsupported_checkpoint_layout() -> None:
    case = _make_case(12, [32], [True])
    # Heads padded apart: the kernel addresses dense per-slot head payloads.
    padded = torch.zeros((4, 12, STATE_BYTES + 16), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="2FP4 checkpoint"):
        _prefill(case, padded[..., :STATE_BYTES], case["slots"], "auto")
