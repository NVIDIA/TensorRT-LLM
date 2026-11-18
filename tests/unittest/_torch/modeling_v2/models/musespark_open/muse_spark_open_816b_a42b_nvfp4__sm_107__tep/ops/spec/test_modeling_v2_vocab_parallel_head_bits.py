# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vocabulary-parallel head's tokens are the built-in head's, bit for bit (``ops/spec/head_sampler``).

``VocabParallelEmulation`` runs the served ``vp_fold`` / ``vp_draw`` / ``vp_decide`` / ``vp_argmax`` kernels for
four virtual ranks on one GPU (every value stored into each rank's copy). Each rank folds its shard onto the previous
rank's per-virtual-thread states and continues the previous rank's walk, so its tokens must equal ``split_sample`` on
the gathered BF16 row and ``fused_sample_from_logits`` on the soft-capped FP32 row on every row: the 1024-thread
block (at most 128 rows) and the 512-thread block (more rows), flat to peaky rows, mixed temperatures, rows with a
filter, and greedy batches. Every exchange word is the sentinel again after each call.

Needs an SM 10.7 device (the library is built for ``sm_107a`` on first use).
"""

from __future__ import annotations

import pytest
import torch

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
RANKS = 4
#: The served TP4 shard of the 202,048-token vocabulary: 12,628 vectors, not a multiple of either virtual block.
SHARD = 202048 // RANKS
MULTIPLIER = 0.5
SOFT_CAP = 30.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the head sampler library is built for sm_107a",
)


@pytest.fixture(scope="module")
def hs():
    import importlib

    return importlib.import_module(TARGET + ".ops.spec.head_sampler")


@pytest.fixture(scope="module")
def emulation(hs):
    return hs.VocabParallelEmulation(RANKS, SHARD, hs.WIDE_MAX_ROWS, "cuda")


def _row_params(rows: int, gen: torch.Generator, mixed: bool):
    dev = torch.device("cuda")
    temps = torch.ones(rows, dtype=torch.float32, device=dev)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=dev)
    top_ps = torch.ones(rows, dtype=torch.float32, device=dev)
    min_ps = torch.zeros(rows, dtype=torch.float32, device=dev)
    if mixed:
        temps[1::3] = 0.6
        temps[2::5] = 1.7
        top_ks[3::11] = 40
        top_ps[5::13] = 0.9
    seeds = torch.randint(0, 2**62, (rows,), dtype=torch.int64, device=dev, generator=gen)
    offsets = torch.randint(0, 2**40, (rows,), dtype=torch.int64, device=dev, generator=gen)
    return temps, top_ks, top_ps, min_ps, seeds, offsets


@pytest.mark.parametrize("rows", [16, 128, 129, 744])
@pytest.mark.parametrize("scale", [0.25, 2.0, 16.0])
@pytest.mark.parametrize("mixed", [False, True])
def test_sampled_tokens_bitwise(hs, emulation, rows, scale, mixed):
    gen = torch.Generator(device="cuda").manual_seed(rows * 1000 + int(scale * 8) + int(mixed))
    logits = (torch.randn(rows, RANKS * SHARD, device="cuda", generator=gen) * scale).to(
        torch.bfloat16
    )
    temps, top_ks, top_ps, min_ps, seeds, offsets = _row_params(rows, gen, mixed)
    locals_ = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(RANKS)]

    tokens, pendings = emulation.sample(
        locals_, temps, top_ks, top_ps, min_ps, seeds, offsets, MULTIPLIER, SOFT_CAP
    )
    states = hs.new_states(hs.WIDE_MAX_ROWS, logits.device)
    split = hs.split_sample(
        logits, temps, top_ks, top_ps, min_ps, seeds, offsets, MULTIPLIER, SOFT_CAP, states
    )[:rows]
    capped = hs.reference_capped(logits, MULTIPLIER, SOFT_CAP)
    builtin = torch.ops.trtllm.fused_sample_from_logits(
        capped, temps, top_ks, top_ps, min_ps, seeds, offsets
    )[:rows]

    assert torch.equal(split.long(), builtin.long())
    for rank_tokens in tokens:
        assert torch.equal(rank_tokens[:rows].long(), builtin.long())
    filtered = (top_ks > 0) | (top_ps < 1.0) | (min_ps > 0.0)
    for pending in pendings:
        assert torch.equal(pending[filtered], logits[filtered])
    assert emulation.all_sentinel()


@pytest.mark.parametrize("rows", [16, 744])
def test_greedy_tokens_bitwise(hs, emulation, rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    logits = (torch.randn(rows, RANKS * SHARD, device="cuda", generator=gen) * 4.0).to(
        torch.bfloat16
    )
    locals_ = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(RANKS)]
    tokens = emulation.argmax(locals_, MULTIPLIER, SOFT_CAP)
    expected = torch.argmax(hs.reference_capped(logits, MULTIPLIER, SOFT_CAP), dim=-1)
    for rank_tokens in tokens:
        assert torch.equal(rank_tokens[:rows], expected)
    assert emulation.all_sentinel()
