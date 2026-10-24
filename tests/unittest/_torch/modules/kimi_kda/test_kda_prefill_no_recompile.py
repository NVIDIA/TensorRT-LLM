# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill chain compiles once per layout, not once per batch shape.

Chunked prefill changes the token count and the sequence count on nearly every step. A CuTe
compilation that depends on either costs about a second on every rank of a lock-stepped
deployment, so after the first call of each path nothing in the chain may compile again:
the fused convolution, the BT16 chunk kernel (engine and decomposition routes) and the
Small-BH kernels.
"""

import random
from itertools import accumulate

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops  # noqa: F401  (registers the ops)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)

LOWER_BOUND = -5.0
MAX_TOKENS = 4096
MAX_SEQS = 64
POOL_SLOTS = MAX_SEQS + 8


def _split(total: int, seqs: int, rng: random.Random) -> list[int]:
    """Random lengths of ``total`` tokens over ``seqs`` sequences (random cut points: some are empty)."""
    cuts = sorted(rng.randint(0, total) for _ in range(seqs - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, total])]


class _Pools:
    """State pools shared by every launch, as in serving: only the batch changes."""

    def __init__(self, heads: int) -> None:
        self.heads = heads
        self.conv = torch.zeros(POOL_SLOTS, 3 * heads * 128, 3, device="cuda", dtype=torch.bfloat16)
        self.ssm = torch.zeros(POOL_SLOTS, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
        self.weight = torch.randn(3 * heads * 128, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        self.a_log = torch.log(torch.linspace(1.0, 16.0, heads, device="cuda"))
        self.dt_bias = torch.zeros(heads * 128, device="cuda")

    def run(self, lengths: list[int]) -> None:
        heads, n, total = self.heads, len(lengths), sum(lengths)
        offsets = [0, *accumulate(lengths)]
        starts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        slots = torch.arange(n, device="cuda", dtype=torch.int32)
        has_init = torch.arange(n, device="cuda") % 2 == 0
        src = torch.randn(total, 4 * heads * 128, device="cuda", dtype=torch.bfloat16) * 0.1
        # As in the mixer: the whole fused [q|k|v|g] projection, row stride 4 * H * 128.
        q, k, v = torch.ops.trtllm.kda_prefill_conv(
            src,
            self.weight,
            self.conv,
            starts,
            slots,
            has_init,
            max(lengths),
        )
        shape = (1, total, heads, 128)
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
        out = torch.ops.trtllm.kda_prefill_flashinfer(
            q,
            k,
            v,
            g,
            beta,
            self.ssm,
            slots,
            torch.tensor(offsets, device="cuda", dtype=torch.int64),
            self.a_log,
            self.dt_bias,
            128**-0.5,
            LOWER_BOUND,
            "auto",
            max(lengths),
        )
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("heads", [6, 96])
@torch.inference_mode()
def test_prefill_chain_does_not_compile_after_warmup(heads: int, cute_compiles) -> None:
    pools = _Pools(heads)
    rng = random.Random(20261020 + heads)
    # One launch per kernel family: the conv and Small-BH (one sequence of several tokens), BT16
    # (a single token cannot use Small-BH) and, whichever route the architecture picks for them,
    # a few and many sequences. BT16's one compilation serves its engine and decomposition routes.
    for warm_up in ([64], [1], [40, 24], [3] * 9, [1] * MAX_SEQS):
        pools.run(warm_up)
    warmed = len(cute_compiles)

    edge = [[1], [2], [1, 1], [MAX_TOKENS], [1] * MAX_SEQS, [3] * 17, [MAX_TOKENS - 63] + [1] * 63]
    random_batches = [
        _split(rng.randint(1, MAX_TOKENS), rng.randint(1, MAX_SEQS), rng) for _ in range(32)
    ]
    for lengths in edge + random_batches:
        pools.run(lengths)
    assert len(cute_compiles) == warmed, cute_compiles[warmed:]
