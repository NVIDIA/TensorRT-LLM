# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's Engram look-back hash (`engram_hash.EngramHash`, every target's) over context chunks longer than the
history ring equals the token-by-token look-back."""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.engram_hash import ENGRAM_RING as _RING
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.engram_hash import EngramHash

pytestmark = pytest.mark.cpu_only

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SLOTS, MAX_N, HEADS, LAYERS, VOCAB = 4, 3, 8, 2, 1000


def _hash():
    h = EngramHash.__new__(EngramHash)
    h.max_n, h.pad_id = MAX_N, 7
    h.token_map = torch.arange(VOCAB, device=DEV) % 613
    rng = np.random.default_rng(0)
    h.mults = torch.tensor(rng.integers(0, 1 << 40, size=(LAYERS, MAX_N)) * 2 + 1, device=DEV)
    h.primes = torch.tensor(rng.integers(500, 1000, size=(LAYERS, MAX_N - 1, HEADS)), device=DEV)
    return h


def _run(h, steps):
    """steps: [(slot, first position, tokens)] per step -> hashes per (slot, position)."""
    hist = torch.full((SLOTS + 1, _RING), h.pad_id, dtype=torch.int32, device=DEV)
    out = {}
    for step in steps:
        ids = torch.tensor([t for _, _, toks in step for t in toks], device=DEV)
        pos = torch.tensor([p + i for _, p, toks in step for i in range(len(toks))], device=DEV)
        slot = torch.tensor([s for s, _, toks in step for _ in toks], device=DEV)
        got = h.lookback(ids, pos, slot, hist)
        for r in range(len(ids)):
            out[(int(slot[r]), int(pos[r]))] = got[r]
    return out


@pytest.mark.parametrize("chunk", [5, _RING, _RING + 1, 3 * _RING + 5, 2100])
def test_chunked_equals_token_by_token(chunk):
    h = _hash()
    gen = torch.Generator().manual_seed(chunk)
    seqs = {
        s: torch.randint(0, VOCAB, (n,), generator=gen).tolist()
        for s, n in ((0, 2 * chunk + 3), (2, chunk + 1))
    }
    ref = _run(h, [[(s, p, [toks[p]])] for s, toks in seqs.items() for p in range(len(toks))])
    # context chunks of `chunk` tokens of both requests in the same steps, then one decode token each
    steps, starts = [], {s: 0 for s in seqs}
    while any(starts[s] < len(seqs[s]) - 1 for s in seqs):
        step = []
        for s, toks in seqs.items():
            end = min(starts[s] + chunk, len(toks) - 1)
            if end > starts[s]:
                step.append((s, starts[s], toks[starts[s] : end]))
                starts[s] = end
        steps.append(step)
    steps.append([(s, len(toks) - 1, [toks[-1]]) for s, toks in seqs.items()])
    got = _run(h, steps)
    assert got.keys() == ref.keys()
    for k in ref:
        assert torch.equal(got[k], ref[k]), k
