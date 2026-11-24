# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_sample_from_logits entry.

A draw has no closed form, so the entry's reference is the distribution it draws from and the
tokens are gated by it: inside the support, greedy-sentinel rows on the argmax, reproducible
under a fixed (seed, offset), and with frequencies matching the distribution.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_sample_from_logits import (  # noqa: E501
    fused_sample_from_logits as op,
)

__extra_import_path__ = ["../../../..", "."]
from _k31_spec_inputs import (  # noqa: E402
    DISABLE_TOP_K,
    hot_rows,
    on_sm103,
    philox_state,
    sampling_inputs,
    seed_of,
)
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    logits, temperatures, top_ks, top_ps, min_ps = sampling_inputs(spec, seed_of(cell), fused=True)
    seeds, offsets = philox_state(spec["rows"], seed_of(cell))
    tokens = op(logits, temperatures, top_ks, top_ps, min_ps, seeds, offsets)
    probs = op.reference(logits, temperatures, top_ks, top_ps, min_ps, seeds, offsets)
    assert tokens.shape == (spec["rows"],) and tokens.dtype == torch.int32
    picked = probs.gather(1, tokens.long().view(-1, 1)).squeeze(1)
    assert (picked > 0).all(), (
        f"rows {(picked <= 0).nonzero().flatten().tolist()} drew outside the support"
    )
    greedy = temperatures <= 1e-4
    assert torch.equal(tokens[greedy].long(), logits[greedy].float().argmax(-1))
    again = op(logits, temperatures, top_ks, top_ps, min_ps, seeds, offsets)
    assert torch.equal(tokens, again), "the same (seed, offset) must draw the same tokens"


def test_draws_follow_the_distribution() -> None:
    """4096 draws of one row (distinct offsets) land with the reference's frequencies.

    Eight hot tokens carry the mass, so each frequency is pinned to about 0.008 and the total
    variation of a correct sampler stays near 0.02.
    """
    rows, vocab = 4096, 163840
    logits = hot_rows(rows, vocab, seed=5)
    temperatures = torch.full((rows,), 0.7, device="cuda")
    top_ks = torch.full((rows,), DISABLE_TOP_K, dtype=torch.int32, device="cuda")
    top_ps = torch.full((rows,), 0.9, device="cuda")
    min_ps = torch.zeros(rows, device="cuda")
    seeds, offsets = philox_state(rows, 4321)
    tokens = op(logits, temperatures, top_ks, top_ps, min_ps, seeds, offsets)
    one = slice(0, 1)
    probs = op.reference(logits[one], temperatures[one], top_ks[one], top_ps[one], min_ps[one])[0]
    freq = torch.bincount(tokens.long(), minlength=vocab).float() / rows
    tv = 0.5 * (freq - probs).abs().sum().item()
    assert tv < 0.05, f"total variation {tv:.4f} between the draws and the distribution"
    assert (probs[tokens.long()] > 0).all()
