# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sampled verification's placeholder rows (``SpecMetadata.real_draft_rows``).

A generation row that verifies drafts the drafter did not make (a request's first generation
step after a context step elsewhere) must verify them against their point mass, whatever its
sequence slot's store still holds. Here every slot's store holds a walk equal to the verified
drafts, so only the real-rows mark tells the rows apart: a marked-placeholder row must verify
exactly as it does with its store row invalidated, and the real rows exactly as without a mark.
Without a mark (forced acceptance off) nothing changes.
"""

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]
from _rubin_rs_rig import (  # noqa: E402
    VOCAB,
    Bufs,
    K,
    Markov,
    build_worker,
    sampled_spec,
    stub_drafter,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the DFlash2 sampled path's kernels are certified on sm_103 / sm_107",
)


def _verify(real, invalidate=()):
    n = 8
    drafter = stub_drafter()
    worker = build_worker("helix16", drafter)
    spec = sampled_spec(n, seed=77)
    markov = Markov()
    bufs = Bufs.make(n, rounds=1)
    bufs.last.copy_(torch.arange(n, device="cuda") * 5 % VOCAB)
    drafts = worker._sample_block_path(
        drafter, markov.draft_base[bufs.last], bufs.hidden, bufs.last, spec, 0, n
    )
    store = worker._dflash2_proposals
    for row in invalidate:
        store.tokens[row] = -1
    spec.real_draft_rows = real
    spec.draft_tokens.copy_(drafts.reshape(-1))
    prefix = torch.cat((bufs.last[:, None], drafts.long()), dim=1)
    logits = markov.log_trans[prefix].view(n * (K + 1), VOCAB).contiguous()
    attn = SimpleNamespace(num_seqs=n, num_contexts=0)
    accepted, num = worker._verify_sampled_block(logits, attn, spec)
    torch.cuda.synchronize()
    return accepted, num


def test_placeholder_rows_verify_against_the_point_mass():
    real = torch.tensor([True, False, True, True, False, True, True, True], device="cuda")
    placeholders = [i for i, r in enumerate(real.tolist()) if not r]
    marked = _verify(real)
    invalidated = _verify(None, invalidate=placeholders)
    unmarked = _verify(None)
    assert torch.equal(marked[0], invalidated[0]) and torch.equal(marked[1], invalidated[1])
    for row in range(8):
        if row not in placeholders:
            assert torch.equal(marked[0][row], unmarked[0][row])
            assert int(marked[1][row]) == int(unmarked[1][row])
    # The control: the walked q and the point mass give other counts on some placeholder row.
    assert any(int(marked[1][r]) != int(unmarked[1][r]) for r in placeholders) or any(
        not torch.equal(marked[0][r], unmarked[0][r]) for r in placeholders
    ), "the placeholder rows verified the same with and without the point mass"
