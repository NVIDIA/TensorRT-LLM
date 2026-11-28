# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 strict_accept entry (integer ops: exact)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.strict_accept import strict_accept

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import on_sm103, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

VOCAB = 163840


def _accept_inputs(spec: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Drafts that agree with the target for a random prefix of each row, then diverge."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_contexts, num_gens, draft_len = spec["num_contexts"], spec["num_gens"], spec["draft_len"]
    width = draft_len + 1
    ctx = torch.randint(0, VOCAB, (num_contexts,), generator=gen, device="cuda")
    target = torch.randint(0, VOCAB, (num_gens, width), generator=gen, device="cuda")
    buffer_width = spec.get("draft_buffer_width", draft_len)
    drafts = torch.randint(0, VOCAB, (num_gens, buffer_width), generator=gen, device="cuda")
    agree = torch.randint(0, width, (num_gens,), generator=gen, device="cuda")
    cols = torch.arange(draft_len, device="cuda")
    drafts[:, :draft_len] = torch.where(
        cols[None, :] < agree[:, None], target[:, :draft_len], drafts[:, :draft_len]
    )
    target_tokens = torch.cat((ctx, target.flatten())).to(torch.int32)
    # Cast the whole buffer, then slice: a strided view stays strided.
    return target_tokens, drafts.to(torch.int32)[:, :draft_len]


CELLS = sample_cells(strict_accept.CELLS, size_key="num_gens")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells_strict_accept(cell, certifying) -> None:  # noqa: F811 — by name
    certifying(strict_accept)
    spec = cell.spec
    target_tokens, drafts = _accept_inputs(spec, seed_of(cell))
    num_contexts = spec["num_contexts"]
    batch_size = num_contexts + spec["num_gens"]
    accepted, num_accepted = strict_accept(target_tokens, drafts, num_contexts, batch_size)
    ref_accepted, ref_num = strict_accept.reference(target_tokens, drafts, num_contexts, batch_size)
    assert torch.equal(num_accepted, ref_num)
    # Context rows define column 0 only.
    assert torch.equal(accepted[:num_contexts, 0], ref_accepted[:num_contexts, 0])
    assert torch.equal(accepted[num_contexts:], ref_accepted[num_contexts:])
