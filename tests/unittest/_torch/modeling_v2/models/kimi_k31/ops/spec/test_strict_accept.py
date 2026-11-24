# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 strict_accept entry (integer ops: exact)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.strict_accept import (
    strict_accept,
)

__extra_import_path__ = ["../../../..", "."]
from _k31_spec_inputs import on_sm103, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

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


@pytest.mark.parametrize("cell", strict_accept.CELLS, ids=[c.why[:48] for c in strict_accept.CELLS])
def test_certified_cells_strict_accept(cell, certifying) -> None:  # noqa: F811 — by name
    certifying(strict_accept)
    spec = cell.spec
    target_tokens, drafts = _accept_inputs(spec, seed_of(cell))
    if "draft_buffer_width" in spec:
        assert not drafts.is_contiguous()
    num_contexts = spec["num_contexts"]
    batch_size = num_contexts + spec["num_gens"]
    accepted, num_accepted = strict_accept(target_tokens, drafts, num_contexts, batch_size)
    ref_accepted, ref_num = strict_accept.reference(target_tokens, drafts, num_contexts, batch_size)
    assert accepted.shape == (batch_size, spec["draft_len"] + 1) and accepted.dtype == torch.int32
    assert num_accepted.shape == (batch_size,) and num_accepted.dtype == torch.int32
    assert torch.equal(num_accepted, ref_num)
    # Context rows define column 0 only.
    assert torch.equal(accepted[:num_contexts, 0], ref_accepted[:num_contexts, 0])
    assert torch.equal(accepted[num_contexts:], ref_accepted[num_contexts:])


def test_every_acceptance_length() -> None:
    """Each count 1 .. K + 1 for K = 4 and 7, row by row."""
    for draft_len in (4, 7):
        width = draft_len + 1
        target = torch.arange(width * width, device="cuda", dtype=torch.int32).view(width, width)
        drafts = target[:, :draft_len].clone()
        for row in range(width - 1):
            drafts[row, row] = -1  # the first mismatch at column `row`
        accepted, num = strict_accept(target.flatten(), drafts, 0, width)
        assert num.tolist() == list(range(1, width + 1)), (draft_len, num.tolist())
        assert torch.equal(accepted, target)


def test_cuda_graph_replay() -> None:
    target_tokens, drafts = _accept_inputs(dict(num_contexts=0, num_gens=8, draft_len=4), seed=3)
    strict_accept(target_tokens, drafts, 0, 8)  # Triton compiles outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        accepted, num = strict_accept(target_tokens, drafts, 0, 8)
    for seed in range(3):
        new_target, new_drafts = _accept_inputs(
            dict(num_contexts=0, num_gens=8, draft_len=4), seed=10 + seed
        )
        target_tokens.copy_(new_target)
        drafts.copy_(new_drafts)
        graph.replay()
        ref_accepted, ref_num = strict_accept.reference(target_tokens, drafts, 0, 8)
        assert torch.equal(num, ref_num) and torch.equal(accepted, ref_accepted)


def test_guard_refuses_a_short_target_vector() -> None:
    """Two generation rows of w5 need 10 target tokens; 8 are read past their end."""
    target = torch.zeros(8, dtype=torch.int32, device="cuda")
    drafts = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="out of bounds"):
        with validating(strict_accept):
            strict_accept(target, drafts, 0, 2)


def test_guard_refuses_a_strided_target_vector() -> None:
    """The kernel addresses target_tokens densely: a strided view is read from the wrong slots."""
    target = torch.zeros(20, dtype=torch.int32, device="cuda")[::2]
    drafts = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(strict_accept):
            strict_accept(target, drafts, 0, 2)
