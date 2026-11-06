# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_path_sample``: bitwise the built-in DFlash 2 sampled walk.

The reference is ``dflash2_sample_candidate_paths`` with the sampling step
``DFlashWorker._sample_dflash2_block`` runs (``nan_to_num``, the request's
Philox offset for the position's window slot, the shipped
``fused_sample_from_logits_with_probs``), then that method's token gather and
``draft_probs`` scatter. Tokens, distributions and ``draft_probs`` are compared
bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    draft_path_sample as dps_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    head_sampler,
)
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm107 head sampler library is built for sm_107a",
)

TOP_K = 16
POSITIONS = 7
VOCAB = 4096
#: The window slot of draft position 0: after a request's target rows.
SLOT_BASE = 8
#: Rows past the batch in the per-request buffers, as the persistent buffers have.
SPARE = 3

draft_path_sample = dps_mod.DraftPathSample()


@pytest.fixture(scope="module", autouse=True)
def library():
    head_sampler.load_library()


def _sampling(kind: str, requests: int, g: torch.Generator):
    rows = requests + SPARE
    ones = torch.ones(rows, device="cuda")
    if kind == "served":
        # The AgentX requests: the checkpoint's generation config (temperature 1, top_k 64, top_p 1).
        return ones, torch.full((rows,), 64, dtype=torch.int32, device="cuda"), ones, ones * 0
    temps = torch.rand(rows, generator=g, device="cuda") * 1.2 + 0.3
    top_ks = torch.randint(0, TOP_K + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    top_ps = torch.rand(rows, generator=g, device="cuda") * 0.5 + 0.5
    min_ps = torch.rand(rows, generator=g, device="cuda") * 0.2
    if kind == "mixed":
        pick = torch.randint(0, 3, (rows,), generator=g, device="cuda")
        temps = torch.where(pick == 0, ones, temps)
        top_ks = torch.where(pick == 0, 64, top_ks).to(torch.int32)
        top_ps = torch.where(pick == 0, ones, top_ps)
        min_ps = torch.where(pick == 0, 0.0, min_ps)
        # Greedy rows carry the sampler's tiny-temperature sentinel.
        temps = torch.where(pick == 1, 1e-5, temps)
    return temps, top_ks, top_ps, min_ps


def _lattice(requests: int, g: torch.Generator, degenerate: bool = False):
    unary = torch.randn(requests, POSITIONS, TOP_K, generator=g, device="cuda") * 3 + 10
    edges = torch.randn(requests, POSITIONS, TOP_K, TOP_K, generator=g, device="cuda") * 2
    keys = torch.rand(requests * POSITIONS, VOCAB, generator=g, device="cuda")
    ids = keys.argsort(dim=-1)[:, :TOP_K].reshape(requests, POSITIONS, TOP_K).contiguous()
    if degenerate:
        flat = edges.view(-1)
        n = flat.numel()
        idx = torch.randint(0, n, (3, max(1, n // 50)), generator=g, device="cuda")
        flat[idx[0]] = float("nan")
        flat[idx[1]] = float("inf")
        flat[idx[2]] = float("-inf")
        # One whole row under every predecessor: all -inf, so nan_to_num leaves zeros.
        edges[0, POSITIONS // 2] = float("-inf")
        unary[0, POSITIONS // 2] = float("-inf")
    return edges.contiguous(), unary.contiguous(), ids


def _rng(requests: int, g: torch.Generator):
    rows = requests + SPARE
    seed = torch.randint(0, 2**62, (rows,), generator=g, device="cuda")
    offset = torch.randint(0, 2**40, (rows,), generator=g, device="cuda") * 1024
    return seed, offset


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def _run(requests, kind, seed=0, degenerate=False, with_probs=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    edges, unary, ids = _lattice(requests, g, degenerate)
    sampling = _sampling(kind, requests, g)
    rng_seed, rng_offset = _rng(requests, g)
    probs = slots = ref_probs = None
    if with_probs:
        num_slots = requests + SPARE + 1
        slots = torch.randperm(num_slots, generator=g, device="cuda")[: requests + SPARE]
        probs = torch.zeros(num_slots, POSITIONS + 1, VOCAB, device="cuda")
        ref_probs = probs.clone()
    args = (edges, unary, ids, *sampling, rng_seed, rng_offset, SLOT_BASE, _RNG_SLOT_SPAN)
    draft_path_sample.is_valid(*args, probs, slots)
    tokens, proposal = draft_path_sample(*args, probs, slots)
    want_tokens, want_proposal = draft_path_sample.reference(*args, ref_probs, slots)
    torch.cuda.synchronize()
    assert torch.equal(tokens, want_tokens)
    assert _same_bits(proposal, want_proposal)
    if with_probs:
        assert _same_bits(probs, ref_probs)
    return tokens, ids


@pytest.mark.parametrize("requests", [1, 3, 16, 128, 129])
@pytest.mark.parametrize("kind", ["served", "filtered", "mixed"])
def test_the_builtin_sampled_walk(requests, kind):
    tokens, ids = _run(requests, kind, seed=requests)
    # Every token is one of its position's candidates.
    assert (tokens.long().unsqueeze(-1) == ids).any(-1).all()


@pytest.mark.parametrize("kind", ["served", "filtered"])
def test_degenerate_lattice_rows(kind):
    _run(5, kind, seed=11, degenerate=True)


def test_without_draft_probs():
    _run(4, "served", seed=7, with_probs=False)


def test_seeds_and_offsets_drive_the_draws():
    g = torch.Generator(device="cuda").manual_seed(3)
    edges, unary, ids = _lattice(8, g)
    sampling = _sampling("served", 8, g)
    rng_seed, rng_offset = _rng(8, g)
    first, _ = draft_path_sample(
        edges, unary, ids, *sampling, rng_seed, rng_offset, SLOT_BASE, _RNG_SLOT_SPAN
    )
    again, _ = draft_path_sample(
        edges, unary, ids, *sampling, rng_seed, rng_offset, SLOT_BASE, _RNG_SLOT_SPAN
    )
    moved, _ = draft_path_sample(
        edges, unary, ids, *sampling, rng_seed, rng_offset + 1, SLOT_BASE, _RNG_SLOT_SPAN
    )
    assert torch.equal(first, again)
    assert not torch.equal(first, moved)
