# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's `engram_lookback_hash` (the dep and tep8 targets' n-gram hash ids and look-back ring update in two
launches): integer arithmetic, gated bit for bit against its plain-torch reference (the targets' own
`_EngramHash.lookback`), over every cell in both storage layouts (tep8 asks for layer-major ids), across consecutive
steps (each step reads what the previous one wrote) and replayed from a CUDA graph."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookback_hash import (  # noqa: E402
    MAX_ROWS,
    engram_lookback_hash,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)

_SLOTS, _RING, _VOCAB = 512, 16, 1000  # the targets' look-back slots


def _hash_inputs(spec, cell):
    gen = gen_for(cell)
    reqs, per, start = spec["reqs"], spec["per"], spec["start"]
    n = reqs * per
    input_ids = torch.randint(0, _VOCAB, (n,), generator=gen, device=DEV, dtype=torch.int32)
    if spec.get("neg"):
        # ids outside the vocabulary clamp to the map's ends in the kernel, the reference and the engine's decode hash
        input_ids[::5] = -1
        input_ids[2::5] = -2
    offset = 7 * torch.arange(reqs, device=DEV).repeat_interleave(per)
    positions = (torch.arange(per, device=DEV).repeat(reqs) + start + offset).to(torch.int32)
    tok_req = torch.arange(n, device=DEV) // per
    req_slot = torch.full((max(128, reqs),), _SLOTS, dtype=torch.int32, device=DEV)
    live = reqs - spec.get("dummy", 0)
    req_slot[:live] = torch.randperm(_SLOTS, generator=gen, device=DEV)[:live].to(torch.int32)
    token_map = torch.randint(0, 900, (_VOCAB,), generator=gen, device=DEV, dtype=torch.int64)
    hist = torch.randint(0, 900, (_SLOTS + 1, _RING), generator=gen, device=DEV, dtype=torch.int32)
    bound = (2**63 - 1) // 900 // 2
    mults = torch.randint(0, bound, (2, 4), generator=gen, device=DEV, dtype=torch.int64) * 2 + 1
    primes = (
        (1000003 + 2 * torch.arange(48, device=DEV, dtype=torch.int64)).view(2, 3, 8).contiguous()
    )
    return input_ids, positions, tok_req, req_slot, token_map, hist, mults, primes, 5


@pytest.mark.parametrize("layer_major", [False, True], ids=["token-major", "layer-major"])
@pytest.mark.parametrize(
    "cell", engram_lookback_hash.CELLS, ids=[c.why[:48] for c in engram_lookback_hash.CELLS]
)
def test_lookback_hash_cells(cell, layer_major, certifying):  # noqa: F811
    certifying(engram_lookback_hash)
    args = _hash_inputs(cell.spec, cell)
    hist_ref = args[5].clone()
    ref = engram_lookback_hash.reference(*args[:5], hist_ref, *args[6:])
    got = engram_lookback_hash(*args, layer_major)
    assert got.dtype == torch.int64 and torch.equal(got, ref)
    if (
        layer_major
    ):  # every Engram layer's ids are contiguous, as tep8's table lookup reads them (no copy)
        assert all(got[:, layer].is_contiguous() for layer in range(got.shape[1]))
    else:
        assert got.is_contiguous()
    # the dummy row's content depends on the write order; every live slot's ring must match
    assert torch.equal(args[5][:_SLOTS], hist_ref[:_SLOTS])


def test_lookback_hash_next_step_reads_the_ring():
    """Consecutive decode steps of 40 requests: each step's look-back comes from what the previous one wrote."""
    cell = SimpleNamespace(why="steps", spec=dict(reqs=40, per=4, start=50))
    args = list(_hash_inputs(cell.spec, cell))
    hist_ref = args[5].clone()
    for _ in range(3):
        ref = engram_lookback_hash.reference(*args[:5], hist_ref, *args[6:])
        got = engram_lookback_hash(*args)
        assert torch.equal(got, ref) and torch.equal(args[5][:_SLOTS], hist_ref[:_SLOTS])
        args[1] = args[1] + 4
        args[0] = (args[0] * 7 + 3) % _VOCAB


def test_lookback_hash_cuda_graph():
    cell = SimpleNamespace(why="graph", spec=dict(reqs=40, per=4, start=300))
    args = list(_hash_inputs(cell.spec, cell))
    hist_ref = args[5].clone()
    engram_lookback_hash(*args)  # compile
    args[5].copy_(hist_ref)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = engram_lookback_hash(*args)
    args[5].copy_(hist_ref)
    for _ in range(3):
        g.replay()
        ref = engram_lookback_hash.reference(*args[:5], hist_ref, *args[6:])
        assert torch.equal(out, ref) and torch.equal(args[5][:_SLOTS], hist_ref[:_SLOTS])
        args[1].add_(4)
        args[0].copy_((args[0] * 7 + 3) % _VOCAB)


def test_lookback_hash_rejections():
    args = _hash_inputs(dict(reqs=1, per=6, start=9), SimpleNamespace(why="rejections", spec={}))
    engram_lookback_hash.is_valid(*args)
    with pytest.raises(AssertionError):
        engram_lookback_hash.is_valid(
            *args[:6], args[6][:, :3].contiguous(), args[7][:, :3].contiguous(), args[8]
        )
    with pytest.raises(AssertionError, match="a program holds"):
        big = torch.zeros(2 * MAX_ROWS, dtype=torch.int32, device=DEV)
        engram_lookback_hash.is_valid(big, big, big.long(), *args[3:])
