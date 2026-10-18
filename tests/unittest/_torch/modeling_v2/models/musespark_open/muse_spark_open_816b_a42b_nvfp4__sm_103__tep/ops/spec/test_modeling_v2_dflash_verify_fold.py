# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``dflash_verify_fold`` (fast path ``verify_fold``).

The entry's claim is bitwise equality with the built-in chain -- ``dflash_accept``
and the 18 launches of DFlash bookkeeping behind it -- so every output, and
the in-place ``kv_lens`` and forced-acceptance counter, is gated with
``torch.equal`` (dtype and shape included) against the entry's ``reference``
(the built-in lines unrolled), and its acceptance against ``dflash_accept``
itself, over many steps, eagerly and replayed from a CUDA graph.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_accept import (  # noqa: E501
    FORCE_FRACTIONAL,
    dflash_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_verify_fold import (  # noqa: E501
    DflashVerifyFold,
    VerifyFold,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_verify_fold import (  # noqa: E501
    dflash_verify_fold as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "dflash_verify_fold requires a CUDA device"

VOCAB = 512
SLOTS = 9
MASK_ID = 202047
MAX_CTX = 1 << 20


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _force_args(rate, k):
    int_part = int(rate)
    frac = rate - int_part
    max_total = k + 1
    return min(int_part + 1, max_total), max_total, frac


class _State:
    """The worker and metadata buffers the fold reads and writes, and an independent copy."""

    def __init__(self, gens, g, near_ceiling=False, bare=False):
        self.pool = torch.rand(1 << 16, generator=g, device="cuda", dtype=torch.float32)
        self.counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.slots = torch.randperm(SLOTS, generator=g, device="cuda")[:gens].long().contiguous()
        top = MAX_CTX if near_ceiling else 1 << 18
        low = MAX_CTX - 5 if near_ceiling else 0
        self.ctx_len = torch.randint(low, top, (SLOTS,), generator=g, device="cuda").long()
        self.kv_lens = (
            None if bare else torch.randint(0, 1 << 18, (16,), generator=g, device="cuda").int()
        )
        self.seq_lens = (
            None if bare else torch.randint(1, 9, (16,), generator=g, device="cuda").int()
        )

    def clone(self):
        other = object.__new__(_State)
        for name, value in vars(self).items():
            setattr(other, name, None if value is None else value.clone())
        return other


def _tokens(gens, k, g, int64=False):
    target = torch.randint(0, VOCAB, (gens * (k + 1),), generator=g, device="cuda")
    target = target.to(torch.int64 if int64 else torch.int32)
    gen_targets = target.view(gens, k + 1)
    draft = torch.randint(0, VOCAB, (gens, k), generator=g, device="cuda").to(torch.int32)
    lead = torch.randint(0, k + 1, (gens,), generator=g, device="cuda")
    keep = torch.arange(k, device="cuda").unsqueeze(0) < lead.unsqueeze(1)
    return target, torch.where(keep, gen_targets[:, :k].to(torch.int32), draft).contiguous()


def _call(fn, target, draft, spec, state, block):
    base_total, max_total, frac = _force_args(spec["rate"], spec["k"])
    return fn(
        target,
        draft,
        spec["force"],
        base_total,
        max_total,
        frac,
        state.pool,
        state.counter,
        state.slots,
        state.ctx_len,
        MAX_CTX,
        block,
        MASK_ID,
        state.kv_lens,
        state.seq_lens,
    )


def _check(out: VerifyFold, ref: VerifyFold) -> None:
    for name in VerifyFold._fields:
        a, b = getattr(out, name), getattr(ref, name)
        if b is None:
            assert a is None, name
            continue
        assert a.dtype == b.dtype, (name, a.dtype, b.dtype)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert torch.equal(a, b), (name, a, b)


def _check_state(state: _State, ref: _State) -> None:
    for name in ("counter", "kv_lens", "seq_lens", "ctx_len", "slots"):
        a, b = getattr(state, name), getattr(ref, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name


@pytest.mark.parametrize(
    "cell", DflashVerifyFold.CELLS, ids=[c.why[:48] for c in DflashVerifyFold.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    state = _State(spec["gens"], g, spec.get("near_ceiling", False), spec.get("bare", False))
    ref_state = state.clone()
    accept_counter = state.counter.clone()
    for _ in range(40):
        target, draft = _tokens(spec["gens"], spec["k"], g, spec.get("int64", False))
        before = (target.clone(), draft.clone(), state.ctx_len.clone(), state.slots.clone())
        out = _call(op, target, draft, spec, state, spec["block"])
        ref = _call(op.reference, target, draft, spec, ref_state, spec["block"])
        _check(out, ref)
        _check_state(state, ref_state)
        # The acceptance is dflash_accept's, launch for launch.
        base_total, max_total, frac = _force_args(spec["rate"], spec["k"])
        accepted, num_accepted = dflash_accept(
            target, draft, 0, spec["force"], base_total, max_total, frac, state.pool, accept_counter
        )
        assert torch.equal(accepted, out.accepted) and torch.equal(num_accepted, out.num_accepted)
        assert torch.equal(accept_counter, state.counter)
        assert torch.equal(before[0], target) and torch.equal(before[1], draft), "inputs mutated"
        assert torch.equal(before[2], state.ctx_len), (
            "_ctx_len is the arena update's, not the fold's"
        )
        # Move the context lengths as the arena update would, so later steps see new values.
        state.ctx_len[state.slots] += out.num_accepted_long
        ref_state.ctx_len[ref_state.slots] += ref.num_accepted_long
        state.ctx_len.clamp_(max=MAX_CTX)
        ref_state.ctx_len.clamp_(max=MAX_CTX)


def test_fractional_override_over_the_whole_pool():
    """Every pool slot the counter reaches, against dflash_accept's comparison."""
    spec = dict(gens=8, k=7, block=8, force=FORCE_FRACTIONAL, rate=3.05)
    g = torch.Generator(device="cuda").manual_seed(3)
    state = _State(8, g)
    ref_state = state.clone()
    target = torch.zeros(8 * 8, dtype=torch.int32, device="cuda")
    draft = torch.zeros(8, 7, dtype=torch.int32, device="cuda")
    for _ in range((1 << 16) // 8 + 16):
        _check(
            _call(op, target, draft, spec, state, 8),
            _call(op.reference, target, draft, spec, ref_state, 8),
        )
    _check_state(state, ref_state)


def test_cuda_graph_replay_matches_the_reference():
    spec = dict(gens=1, k=7, block=8, force=FORCE_FRACTIONAL, rate=3.05)
    g = torch.Generator(device="cuda").manual_seed(17)
    state = _State(1, g)
    target, draft = _tokens(1, 7, g)
    _call(op, target, draft, spec, state.clone(), 8)  # compile outside the capture
    ref_state = state.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = _call(op, target, draft, spec, state, 8)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(300):
        new_target, new_draft = _tokens(1, 7, g)
        target.copy_(new_target)
        draft.copy_(new_draft)
        state.ctx_len.copy_(torch.randint(0, 1 << 18, (SLOTS,), generator=g, device="cuda"))
        ref_state.ctx_len.copy_(state.ctx_len)
        graph.replay()
        ref = _call(op.reference, target, draft, spec, ref_state, 8)
        torch.cuda.synchronize()
        _check(out, ref)
        _check_state(state, ref_state)


def test_guard_refuses_what_the_kernel_cannot_take():
    g = torch.Generator(device="cuda").manual_seed(1)
    state = _State(2, g)
    spec = dict(gens=2, k=7, block=8, force=0, rate=0.0)
    target, draft = _tokens(2, 7, g)
    with validating(op):
        with pytest.raises(AssertionError):  # a mixed block: not generation-only
            _call(op, torch.cat([target, target[:1]]), draft, spec, state, 8)
        with pytest.raises(AssertionError):  # int32 slots
            state.slots = state.slots.int()
            _call(op, target, draft, spec, state, 8)
