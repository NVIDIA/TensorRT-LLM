# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the FP8 unified target's `SpeculativeRollback`: after a simulated verify step,
every pool and plane is bitwise that of a run over the accepted prefixes only, compared in full
so a write to a slot the step does not own fails too."""

from __future__ import annotations

import itertools

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rollback_planes import (  # noqa: E501
    plane_table as _plane_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ngram_token_history import (  # noqa: E501
    ngram_token_history,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.modeling import (  # noqa: E501
    SpeculativeRollback,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_index_store import (  # noqa: E501
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.spec_rollback_begin__verify import (  # noqa: E501
    spec_rollback_begin,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the rollback handler test requires a CUDA device"

DEV = "cuda"
EOS, VOCAB, NGRAM = 248044, 248320, 3
KERNEL, DILATION, CHANNELS = 4, 3, 64
STATE_LEN = (KERNEL - 1) * DILATION
TPB, INDEX_DIM, LAYERS, COMPRESS = 32, 128, 3, 4
SLOTS, PAGES, BLOCKS_PER_SEQ = 24, 48, 4


class _Batch:
    """One verify step: fresh context requests of `ctx_lens` rows, then generation requests of
    `p` rows at `gen_starts` cached tokens, in `slots`. Generation request `s` reuses request
    `dup_of[s]`'s block-table row, as the CUDA-graph padding dummy does."""

    def __init__(self, seed, ctx_lens, gen_starts, p, slots, dup_of=None):
        g = torch.Generator().manual_seed(seed)
        self.ctx_lens, self.num_contexts, self.p = list(ctx_lens), len(ctx_lens), p
        self.n = len(gen_starts)
        self.slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
        assert self.slots.shape[0] == self.num_contexts + self.n
        lengths = self.ctx_lens + [p] * self.n
        starts = [0] * self.num_contexts + list(gen_starts)
        perm = torch.randperm(PAGES, generator=g)
        table = torch.reshape(perm[: len(lengths) * BLOCKS_PER_SEQ], [len(lengths), -1])
        for dup, original in (dup_of or {}).items():
            table[self.num_contexts + dup] = table[self.num_contexts + original]
        self.block_table = table.to(DEV, torch.int32)
        req, logical = [], []
        for r, (length, start) in enumerate(zip(lengths, starts)):
            req += [r] * length
            logical += range(start, start + length)
        self.rows = len(req)
        self.prefix = sum(self.ctx_lens)
        self.req_idx = torch.tensor(req, dtype=torch.int32, device=DEV)
        self.logical = torch.tensor(logical, dtype=torch.int64, device=DEV)
        self.carry = torch.tensor(_carried(self), dtype=torch.bool, device=DEV)
        self.lengths = lengths
        self.qsl = _qsl(lengths)

        # Every pre-step pool and plane is random, so a stray write shows.
        self.ngram0 = torch.randint(0, VOCAB, (SLOTS, NGRAM - 1), generator=g).to(DEV)
        self.conv0 = torch.randn(SLOTS, CHANNELS, STATE_LEN, generator=g).to(DEV, torch.bfloat16)
        self.index0 = [
            torch.randn(PAGES, TPB, 1, INDEX_DIM, generator=g).to(DEV, torch.bfloat16)
            for _ in range(LAYERS)
        ]
        self.position0 = torch.randint(0, 1 << 20, (PAGES, TPB, 3), generator=g).to(
            DEV, torch.int32
        )
        self.ids = torch.randint(0, VOCAB, (self.rows,), generator=g).to(DEV, torch.int32)
        self.x = torch.randn(self.rows, CHANNELS, generator=g).to(DEV, torch.bfloat16)
        self.keys = [
            torch.randn(self.rows, 1, INDEX_DIM, generator=g).to(DEV, torch.bfloat16)
            for _ in range(LAYERS)
        ]
        self.anchor_keys = [
            torch.randn(self.rows, 1, INDEX_DIM, generator=g).to(DEV, torch.bfloat16)
            for _ in range(LAYERS)
        ]
        self.coords = torch.randint(0, 1 << 20, (self.rows, 3), generator=g).to(DEV, torch.int32)
        self.ngram, self.conv = self.ngram0.clone(), self.conv0.clone()
        self.index, self.position = [t.clone() for t in self.index0], self.position0.clone()
        self.accepted = torch.ones(self.num_contexts + self.n, dtype=torch.int32, device=DEV)

    def reset(self):
        self.ngram.copy_(self.ngram0)
        self.conv.copy_(self.conv0)
        for live, pre in zip(self.index, self.index0):
            live.copy_(pre)
        self.position.copy_(self.position0)

    def set_accepted(self, gen_accepted):
        assert len(gen_accepted) == self.n
        assert all(1 <= a <= self.p for a in gen_accepted)
        self.gen_accepted = list(gen_accepted)
        values = [1] * self.num_contexts + list(gen_accepted)
        self.accepted.copy_(torch.tensor(values, dtype=torch.int32))


def _qsl(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int64, device=DEV)


def _conv_state(conv, x, lengths, slots, carry, publish_end=None):
    """Stand-in for `ple_gate_conv_inject_norm`'s conv state: per sequence below `publish_end`, the
    last `STATE_LEN` columns of `[carried state or zeros | its input]`, by capturable slices."""
    start = 0
    for s, length in enumerate(lengths):
        if publish_end is not None and s >= publish_end:
            break
        slot = slots[s : s + 1].long()
        prev = conv[slot][0] if carry[s] else torch.zeros_like(conv[0])
        window = torch.cat([prev, x[start : start + length].t()], 1)
        conv[slot] = window[:, -STATE_LEN:].unsqueeze(0)
        start += length


def _carried(b):
    """Per sequence, whether it carries a state (generation) or starts fresh (context)."""
    return [False] * b.num_contexts + [True] * b.n


def _write_planes(b, layer, plane, position, keep=slice(None)):
    """The index store over rows `keep`, then a compression stand-in overwriting anchor slots."""
    req_idx, logical, anchor_keys = b.req_idx[keep], b.logical[keep], b.anchor_keys[layer][keep]
    keys, coords = b.keys[layer][keep], b.coords[keep]
    qsa_paged_index_store(keys, coords, plane, position, req_idx, logical, b.block_table, TPB)
    pages = b.block_table[req_idx.long(), logical // TPB].long()
    within = logical % TPB
    anchor = ((logical + 1) % COMPRESS == 0).reshape(-1, 1, 1)
    plane[pages, within] = torch.where(anchor, anchor_keys, plane[pages, within])


def _verify_step(b, rollback, fused, begin_position=False):
    """The forward's writes over all `p` rows, staged at its call sites, then the commit. Under
    `fused` the generation conv rows stay unpublished and the commit rebuilds them from the pool."""
    rollback.discard()
    rollback.begin(
        num_contexts=b.num_contexts,
        prefix_tokens=b.prefix,
        num_gens=b.n,
        rows_per_gen=b.p,
        slots=b.slots[b.num_contexts :],
        request_indices=b.req_idx[b.prefix :],
        logical_positions=b.logical[b.prefix :],
        block_table=b.block_table,
        tokens_per_block=TPB,
        fused=fused,
        **(dict(position_cache=b.position) if begin_position else {}),
    )
    rollback.stage_ple_verify(b.ngram, b.conv, b.ids)
    ngram_token_history(b.ids, b.ngram, b.slots, b.carry, b.qsl, EOS, NGRAM)
    rollback.stage_ple_conv_input(b.x)
    _conv_state(b.conv, b.x, b.lengths, b.slots, _carried(b), b.num_contexts if fused else None)
    for layer in range(LAYERS):
        rollback.stage_index(b.position)
        _write_planes(b, layer, b.index[layer], b.position)
    rollback.commit_speculative_states(b.accepted, b.slots, b.num_contexts)


def _reference(b):
    """The same writes from the pre-step state over context rows and accepted prefixes only."""
    keep = list(range(b.prefix))
    lengths = list(b.ctx_lens)
    for s, a in enumerate(b.gen_accepted):
        first = b.prefix + s * b.p
        keep += range(first, first + a)
        lengths.append(a)
    keep = torch.tensor(keep, dtype=torch.long, device=DEV)
    ngram, conv = b.ngram0.clone(), b.conv0.clone()
    index = [t.clone() for t in b.index0]
    position = b.position0.clone()
    ngram_token_history(b.ids[keep], ngram, b.slots, b.carry, _qsl(lengths), EOS, NGRAM)
    _conv_state(conv, b.x[keep], lengths, b.slots, _carried(b))
    for layer in range(LAYERS):
        _write_planes(b, layer, index[layer], position, keep)
    return ngram, conv, index, position


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _assert_state(b, expected, what):
    ngram, conv, index, position = expected
    assert torch.equal(b.ngram, ngram), f"{what}: n-gram pool"
    assert torch.equal(_bits(b.conv), _bits(conv)), f"{what}: conv pool"
    for layer in range(LAYERS):
        assert torch.equal(_bits(b.index[layer]), _bits(index[layer])), f"{what}: index {layer}"
    assert torch.equal(b.position, position), f"{what}: position plane"


# (ctx_lens, gen_starts, p, gen_accepted): positions straddle pages and land on compression
# anchors; p is 4 (MTP-3) behind contexts, and 2 and 8 (max_draft_len 1 and 7) in decode alone.
STEPS = {
    "mixed": ((7, 3), (12, 29, 95), 4, (2, 1, 3)),
    "decode_p2": ((), (3, 31, 64), 2, (1, 2, 1)),
    "decode_p8": ((), (0, 27, 60, 90, 101, 5, 44, 119), 8, (1, 2, 3, 4, 5, 6, 7, 8)),
}
#: Acceptance counts one captured decode_p8 step is replayed with (bonus token included).
_REPLAYS = [(1,) * 8, (8, 7, 6, 5, 4, 3, 2, 1), (2, 8, 1, 7, 3, 6, 4, 5), (8,) * 8]


def _slots(num):
    return list(range(3, 3 + 2 * num, 2))


#: Handler forms: torch staging with batched plane gather; `fused` (one-launch stage and commit);
#: `fused_begin`, opened with the position plane so one launch stages all, as in a captured decode.
MODES = ["batched", "fused", "fused_begin"]


def _mode_args(mode):
    """`(fused, begin_position)` of a handler form."""
    return mode != "batched", mode == "fused_begin"


def _rollback(b):
    rollback = SpeculativeRollback()
    rollback.set_index_planes(b.index)
    return rollback


def _eager(case, mode):
    ctx_lens, gen_starts, p, gen_accepted = STEPS[case]
    b = _Batch(11, ctx_lens, gen_starts, p, _slots(len(ctx_lens) + len(gen_starts)))
    b.set_accepted(gen_accepted)
    _verify_step(b, _rollback(b), *_mode_args(mode))
    _assert_state(b, _reference(b), case)


def _captured(case, mode):
    """The step and commit captured once, replayed with new acceptances over reset state."""
    ctx_lens, gen_starts, p, _ = STEPS[case]
    b = _Batch(18, ctx_lens, gen_starts, p, _slots(len(gen_starts)))
    rollback = _rollback(b)
    b.set_accepted((p,) * len(gen_starts))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _verify_step(b, rollback, *_mode_args(mode))
    torch.cuda.current_stream().wait_stream(side)
    b.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _verify_step(b, rollback, *_mode_args(mode))
    for accepted in _REPLAYS:
        b.reset()
        b.set_accepted(accepted)
        graph.replay()
        torch.cuda.synchronize()
        _assert_state(b, _reference(b), f"replay {accepted}")


def _padded(case, mode):
    """Requests 2 and 3 are the CUDA-graph padding dummy request 1 is: one slot, one block-table
    row, written concurrently with no defined winner, so only the unique request is compared."""
    gen_starts, p = (5, 40, 40, 40), 4
    b = _Batch(17, (), gen_starts, p, [3, 7, 7, 7], dup_of={2: 1, 3: 1})
    b.set_accepted((2, 3, 3, 3))
    _verify_step(b, _rollback(b), *_mode_args(mode))
    pages = b.block_table[0].long()
    expected = _reference(_single(b, 0))
    assert torch.equal(b.ngram[3], expected[0][3])
    assert torch.equal(_bits(b.conv[3]), _bits(expected[1][3]))
    for layer in range(LAYERS):
        assert torch.equal(_bits(b.index[layer][pages]), _bits(expected[2][layer][pages])), (
            f"index {layer}"
        )
    assert torch.equal(b.position[pages], expected[3][pages])


def _single(b, s):
    """`b` restricted to its generation request `s`, same pre-step state."""
    one = _Batch.__new__(_Batch)
    one.__dict__.update(b.__dict__)
    rows = torch.arange(b.prefix + s * b.p, b.prefix + (s + 1) * b.p, device=DEV)
    one.ctx_lens, one.num_contexts, one.n, one.prefix = [], 0, 1, 0
    one.lengths = [b.p]
    one.slots = b.slots[b.num_contexts + s : b.num_contexts + s + 1]
    one.carry = b.carry[b.num_contexts + s : b.num_contexts + s + 1]
    one.qsl = _qsl(one.lengths)
    one.ids, one.x = b.ids[rows], b.x[rows]
    one.keys = [k[rows] for k in b.keys]
    one.anchor_keys = [k[rows] for k in b.anchor_keys]
    one.coords = b.coords[rows]
    one.logical = b.logical[rows]
    one.req_idx = torch.zeros_like(b.req_idx[rows])
    one.block_table = b.block_table[b.num_contexts + s : b.num_contexts + s + 1]
    one.gen_accepted = [b.gen_accepted[s]]
    return one


_RUNS = (
    [(_eager, case, mode) for case in STEPS for mode in MODES]
    + [(_captured, "decode_p8", mode) for mode in ("fused", "fused_begin")]
    + [(_padded, "dup", "fused")]
)


@pytest.mark.parametrize(
    "run,case,mode", _RUNS, ids=[f"{r.__name__[1:]}-{c}-{m}" for r, c, m in _RUNS]
)
def test_commit_leaves_the_accepted_prefix_state(run, case, mode):
    """After partial acceptance, the n-gram and conv pools and every index and position plane
    are bitwise those of a run over the context rows and the accepted prefixes only."""
    run(case, mode)


@pytest.mark.parametrize(
    "cell", spec_rollback_begin.CELLS, ids=[c.why[:40] for c in spec_rollback_begin.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    """Bitwise the composed torch reference, eagerly and in a capture replayed at new slots."""
    certifying(spec_rollback_begin)
    n, p = cell.spec["requests"], cell.spec["p"]
    starts = [(37 * s + 5) % 100 for s in range(n)]
    b = _Batch(23 + n * p, (), starts, p, _slots(n))
    planes = [torch.randn_like(t) for t in b.index0] * (cell.spec["planes"] // LAYERS)
    ptrs, strides = _plane_table(planes)
    args = (b.req_idx, b.logical, b.block_table, TPB)
    got = spec_rollback_begin(*args, ptrs, strides, b.position0)
    want = spec_rollback_begin.reference(*args, planes, b.position0)
    for g, w in zip(got, want):
        assert torch.equal(_bits(g), _bits(w))
    if not cell.spec["capture"]:
        return
    logical = b.logical.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = spec_rollback_begin(
            b.req_idx, logical, b.block_table, TPB, ptrs, strides, b.position0
        )
    logical.add_(3)
    graph.replay()
    torch.cuda.synchronize()
    want = spec_rollback_begin.reference(
        b.req_idx, logical, b.block_table, TPB, planes, b.position0
    )
    for g, w in zip(outs, want):
        assert torch.equal(_bits(g), _bits(w)), "replay at new slots"
