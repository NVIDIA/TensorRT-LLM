# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's speculative rollback handler (`SpeculativeRollback`).

A verify step is simulated with the target's own state-writing ops -- the n-gram id
history, the dilated short conv and the QSA index store, plus an anchor-slot overwrite
standing in for the compression -- run over every draft row the way the forward runs
them, with the handler staged at the forward's call sites and committed afterwards.

The reference shares no code with the commit: it runs the same ops from the same
pre-step state over each generation request's accepted prefix only, which is by
definition the state a verify step must leave behind. Every pool and plane is compared
bitwise in full, so a write to a slot the step does not own fails too. The step is also
run sync-free and under CUDA-graph capture and replay with the accepted counts changed
in place.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling import (  # noqa: E501
    SpeculativeRollback,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_index_store import (  # noqa: E501
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.ngram_token_history import (  # noqa: E501
    ngram_token_history,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.dilated_short_conv_paged import (  # noqa: E501
    dilated_short_conv_paged,
)

assert torch.cuda.is_available(), "the rollback handler test requires a CUDA device"

DEV = "cuda"
EOS = 248044
VOCAB = 248320
NGRAM = 3
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION
CHANNELS = 64
TPB = 32
INDEX_DIM = 128
LAYERS = 3
COMPRESS = 4
SLOTS = 24
PAGES = 48
BLOCKS_PER_SEQ = 4


class _Batch:
    """One verify step: `ctx_lens` context requests (fresh) followed by
    generation requests of `p` rows each, at `starts` cached tokens, in the
    recurrent `slots` given (contexts first). Generation request `s` reuses the
    block-table row of request `dup_of[s]` when given -- the CUDA-graph padding
    dummy, repeated. With `shared_page` the live index and position planes are strided
    views of one per-slot buffer, as an NVFP4 KV cache's manager serves them."""

    def __init__(self, seed, ctx_lens, gen_starts, p, slots, dup_of=None, shared_page=False):
        g = torch.Generator().manual_seed(seed)
        self.ctx_lens = list(ctx_lens)
        self.num_contexts = len(ctx_lens)
        self.n = len(gen_starts)
        self.p = p
        self.slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
        assert self.slots.shape[0] == self.num_contexts + self.n
        lengths = self.ctx_lens + [p] * self.n
        starts = [0] * self.num_contexts + list(gen_starts)
        # Distinct pages per request; a duplicate takes its original's row.
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
        self.carry = torch.tensor(
            [False] * self.num_contexts + [True] * self.n, dtype=torch.bool, device=DEV
        )
        self.lengths = lengths
        self.qsl = _qsl(lengths)

        # Pre-step state: every pool and plane random, so a stray write shows.
        self.ngram0 = torch.randint(0, VOCAB, (SLOTS, NGRAM - 1), generator=g).to(DEV)
        self.conv0 = torch.randn(SLOTS, CHANNELS, STATE_LEN, generator=g).to(DEV, torch.bfloat16)
        self.index0 = [
            torch.randn(PAGES, TPB, 1, INDEX_DIM, generator=g).to(DEV, torch.bfloat16)
            for _ in range(LAYERS)
        ]
        self.position0 = torch.randint(0, 1 << 20, (PAGES, TPB, 3), generator=g).to(
            DEV, torch.int32
        )
        self.weight = (0.3 * torch.randn(CHANNELS, 1, KERNEL, generator=g)).to(DEV, torch.bfloat16)

        # Step inputs.
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

        # Live state the step mutates.
        self.ngram = self.ngram0.clone()
        self.conv = self.conv0.clone()
        if shared_page:
            self.index, self.position = _shared_planes(self.index0, self.position0)
        else:
            self.index = [t.clone() for t in self.index0]
            self.position = self.position0.clone()
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


def _shared_planes(index0, position0):
    """Live copies of the planes as views of one byte buffer per page: every layer's index-K
    page, then the position page (what the QSA cache manager packs when an NVFP4 pool's data
    and scale pages would otherwise absorb them)."""
    index_bytes = TPB * INDEX_DIM * 2
    page = torch.zeros(PAGES, LAYERS * index_bytes + TPB * 3 * 4, dtype=torch.uint8, device=DEV)
    index = [
        page[:, layer * index_bytes : (layer + 1) * index_bytes]
        .view(torch.bfloat16)
        .view(PAGES, TPB, 1, INDEX_DIM)
        for layer in range(LAYERS)
    ]
    position = page[:, LAYERS * index_bytes :].view(torch.int32).view(PAGES, TPB, 3)
    assert index[0].stride(0) == page.stride(0) // 2 and not index[0].is_contiguous()
    for live, pre in zip(index, index0):
        live.copy_(pre)
    position.copy_(position0)
    return index, position


def _qsl(lengths):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=torch.int64, device=DEV)


def _write_planes(b, layer_index, position, req_idx, logical, keys, anchor_keys, coords):
    """The index store, then a compression stand-in that overwrites each anchor
    row's own slot -- the two writers of an attention layer's planes."""
    qsa_paged_index_store(keys, coords, layer_index, position, req_idx, logical, b.block_table, TPB)
    pages = b.block_table[req_idx.long(), logical // TPB].long()
    within = logical % TPB
    anchor = ((logical + 1) % COMPRESS == 0).reshape(-1, 1, 1)
    layer_index[pages, within] = torch.where(anchor, anchor_keys, layer_index[pages, within])


def _verify_step(b, rollback):
    """The forward's write sequence over all `p` rows of every generation
    request, staged at the forward's call sites, then the worker's commit."""
    rollback.discard()
    qsl = b.qsl
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
    )
    rollback.stage_ple_pools(b.ngram, b.conv)
    window = ngram_token_history(b.ids, b.ngram, b.slots, b.carry, qsl, EOS, NGRAM)
    rollback.stage_ple_rows(window, b.x)
    dilated_short_conv_paged(
        b.x.t(), b.weight, None, b.conv, qsl, b.slots, b.carry, DILATION, True, -1
    )
    for layer in range(LAYERS):
        rollback.stage_index(b.index[layer], b.position)
        _write_planes(
            b,
            b.index[layer],
            b.position,
            b.req_idx,
            b.logical,
            b.keys[layer],
            b.anchor_keys[layer],
            b.coords,
        )
    rollback.commit_speculative_states(b.accepted, b.slots, b.num_contexts)


def _reference(b):
    """The state after running the same ops from the pre-step state over the
    context rows and each generation request's accepted prefix only."""
    keep = list(range(b.prefix))
    lengths = list(b.ctx_lens)
    for s, a in enumerate(b.gen_accepted):
        first = b.prefix + s * b.p
        keep += range(first, first + a)
        lengths.append(a)
    keep = torch.tensor(keep, dtype=torch.long, device=DEV)
    qsl = _qsl(lengths)
    ngram, conv = b.ngram0.clone(), b.conv0.clone()
    index = [t.clone() for t in b.index0]
    position = b.position0.clone()
    ngram_token_history(b.ids[keep], ngram, b.slots, b.carry, qsl, EOS, NGRAM)
    dilated_short_conv_paged(
        b.x[keep].t(), b.weight, None, conv, qsl, b.slots, b.carry, DILATION, True, -1
    )
    for layer in range(LAYERS):
        _write_planes(
            b,
            index[layer],
            position,
            b.req_idx[keep],
            b.logical[keep],
            b.keys[layer][keep],
            b.anchor_keys[layer][keep],
            b.coords[keep],
        )
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


# (ctx_lens, gen_starts, p, gen_accepted). Positions straddle pages and land on
# compression anchors; one request accepts everything. `decode` and `mixed` are the accepted-
# prefix check; `decode_p8` drives the shared-page and capture tests.
CASES = {
    "decode": ((), (5, 30, 61, 100), 4, (1, 2, 4, 3)),
    "mixed": ((7, 3), (12, 29, 95), 4, (2, 1, 3)),
    # max_draft_len 7: every acceptance 1..8 once; blocks at 27, 60 and 90 straddle a page,
    # and an 8-row block completes two compression groups.
    "decode_p8": ((), (0, 27, 60, 90, 101, 5, 44, 119), 8, (1, 2, 3, 4, 5, 6, 7, 8)),
}


def _slots(num):
    return list(range(3, 3 + 2 * num, 2))


@pytest.mark.parametrize("case", ["decode", "mixed"])
def test_commit_matches_the_accepted_prefix(case):
    ctx_lens, gen_starts, p, gen_accepted = CASES[case]
    b = _Batch(11, ctx_lens, gen_starts, p, _slots(len(ctx_lens) + len(gen_starts)))
    b.set_accepted(gen_accepted)
    _verify_step(b, SpeculativeRollback())
    _assert_state(b, _reference(b), case)


def test_commit_over_the_strided_planes_of_a_shared_page():
    """The handler stages and restores the index and position planes by advanced indexing, so
    planes that are strided views of one buffer (an NVFP4 KV cache's side planes) roll back
    like contiguous ones, and the bytes of the other layers' pages in the same buffer stay as
    the step wrote them."""
    case = "decode_p8"
    ctx_lens, gen_starts, p, gen_accepted = CASES[case]
    b = _Batch(
        13, ctx_lens, gen_starts, p, _slots(len(ctx_lens) + len(gen_starts)), shared_page=True
    )
    b.set_accepted(gen_accepted)
    _verify_step(b, SpeculativeRollback())
    _assert_state(b, _reference(b), case)


def test_commit_without_staging_is_a_no_op():
    ctx_lens, gen_starts, p, gen_accepted = CASES["decode"]
    b = _Batch(14, ctx_lens, gen_starts, p, _slots(len(gen_starts)))
    b.set_accepted(gen_accepted)
    rollback = SpeculativeRollback()
    rollback.commit_speculative_states(b.accepted, b.slots, 0)
    _assert_state(b, (b.ngram0, b.conv0, b.index0, b.position0), "unstaged")


def test_discard_drops_a_stale_staging():
    """A staged step whose commit never ran is dropped by the next forward."""
    ctx_lens, gen_starts, p, gen_accepted = CASES["decode"]
    b = _Batch(15, ctx_lens, gen_starts, p, _slots(len(gen_starts)))
    b.set_accepted(gen_accepted)
    rollback = SpeculativeRollback()
    rollback.begin(
        num_contexts=0,
        prefix_tokens=0,
        num_gens=b.n,
        rows_per_gen=p,
        slots=b.slots,
        request_indices=b.req_idx,
        logical_positions=b.logical,
        block_table=b.block_table,
        tokens_per_block=TPB,
    )
    rollback.discard()
    rollback.commit_speculative_states(b.accepted, b.slots, 0)
    _assert_state(b, (b.ngram0, b.conv0, b.index0, b.position0), "discarded")


def test_commit_is_sync_free():
    ctx_lens, gen_starts, p, gen_accepted = CASES["mixed"]
    b = _Batch(16, ctx_lens, gen_starts, p, _slots(len(ctx_lens) + len(gen_starts)))
    b.set_accepted(gen_accepted)
    rollback = SpeculativeRollback()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _verify_step(b, rollback)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    _assert_state(b, _reference(b), "sync-free")


def test_padding_duplicates_leave_the_unique_request_exact():
    # Requests 2 and 3 are the same padding dummy as request 1: one slot, one
    # block-table row, written concurrently with no defined winner, so only the
    # unique request is compared.
    gen_starts, p = (5, 40, 40, 40), 4
    slots = [3, 7, 7, 7]
    b = _Batch(17, (), gen_starts, p, slots, dup_of={2: 1, 3: 1})
    b.set_accepted((2, 3, 3, 3))
    _verify_step(b, SpeculativeRollback())
    # The unique request's slot is exactly its accepted prefix.
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


#: Per case, the acceptance counts one captured graph is replayed with (the accepted count
#: of every generation request, bonus token included).
_REPLAYS = {
    "decode_p8": [
        (1,) * 8,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (2, 8, 1, 7, 3, 6, 4, 5),
        (8,) * 8,
    ],
}


@pytest.mark.parametrize("case", sorted(_REPLAYS))
def test_capture_and_replay_follow_the_accepted_counts(case):
    """The whole staged step and commit captured once, replayed with the
    accepted counts and the pre-step state rewritten in place."""
    ctx_lens, gen_starts, p, _ = CASES[case]
    b = _Batch(18, ctx_lens, gen_starts, p, _slots(len(gen_starts)))
    rollback = SpeculativeRollback()
    b.set_accepted((p,) * len(gen_starts))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _verify_step(b, rollback)
    torch.cuda.current_stream().wait_stream(side)
    b.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _verify_step(b, rollback)

    for accepted in _REPLAYS[case]:
        b.reset()
        b.set_accepted(accepted)
        graph.replay()
        torch.cuda.synchronize()
        _assert_state(b, _reference(b), f"replay {accepted}")
