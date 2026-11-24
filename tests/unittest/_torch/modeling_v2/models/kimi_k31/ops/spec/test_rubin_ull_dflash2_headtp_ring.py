# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The head-parallel drafter's per-rank window ring against today's helix history pages.

One process plays all 16 CP ranks: each holds its owner pages of the history, written by the
target's own generation-step bookkeeping (``dflash_paged_latent_prepare``) and history store
(``store_context_latents``). The ring is filled from those pages at a request's first generation
step (slot 0 through the per-request ``ring_fill``, the others through one ``ring_fill_batch``;
the all-gather played by concatenation), then written every step by the same store kernel at its
own targets
(the served store: every verify row kept, targets from ``ring_step``, checked against the per-list
helpers; also with the history store's accepted/rejected mask).
After every step, the ring's bytes at every history position a block reads (``[n - W + 1, n)``)
must equal the owner rank's pool bytes, and the block's own positions must hold the block latents
just written; through accept counts 0..7 (accepted drafts, plus the bonus token), contexts shorter
than the window, and ring wrap-around; the first step after the fill accepts every draft.
The batched fill also writes exactly the per-request fill's bytes, over any lengths, slot order
and ring contents, without reading the device from the host.
"""

import contextlib
import random

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_dflash2_headtp_ring import (  # noqa: E501
    fill_pack as fill_pack_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_spec_tail_kernels import (  # noqa: E501
    dflash_paged_latent_prepare,
    store_context_latents,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_headtp_ring import (  # noqa: E501
    build_ring,
    fill_capacity,
    fill_pack,
    ring_block_targets,
    ring_fill,
    ring_fill_batch,
    ring_rows,
    ring_step,
    ring_store,
    ring_store_mask,
    ring_store_targets,
    ring_targets,
    window_checksums,
)

CP, PAGE, W, WIDTH, BLOCK, LAT = 16, 64, 1024, 8, 8, 576

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def _fp8(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def test_ring_rows_and_capacity():
    assert ring_rows(W, WIDTH, BLOCK, PAGE) == 1088
    assert ring_rows(W, WIDTH, BLOCK, PAGE) >= W + BLOCK - 1 + WIDTH + BLOCK
    # [L - W, L) over 16 ranks of 64-position pages: at most one whole page and a partial one.
    assert fill_capacity(W, PAGE, CP) == 128


def test_ring_targets_wrap():
    ring = build_ring(1, 3, W, WIDTH, BLOCK, PAGE, LAT, torch.float8_e4m3fn, "cuda")
    pos = torch.tensor([[0, 63, 64, 1087, 1088, 1089, 5000]], device="cuda")
    pages, offs = ring_targets(ring, torch.tensor([2], device="cuda"), pos)
    rows = [p % 1088 for p in pos[0].tolist()]
    assert pages.tolist() == [2 * 17 + r // 64 for r in rows]
    assert offs.tolist() == [r % 64 for r in rows]


class _Rank:
    """One CP rank's history: pool, page table, lengths, as the target's worker keeps them."""

    def __init__(self, rank, slots, layers, local_pages):
        self.rank = rank
        self.pools_data = torch.zeros(
            (layers, slots * local_pages, 1, 1, PAGE, LAT), dtype=torch.float8_e4m3fn, device="cuda"
        )
        self.pools = [self.pools_data[i] for i in range(layers)]
        self.ptrs = torch.tensor(
            [p.data_ptr() for p in self.pools], dtype=torch.int64, device="cuda"
        )
        self.ctx_len = torch.zeros(slots + 1, dtype=torch.int64, device="cuda")  # + the dummy slot
        self.local_pages = local_pages

    def table(self, slots):
        """Local page tables [len(slots), local_pages] int32: slot s holds pages s * n .. s * n + n - 1."""
        s = torch.tensor(slots, device="cuda", dtype=torch.int32).view(-1, 1)
        steps = torch.arange(self.local_pages, device="cuda", dtype=torch.int32)
        return (s * self.local_pages + steps).contiguous()

    def owner_bytes(self, slot, positions):
        """This rank's pool bytes at ``positions`` (all owned by it): [layers, n, LAT] uint8."""
        local = slot * self.local_pages + (positions // PAGE) // CP
        raw = self.pools_data.view(torch.uint8)
        return raw[:, local, 0, 0, positions % PAGE]


def _helix_bytes(ranks, slot, positions):
    """Today's history bytes at ``positions``, each read from its owner rank: [layers, n, LAT]."""
    owner = (positions // PAGE) % CP
    out = None
    for r in ranks:
        sel = owner == r.rank
        if not bool(sel.any()):
            continue
        got = r.owner_bytes(slot, positions[sel])
        if out is None:
            out = torch.empty(
                (got.shape[0], positions.numel(), LAT), dtype=torch.uint8, device="cuda"
            )
        out[:, sel] = got
    return out


def _meta(rank, slots, lengths):
    """``ring_fill_batch``'s ``[G, 2 + table]`` int32 rows: slot, length, the local page table."""
    head = torch.tensor([[s, n] for s, n in zip(slots, lengths)], dtype=torch.int32, device="cuda")
    return torch.cat([head, rank.table(list(slots))], dim=1).contiguous()


def _fill_batch(ring, ranks, metas):
    """Rank 0's ``ring_fill_batch`` (``metas`` each rank's :func:`_meta`); the all-gather
    concatenates every rank's packed shares. Returns what rank 0 sent."""
    sent = []

    def all_gather(x):
        sent.append(x)
        wires = []
        for r, meta in zip(ranks, metas):
            wire = torch.empty_like(x.view(torch.uint8))
            fill_pack_kernel(meta, r.ptrs, r.pools[0].stride(0), r.pools[0].stride(3), wire,
                             r.rank, ring.window, PAGE, CP)  # fmt: skip
            wires.append(wire.view(torch.bfloat16))
        return torch.cat(wires)

    r0 = ranks[0]
    ring_fill_batch(ring, metas[0], r0.pools, r0.ptrs, PAGE, CP, 0, all_gather)
    return sent


@contextlib.contextmanager
def _no_host_sync():
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _ring_bytes(ring, slot, positions):
    pages, offs = ring_targets(ring, torch.tensor([slot], device="cuda"), positions.view(1, -1))
    return ring.data.view(torch.uint8)[:, pages, 0, 0, offs]


@pytest.mark.parametrize("masked", [False, True], ids=["keep_all", "store_mask"])
@pytest.mark.parametrize("lengths,steps,layers", [((700, 4133), 260, 2), ((5000, 1), 120, 5)])
def test_ring_window_equals_helix_pages(lengths, steps, layers, masked):
    rng = random.Random(sum(lengths) + steps)
    gen = torch.Generator(device="cuda").manual_seed(sum(lengths))
    slots = list(range(len(lengths)))
    g = len(slots)
    max_pos = max(lengths) + steps * WIDTH + BLOCK + PAGE
    local_pages = -(-max_pos // (PAGE * CP)) + 1
    ranks = [_Rank(r, g, layers, local_pages) for r in range(CP)]
    ring = build_ring(layers, g, W, WIDTH, BLOCK, PAGE, LAT, torch.float8_e4m3fn, "cuda")

    # The context, as the context server's transfer leaves it: every rank holds its owner pages.
    for slot, length in zip(slots, lengths):
        hist = (2.0 * torch.randn(length, layers, LAT, generator=gen, device="cuda")).bfloat16()
        pos = torch.arange(length, device="cuda")
        for r in ranks:
            owned = ((pos // PAGE) % CP) == r.rank
            pages = slot * local_pages + (pos // PAGE) // CP
            mask = torch.where(owned, 1, -1).to(torch.int8)
            store_context_latents(hist, r.pools, r.ptrs, pages, pos % PAGE, mask)
            r.ctx_len[slot] = length

    # First generation step: the window of each context into the ring, from every rank's share.
    # Slot 0 goes through ring_fill (rank 0's call, the all-gather played by concatenating every
    # rank's packed rows as they travel); the others through one ring_fill_batch.
    cap = fill_capacity(W, PAGE, CP)
    shares = [
        fill_pack(r.pools, r.table([0])[0], lengths[0], W, PAGE, CP, r.rank, cap) for r in ranks
    ]
    wires = [s[0].view(torch.uint8).view(torch.bfloat16) for s in shares]

    def all_gather(x, wires=wires):
        assert torch.equal(x, wires[0])  # rank 0's own share
        return torch.cat(wires)

    ring_fill(ring, 0, ranks[0].pools, ranks[0].table([0])[0].cpu(), lengths[0], PAGE, CP, 0,
              all_gather)  # fmt: skip
    _fill_batch(ring, ranks, [_meta(r, slots[1:], lengths[1:]) for r in ranks])
    for slot, length in zip(slots, lengths):
        first = max(0, length - W)
        pos = torch.arange(first, length, device="cuda")
        assert torch.equal(_ring_bytes(ring, slot, pos), _helix_bytes(ranks, slot, pos))
        # The worker's fill check: the ranks' owner checksums sum to every rank's ring checksum.
        sums = torch.stack(
            [
                window_checksums(ring, slot, r.pools, r.table([slot])[0], length, PAGE, CP, r.rank)
                for r in ranks
            ]
        )
        assert int(sums[:, 0].sum()) == int(sums[0, 1]) and bool((sums[:, 1] == sums[0, 1]).all())

    slot_t = torch.tensor(slots, device="cuda", dtype=torch.int32)
    tables = ranks[0].table(slots)
    counts = torch.full((g,), local_pages, dtype=torch.int32, device="cuda")
    seen_accepts = set()
    wrapped = False
    for step in range(steps):
        # Accepted drafts 0..7; the first step after the fill accepts every draft.
        drafts = [WIDTH - 1 if step == 0 else rng.randrange(WIDTH) for _ in slots]
        seen_accepts.update(drafts)
        nacc = torch.tensor([d + 1 for d in drafts], device="cuda", dtype=torch.int32)
        accepted = torch.randint(0, 1000, (g, WIDTH), device="cuda", dtype=torch.int32)
        latents = (
            2.0 * torch.randn(g * WIDTH, layers, LAT, generator=gen, device="cuda")
        ).bfloat16()
        committed = ranks[0].ctx_len[:g].clone()
        prep = None
        for r in ranks:  # today's helix store, on every rank's own pages
            seq_lens = torch.zeros(g, dtype=torch.int32, device="cuda")
            prep = dflash_paged_latent_prepare(
                nacc, accepted, slot_t, r.ctx_len, counts, tables, seq_lens, 0, g,
                1 << 30, PAGE, BLOCK, WIDTH, cp_size=CP, cp_rank=r.rank,
            )  # fmt: skip
            store_context_latents(latents.view(g * WIDTH, layers, LAT), r.pools, r.ptrs,
                                  prep.store_pages, prep.store_offsets, prep.store_mask)  # fmt: skip
        num_ctx = prep.num_ctx
        assert torch.equal(num_ctx, committed + nacc.long())
        # The ring: the same latents and store kernel at the step's targets, on every rank.
        rows = torch.arange(g, device="cuda").flip(0)  # batch row i holds slot g - 1 - i
        table = slot_t.flip(0).long()
        step_t = ring_step(ring, table, rows, prep.ctx_positions, num_ctx)
        pages, offs = ring_store_targets(ring, slot_t, committed, WIDTH)
        assert torch.equal(step_t.slots, slot_t.long())
        assert torch.equal(step_t.verify_pages, pages) and torch.equal(step_t.verify_offsets, offs)
        mask = ring_store_mask(nacc, WIDTH) if masked else None
        ring_store(ring, latents, step_t.verify_pages, step_t.verify_offsets, mask)
        # The block's own latents, as the block RoPE writes them before the block attends.
        block = (2.0 * torch.randn(g * BLOCK, layers, LAT, generator=gen, device="cuda")).bfloat16()
        pages, offs = ring_block_targets(ring, slot_t, num_ctx, BLOCK)
        assert torch.equal(step_t.block_pages, pages) and torch.equal(step_t.block_offsets, offs)
        store_context_latents(block, ring.pools, ring.ptrs, pages, offs,
                              torch.ones(g * BLOCK, dtype=torch.int8, device="cuda"))  # fmt: skip
        for i, slot in enumerate(slots):
            n = int(num_ctx[i])
            wrapped |= n >= ring.rows + W
            hist = torch.arange(max(0, n - W + 1), n, device="cuda")
            assert torch.equal(_ring_bytes(ring, slot, hist), _helix_bytes(ranks, slot, hist)), (
                step,
                slot,
            )
            own = torch.arange(n, n + BLOCK, device="cuda")
            want = (
                _fp8(block.view(g, BLOCK, layers, LAT)[i].float())
                .view(torch.uint8)
                .permute(1, 0, 2)
            )
            assert torch.equal(_ring_bytes(ring, slot, own), want), (step, slot)
    assert seen_accepts == set(range(WIDTH))
    assert wrapped


def test_window_checksum_sees_one_wrong_byte():
    """The fill check: a ring byte off by one at one position of the window changes the ring's sum."""
    ring = build_ring(1, 1, W, WIDTH, BLOCK, PAGE, LAT, torch.float8_e4m3fn, "cuda")
    r = _Rank(0, 1, 1, 2)
    length = 300
    pos = torch.arange(length, device="cuda")
    hist = torch.randint(0, 100, (1, length, LAT), dtype=torch.uint8, device="cuda")
    raw = ring.data.view(torch.uint8)
    pages, offs = ring_targets(ring, torch.tensor([0], device="cuda"), pos.view(1, -1))
    raw[:, pages, 0, 0, offs] = hist
    base = window_checksums(ring, 0, r.pools, r.table([0])[0], length, PAGE, CP, 0)[1]
    raw[0, pages[123], 0, 0, offs[123], 7] += 1
    assert window_checksums(ring, 0, r.pools, r.table([0])[0], length, PAGE, CP, 0)[1] != base


@pytest.mark.parametrize("layers", [1, 3])
def test_batched_fill_writes_the_per_request_fill_bytes(layers):
    """Requests of every window shape in shuffled slot order: one ``ring_fill_batch`` leaves every
    ring byte as per-request ``ring_fill`` calls do (the window rows written, every other byte as
    it was), and sends one all-gather of ``[G, layers, capacity, row]`` without a host sync."""
    rng = random.Random(layers)
    gen = torch.Generator(device="cuda").manual_seed(layers)
    lengths = [1, 63, 64, 65, 1000, W - 1, W, W + 1, 1088, 4133, 5000 + 3 * PAGE + 17]
    g = len(lengths)
    slots = list(range(g))
    rng.shuffle(slots)
    local_pages = -(-max(lengths) // (PAGE * CP)) + 1
    ranks = [_Rank(r, g, layers, local_pages) for r in range(CP)]
    for r in ranks:
        r.pools_data.view(torch.uint8).random_(0, 256, generator=gen)
    rings = [build_ring(layers, g, W, WIDTH, BLOCK, PAGE, LAT, torch.float8_e4m3fn, "cuda")
             for _ in range(2)]  # fmt: skip
    noise = torch.randint(0, 256, rings[0].data.shape, dtype=torch.uint8, device="cuda",
                          generator=gen)  # fmt: skip
    for ring in rings:
        ring.data.view(torch.uint8).copy_(noise)

    want = rings[0]
    cap = fill_capacity(W, PAGE, CP)
    for slot, length in zip(slots, lengths):
        shares = [
            fill_pack(r.pools, r.table([slot])[0], length, W, PAGE, CP, r.rank, cap)[0]
            for r in ranks
        ]
        wires = torch.cat([s.view(torch.uint8).view(torch.bfloat16) for s in shares])
        ring_fill(want, slot, ranks[0].pools, ranks[0].table([slot])[0].cpu(), length, PAGE, CP,
                  0, lambda x, wires=wires: wires)  # fmt: skip

    got = rings[1]
    metas = [_meta(r, slots, lengths) for r in ranks]  # the worker uploads its rows non-blocking
    _fill_batch(got, ranks, metas)  # compile outside the sync check
    got.data.view(torch.uint8).copy_(noise)
    with _no_host_sync():
        sent = _fill_batch(got, ranks, metas)
    torch.cuda.synchronize()
    assert len(sent) == 1 and sent[0].shape == (g, layers, cap, LAT // 2)
    assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8))
    changed = (want.data.view(torch.uint8) != noise).any(dim=-1).sum()
    assert int(changed) > 0  # the fill wrote something the noise did not already hold
