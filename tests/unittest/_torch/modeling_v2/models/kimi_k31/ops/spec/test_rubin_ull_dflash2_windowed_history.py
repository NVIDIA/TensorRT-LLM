# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the ULL helix16 drafter attends bitwise the same from a windowed history as from a
whole one.

A generation server's cache keeps the drafter's window of its history (``kimi_k31_draft_window``,
the checkpoint's 1024 positions): of a request's 16 CP ranks' owner pages it keeps the ledger
blocks of the window and the next write, and the context server sends only the window's rows.
So a windowed rank's page table has ``BAD_PAGE_INDEX`` (-1) for every released block, and every
row before the window is whatever the page held before (another request's latents). Here each
such row and the page at index -1 hold FP8 NaN bytes, so any read of one shows up.

The drafter numbers its history by position either way (``test_rubin_ull_draft_window_history.py``),
and on the same inputs, against a whole history of the same rows:

* headtp (the helix16 targets' layout): the first generation step's window fill
  (``ring_fill_batch``: every rank's pack, the all-gather, the scatter) leaves every ring byte
  equal, and the drafter's attention over the ring (``rubin_ull_dflash2_headtp_decode``, the block
  at the prompt's end) returns bitwise the same output on every layer;
* helix (the window over each rank's own pages, ``rubin_ull_dflash2_helix_window``): every rank's
  partial, stats and empty mask are bitwise the same (Blackwell and Rubin: the CuTe DSL decode).

The attention over the history is the only part of the drafter that reads it, so the drafter's
logits and proposed tokens are the same.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_dflash2_headtp_ring import (  # noqa: E501
    fill_pack as fill_pack_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_headtp_decode import (  # noqa: E501
    headtp_decode_workspace,
    rubin_ull_dflash2_headtp_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_headtp_ring import (  # noqa: E501
    build_ring,
    ring_fill_batch,
    ring_targets,
)

CP, PAGE, W, WIDTH, BLOCK, LAT = 16, 64, 1024, 8, 8, 576
HEADS_HEADTP, HEADS_HELIX = 6, 96
SCALE = 0.13086080
_FP8_NAN = 0x7F
_LENGTHS = [5, 700, 1023, 1024, 1025, 1088, 4133, 70001]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def _fp8_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def _released(block: int, length: int) -> bool:
    """Whether a windowed cache released ledger block ``block`` (``PAGE * CP`` positions; local
    page ``block`` of every rank) at history ``length``: it lies wholly before the retention, the
    window and the next write."""
    return (block + 1) * PAGE * CP <= max(0, length - (W + 1))


class _Rank:
    """One CP rank's owner pages of one request's history, behind one guard page.

    ``pools[l]`` is layer ``l``'s page-shaped view ``[pages, 1, 1, PAGE, LAT]`` from local page 0;
    the page before it (index -1 of a page table) holds NaN bytes."""

    def __init__(self, rank: int, local_pages: int, layers: int):
        self.rank = rank
        self.data = torch.full(
            (layers, 1 + local_pages, 1, 1, PAGE, LAT), _FP8_NAN, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        self.pools = [self.data[i, 1:] for i in range(layers)]
        self.ptrs = torch.tensor(
            [p.data_ptr() for p in self.pools], dtype=torch.int64, device="cuda"
        )
        self.table = torch.arange(local_pages, dtype=torch.int32, device="cuda")

    def store(self, history: torch.Tensor, positions: torch.Tensor) -> None:
        """``history`` ``[n, layers, LAT]`` uint8 rows at its global ``positions`` (all owned)."""
        local = 1 + (positions // PAGE) // CP
        self.data.view(torch.uint8)[:, local, 0, 0, positions % PAGE] = history.permute(1, 0, 2)


def _ranks(history: torch.Tensor, layers: int, windowed: bool) -> list:
    """The 16 ranks' owner pages of ``history`` ``[L, layers, LAT]`` uint8: all rows (a whole
    history), or (windowed) only the rows of ``[L - W, L)``, with the released blocks at -1 in
    every rank's table."""
    length = history.shape[0]
    local_pages = -(-(length + BLOCK + PAGE) // (PAGE * CP)) + 1
    first = max(0, length - W) if windowed else 0
    positions = torch.arange(first, length, device="cuda")
    ranks = []
    for rank in range(CP):
        r = _Rank(rank, local_pages, layers)
        owned = positions[(positions // PAGE) % CP == rank]
        if owned.numel():
            r.store(history[owned], owned)
        if windowed:
            released = [k for k in range(local_pages) if _released(k, length)]
            r.table[released] = -1
        ranks.append(r)
    return ranks


def _meta(rank: _Rank, length: int) -> torch.Tensor:
    """``ring_fill_batch``'s row for one request in slot 0: slot, length, the local page table."""
    head = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    return torch.cat([head, rank.table]).view(1, -1).contiguous()


def _fill(ring, ranks: list, length: int) -> None:
    """Rank 0's ``ring_fill_batch``; the all-gather concatenates every rank's packed share."""
    metas = [_meta(r, length) for r in ranks]

    def all_gather(x):
        wires = []
        for r, meta in zip(ranks, metas):
            wire = torch.empty_like(x.view(torch.uint8))
            fill_pack_kernel(meta, r.ptrs, r.pools[0].stride(0), r.pools[0].stride(3), wire,
                             r.rank, ring.window, PAGE, CP)  # fmt: skip
            wires.append(wire.view(torch.bfloat16))
        return torch.cat(wires)

    r0 = ranks[0]
    ring_fill_batch(ring, metas[0], r0.pools, r0.ptrs, PAGE, CP, 0, all_gather)


@pytest.mark.parametrize("length", _LENGTHS)
@torch.inference_mode()
def test_headtp_fill_and_attention_match_the_whole_history(length):
    layers = 2
    gen = torch.Generator(device="cuda").manual_seed(length)
    history = _fp8_bytes(2.0 * torch.randn(length, layers, LAT, generator=gen, device="cuda"))
    rings = []
    for windowed in (False, True):
        ring = build_ring(layers, 1, W, WIDTH, BLOCK, PAGE, LAT, torch.float8_e4m3fn, "cuda")
        _fill(ring, _ranks(history, layers, windowed), length)
        rings.append(ring)
    whole, windowed = rings
    assert torch.equal(whole.data.view(torch.uint8), windowed.data.view(torch.uint8))
    assert not bool((whole.data.view(torch.uint8) == _FP8_NAN).any())

    # The block at the prompt's end: its own latents in the ring, then the drafter's attention
    # over [L - W + 1, L + j] for its row j.
    block = _fp8_bytes(2.0 * torch.randn(layers, BLOCK, LAT, generator=gen, device="cuda"))
    own = torch.arange(length, length + BLOCK, device="cuda")
    slot0 = torch.zeros(1, dtype=torch.int64, device="cuda")
    pages, offsets = ring_targets(whole, slot0, own.view(1, -1))
    for ring in rings:
        ring.data.view(torch.uint8)[:, pages, 0, 0, offsets] = block
    query = torch.randn(BLOCK, HEADS_HEADTP, LAT, generator=gen, device="cuda").bfloat16()
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    starts = torch.tensor([length], dtype=torch.int64, device="cuda")
    ws = headtp_decode_workspace(1, BLOCK, HEADS_HEADTP, W, query.device)
    for layer in range(layers):
        outs = [
            rubin_ull_dflash2_headtp_decode(
                query, ring.data[layer].view(-1, LAT), slots, starts, ring.rows, SCALE, 1.0, ws
            ).clone()
            for ring in rings
        ]
        assert torch.equal(outs[0], outs[1]), layer
        assert bool(torch.isfinite(outs[1].float()).all()), layer


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the CuTe DSL MLA decode (Blackwell, Rubin)",
)
@pytest.mark.parametrize("length", _LENGTHS)
@torch.inference_mode()
def test_helix_window_matches_the_whole_history(length):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_helix_window import (  # noqa: E501
        helix_window_workspace,
        rubin_ull_dflash2_helix_window,
    )

    gen = torch.Generator(device="cuda").manual_seed(length + 1)
    history = _fp8_bytes(2.0 * torch.randn(length, 1, LAT, generator=gen, device="cuda"))
    block_kv = _fp8_bytes(2.0 * torch.randn(BLOCK, 1, LAT, generator=gen, device="cuda"))
    block_kv = block_kv.view(torch.float8_e4m3fn)
    query = torch.randn(BLOCK, HEADS_HELIX, LAT, generator=gen, device="cuda").bfloat16()
    lengths = torch.tensor([length], dtype=torch.int64, device="cuda")
    table_rows = torch.zeros(1, dtype=torch.int64, device="cuda")
    ws = helix_window_workspace(BLOCK, HEADS_HELIX, PAGE, W, CP, query.device)
    results = []
    for windowed in (False, True):
        per_rank = []
        for r in _ranks(history, 1, windowed):
            pool = r.pools[0].view(-1, PAGE, LAT)
            part, stats, empty = rubin_ull_dflash2_helix_window(
                query, pool, r.table.view(1, -1), table_rows, lengths, block_kv, BLOCK, SCALE,
                CP, r.rank, ws,
            )  # fmt: skip
            per_rank.append((part.clone(), stats.clone(), empty.clone()))
        results.append(per_rank)
    for rank, (whole, windowed) in enumerate(zip(*results)):
        for got, want in zip(windowed, whole):
            assert torch.equal(got, want), rank
        part, _, empty = windowed
        assert bool(torch.isfinite(part[~empty].float()).all()), rank
