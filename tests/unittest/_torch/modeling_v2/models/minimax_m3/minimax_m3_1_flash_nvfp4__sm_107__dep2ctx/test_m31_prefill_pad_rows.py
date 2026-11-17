# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family module ops/prefill_pad_rows (CPU): the dep2ctx bucket grid and the prefill pad-row mask.

- The grid keeps every bucket above 4096 on one side of every dep2ctx configuration edge, and the edges match the
  target's thresholds (M2 pin bands, C1, the MegaMoE launch ladder).
- ``PrefillPadRows`` writes ``mask[:rows] = arange(rows) >= live`` with one launch on padded / graphed steps only,
  hands the mask to a launch of exactly the forward's rows on padded or breakable-capture calls (None otherwise, a
  chunked launch and a foreign capture included), and keeps one buffer whose captured views see later writes.
- ``wire`` connects only MoE blocks whose backend takes a mask.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import prefill_pad_rows as ppr

pytestmark = pytest.mark.cpu_only


def _state(capacity=16, enabled=True, in_graph=True):
    state = ppr.PrefillPadRows(capacity, enabled=enabled, in_graph=lambda: in_graph)
    state.allocate("cpu")
    return state


@pytest.fixture
def capturing(monkeypatch):
    state = types.SimpleNamespace(on=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.on)
    return state


# ----------------------------------------------------------------------------------------------------------- grid


# ------------------------------------------------------------------------------------------------------ the mask


@pytest.mark.parametrize(("live", "rows"), [(0, 8), (3, 8), (8, 8), (5, 16)])
def test_begin_writes_the_live_prefix_mask(live, rows):
    state = _state()
    state.begin(live, rows, padded=True, graphed=False)
    want = (torch.arange(rows) >= live).to(torch.uint8)
    assert torch.equal(state._mask[:rows], want)
    assert state.writes == 1


def test_a_replay_writes_whatever_its_live_count():
    state = _state()
    state._mask.fill_(1)
    state.begin(8, 8, padded=False, graphed=True)  # live == bucket replays a graph too
    assert state.writes == 1 and not state._mask[:8].any()


def test_captured_views_read_later_writes():
    state = _state()
    state.begin(16, 16, padded=False, graphed=False)
    captured = state.mask_for(16, capturing=True)
    ptr = state._mask.data_ptr()
    for live in (3, 16, 0, 9):
        state.begin(live, 16, padded=False, graphed=True)
        assert torch.equal(captured, (torch.arange(16) >= live).to(torch.uint8))
        state.allocate("cpu")  # idempotent: the buffer never moves
        assert state._mask.data_ptr() == ptr


def test_light_and_idle_ranks_mask_their_pad_rows():
    state = ppr.PrefillPadRows(16384)
    state.allocate("cpu")
    state.begin(4000, 16384, padded=True, graphed=True)
    assert int(state._mask[:16384].sum()) == 12384 and not state._mask[:4000].any()
    state.begin(0, 13312, padded=True, graphed=True)
    assert bool(state._mask[:13312].all())


# ------------------------------------------------------------------------------------------------------ wiring
