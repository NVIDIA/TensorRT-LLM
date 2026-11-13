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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    prefill_pad_rows as ppr,
)

pytestmark = pytest.mark.cpu_only

BUCKETS = list(range(256, 4097, 256))  # the prefill bucket list (prefill_capture_num_tokens)


def _state(capacity=16, enabled=True, in_graph=True):
    state = ppr.PrefillPadRows(capacity, enabled=enabled, in_graph=lambda: in_graph)
    state.allocate("cpu")
    return state


class _Backend:
    def __init__(self, decline=None):
        self.masks = []
        self.decline = decline

    def set_pad_mask(self, mask):
        self.masks.append(mask)

    def pad_mask_decline_reason(self):
        return self.decline


def _layer(backend=None, dense=False, max_rows=0):
    if dense:
        return types.SimpleNamespace(block_sparse_moe=None)
    experts = types.SimpleNamespace(backend=backend, moe_max_num_tokens=max_rows)
    return types.SimpleNamespace(block_sparse_moe=types.SimpleNamespace(experts=experts))


@pytest.fixture
def capturing(monkeypatch):
    state = types.SimpleNamespace(on=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.on)
    return state


# ----------------------------------------------------------------------------------------------------------- grid


def test_the_grid_straddles_no_edge_above_4096():
    assert (
        ppr.band_straddles(
            BUCKETS + list(ppr.DEP2CTX_BCG_GRID), ppr.DEP2CTX_CONFIG_EDGES, floor=4096
        )
        == []
    )
    assert ppr.DEP2CTX_BCG_GRID[-1] == 16384 and min(ppr.DEP2CTX_BCG_GRID) > 4096
    assert {12287, 12288} <= set(ppr.DEP2CTX_BCG_GRID)
    gaps = [b - a for a, b in zip((4096,) + ppr.DEP2CTX_BCG_GRID, ppr.DEP2CTX_BCG_GRID)]
    assert max(gaps) <= 1024


def test_band_straddles_finds_the_known_and_the_coarse_straddles():
    assert ppr.band_straddles(BUCKETS, ppr.DEP2CTX_CONFIG_EDGES) == [
        (768, 1024, 1024),
        (3840, 4096, 4096),
    ]
    assert (8192, 12288, 12288) in ppr.band_straddles(
        [4096, 8192, 12288, 16384], ppr.DEP2CTX_CONFIG_EDGES
    )
    # Dropping 12287 lets 11265..12287 replay the 12288 band.
    no_edge = [b for b in ppr.DEP2CTX_BCG_GRID if b != 12287]
    assert ppr.band_straddles(BUCKETS + no_edge, ppr.DEP2CTX_CONFIG_EDGES, floor=4096) == [
        (11264, 12288, 12288)
    ]


def test_the_edges_match_the_dep2ctx_thresholds():
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
            modeling,
        )
    except NoTarget as exc:
        pytest.skip(str(exc))

    starts = {lo for bands in modeling._MXFP8_CUBLASLT_PINS.values() for lo, _, _ in bands}
    assert starts <= set(ppr.DEP2CTX_CONFIG_EDGES)
    assert modeling._C1_MIN_ROWS in ppr.DEP2CTX_CONFIG_EDGES
    # Every row floor below the MegaMoE 4096 rung sits at or below the bucket list's top bucket.
    assert max(modeling._CTX_QKV_FUSED_MIN_ROWS, modeling._NORMQ_MIN_ROWS) <= 4096


def test_the_unpinned_edges_are_the_documented_straddles():
    """The M-keyed tactics dep2ctx does not pin are straddled (replay vs eager at the live rows is not guaranteed
    bitwise there); the straddled live ranges are the O2 oracle's off-bucket cells."""
    import ast
    from pathlib import Path

    torch_dir = next(p for p in Path(ppr.__file__).resolve().parents if p.name == "_torch")
    src = (torch_dir / "custom_ops" / "torch_custom_ops.py").read_text()
    runner = next(
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ClassDef) and node.name == "MXFP8GemmRunner"
    )
    bands = next(
        ast.literal_eval(node.value)
        for node in runner.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_LARGE_M_BANDS" for t in node.targets)
    )
    in_grid = {lo for lo, hi in bands if 4096 < lo <= ppr.DEP2CTX_BCG_GRID[-1]}
    assert in_grid == {6553, 13106} and in_grid <= set(ppr.DEP2CTX_UNPINNED_EDGES)
    # Each band ends on a grid bucket, so only its start is straddled.
    assert {hi for lo, hi in bands if lo in in_grid} <= set(ppr.DEP2CTX_BCG_GRID)
    assert {8192, 16384} <= set(ppr.DEP2CTX_UNPINNED_EDGES)  # last_positive_power_of_2 steps
    assert ppr.band_straddles(
        BUCKETS + list(ppr.DEP2CTX_BCG_GRID), ppr.DEP2CTX_UNPINNED_EDGES, floor=4096
    ) == [(6144, 7168, 6553), (7168, 8192, 8192), (12288, 13312, 13106), (15360, 16384, 16384)]


def test_every_new_bucket_launches_the_full_megamoe_rung(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mmc

    monkeypatch.setattr(mmc, "get_sm_version", lambda: 107)
    rungs = mmc.MegaMoECuteDsl._resolve_maxt_buckets(16384)
    assert rungs[-2:] == [4096, 16384]
    assert 4097 in ppr.DEP2CTX_CONFIG_EDGES
    for bucket in ppr.DEP2CTX_BCG_GRID:
        assert min(b for b in rungs if b >= bucket) == 16384


# ------------------------------------------------------------------------------------------------------ the mask


@pytest.mark.parametrize(("live", "rows"), [(0, 8), (3, 8), (8, 8), (5, 16)])
def test_begin_writes_the_live_prefix_mask(live, rows):
    state = _state()
    state.begin(live, rows, padded=True, graphed=False)
    want = (torch.arange(rows) >= live).to(torch.uint8)
    assert torch.equal(state._mask[:rows], want)
    assert state.writes == 1


def test_an_unpadded_eager_step_launches_nothing_and_hands_none(capturing):
    state = _state()
    state._mask.fill_(7)
    state.begin(3, 8, padded=False, graphed=False)
    assert state.writes == 0 and int(state._mask[0]) == 7
    assert state.mask_for(8, capturing=False) is None


def test_a_replay_writes_whatever_its_live_count():
    state = _state()
    state._mask.fill_(1)
    state.begin(8, 8, padded=False, graphed=True)  # live == bucket replays a graph too
    assert state.writes == 1 and not state._mask[:8].any()


def test_hand_off_rule(capturing):
    state = _state()
    state.begin(5, 8, padded=True, graphed=False)
    assert torch.equal(state.mask_for(8, capturing=False), (torch.arange(8) >= 5).to(torch.uint8))
    assert state.mask_for(7, capturing=False) is None  # not this forward's rows
    state.end()
    assert state.mask_for(8, capturing=False) is None
    # A capture (live == bucket) arms the rows without a write and hands the persistent view.
    state.begin(8, 8, padded=False, graphed=False)
    view = state.mask_for(8, capturing=True)
    assert (
        view.data_ptr() == state._mask.data_ptr() and view.dtype == torch.uint8 and view.dim() == 1
    )


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


def test_out_of_range_rows_and_a_disabled_state_are_inert():
    state = _state(capacity=8)
    state.begin(1, 9, padded=True, graphed=True)
    state.begin(0, 0, padded=True, graphed=True)
    assert state.writes == 0 and state.mask_for(9, capturing=True) is None
    off = ppr.PrefillPadRows(8, enabled=False)
    off.allocate("cpu")
    off.begin(1, 8, padded=True, graphed=True)
    assert off._mask is None and off.mask_for(8, capturing=True) is None


def test_begin_allocates_lazily_outside_capture_only(capturing):
    state = ppr.PrefillPadRows(8)
    capturing.on = True
    state.begin(2, 8, padded=True, graphed=False, device="cpu")
    assert not state.allocated
    capturing.on = False
    state.begin(2, 8, padded=True, graphed=False, device="cpu")
    assert state.allocated and state.writes == 1


def test_hand_over_sets_the_mask_or_clears_it(capturing):
    state, backend = _state(), _Backend()
    state.begin(5, 8, padded=True, graphed=False)
    state.hand_over(backend, 0, 8)
    state.hand_over(backend, 0, 6)
    state.hand_over(backend, 4, 8)  # a chunked launch (4-row chunks) takes no whole-step mask
    assert backend.masks[0] is not None and backend.masks[1:] == [None, None]


def test_only_a_breakable_capture_takes_the_mask(capturing):
    backend = _Backend()
    capturing.on = True
    foreign = _state(in_graph=False)
    foreign.begin(8, 8, padded=False, graphed=False)
    foreign.hand_over(backend, 0, 8)
    bcg = _state(in_graph=True)
    bcg.begin(8, 8, padded=False, graphed=False)
    bcg.hand_over(backend, 0, 8)
    assert backend.masks[0] is None and backend.masks[1] is not None
    # A padded step under another capture (a piecewise prefill graph) takes none either: no writer is recorded.
    padded = _state(in_graph=False)
    padded.begin(5, 8, padded=True, graphed=False)
    padded.hand_over(backend, 0, 8)
    assert backend.masks[2] is None


def test_hand_over_does_nothing_while_compiling(monkeypatch):
    state, backend = _state(), _Backend()
    state.begin(5, 8, padded=True, graphed=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    state.hand_over(backend, 0, 8)
    assert backend.masks == []


# ------------------------------------------------------------------------------------------------------ wiring


def test_wire_connects_only_blocks_that_take_a_mask(capturing):
    good, decline = _Backend(), _Backend(decline="the top-k weights apply in FC1")
    layers = [
        _layer(dense=True),
        _layer(good, max_rows=16384),
        _layer(decline),
        _layer(object()),
        _layer(good),
    ]
    state = _state()
    wired, left_out = ppr.wire(layers, state)
    assert (wired, left_out) == (2, "the top-k weights apply in FC1")
    assert state.layers == 2
    assert not hasattr(layers[2].block_sparse_moe, "pad_rows_hand_over")
    state.begin(5, 8, padded=True, graphed=False)
    layers[1].block_sparse_moe.pad_rows_hand_over(8)
    assert good.masks[-1] is not None
    # Idempotent, and a disabled state unwires.
    assert ppr.wire(layers, state)[0] == 2
    off = ppr.PrefillPadRows(8, enabled=False)
    assert ppr.wire(layers, off) == (0, "the top-k weights apply in FC1")
    assert layers[1].block_sparse_moe.pad_rows_hand_over is None


@pytest.mark.skip(
    reason="needs the MegaMoE-local backend's pad-mask seat (set_pad_mask / _take_pad_mask), which this engine's "
    "MegaMoE backend does not have"
)
def test_a_megamoe_local_leaf_takes_the_handed_view(capturing):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
        TrtllmCutedslMegaMoeLocalNvfp4Impl as Local,
    )

    leaf = Local.__new__(Local)
    leaf.apply_topk_in_fc1 = False
    leaf._pad_mask = None
    state = _state()
    state.begin(5, 8, padded=True, graphed=False)
    state.hand_over(leaf, 0, 8)
    taken = leaf._take_pad_mask(8)
    assert torch.equal(taken, (torch.arange(8) >= 5).to(torch.uint8))
    state.hand_over(leaf, 0, 8)
    with pytest.raises(ValueError, match="the pad mask has 8 rows, the launch 7"):
        leaf._take_pad_mask(7)
    state.hand_over(leaf, 0, 6)  # other rows: the pending mask is cleared
    assert leaf._take_pad_mask(6) is None
