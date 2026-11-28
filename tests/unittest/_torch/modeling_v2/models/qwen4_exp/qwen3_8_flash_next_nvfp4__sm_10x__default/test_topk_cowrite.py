# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation cluster top-k's coalesced write (`DP_TOPK_NARROW_ENV` at dep4 / dep8,
`TEP_TOPK_NARROW_ENV` at tep4; opt-in).

* `_selection` passes `cowrite=True` to `qsa_topk_expand_cluster` on a generation step's rows
  only when the core settled it, and never on a context chunk.
* `_topk_cowrite_build`: off unset or at `0`; on at its own layout on SM103 with `1` or the
  measured `16w` / `w`; the other layout's switch, another GPU or layout warns and stays off; a
  spelling outside them raises; a failing or raising build check declines this rank alone.
* `_topk_cowrite_check` names a differing output.

The ops are recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


def _core(**kw):
    base = dict(
        dp_size=4,
        dp_rank=1,
        attn_tp_size=1,
        one_rank=False,
        compress_ratio=4,
        token_topk=2048,
        block_topk=512,
        max_verify_rows=4,
        dp_topk_cluster_rows=37,
        mapping=SimpleNamespace(tp_rank=1, tp_size=4, world_size=4),
        local_experts=128,
        num_experts=512,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 3, 8]),
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("cowrite", [False, True])
@pytest.mark.parametrize("context", [False, True])
def test_selection_passes_cowrite_on_generation_rows(monkeypatch, cowrite, context):
    seen = []
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores, max_rows: True)
    monkeypatch.setattr(M, "qsa_topk_expand_cluster", lambda *a, **k: seen.append(k))
    core = _core(topk_cowrite=cowrite)
    out = torch.zeros(20, 1)
    M._selection(core, torch.zeros(20, 8), None, None, None, None, None, out, context=context)
    want = dict(out=out)
    if cowrite and not context:
        want["cowrite"] = True
    assert seen == [want]


def _clear(monkeypatch):
    for env in (M.DP_TOPK_NARROW_ENV, M.TEP_TOPK_NARROW_ENV):
        monkeypatch.delenv(env, raising=False)


@pytest.fixture
def sm103(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))


@pytest.mark.parametrize("value", ["1", "16w", "w", "on"])
def test_on_at_a_dep_rank(monkeypatch, sm103, value):
    _clear(monkeypatch)
    monkeypatch.setenv(M.DP_TOPK_NARROW_ENV, value)
    monkeypatch.setattr(M, "_topk_cowrite_check", lambda core, device: "")
    assert M._topk_cowrite_build(_core(), torch.device("cuda")) is True


def test_on_at_a_tep4_rank(monkeypatch, sm103):
    _clear(monkeypatch)
    monkeypatch.setenv(M.TEP_TOPK_NARROW_ENV, "16w")
    monkeypatch.setattr(M, "_topk_cowrite_check", lambda core, device: "")
    tep = _core(dp_size=1, attn_tp_size=4)
    assert M._topk_cowrite_build(tep, torch.device("cuda")) is True


@pytest.mark.parametrize(
    "env, value, kw, capability",
    [
        (None, None, {}, (10, 3)),  # unset: off
        ("DP_TOPK_NARROW_ENV", "0", {}, (10, 3)),
        ("TEP_TOPK_NARROW_ENV", "1", {}, (10, 3)),  # the tep4 switch at a dep rank
        ("DP_TOPK_NARROW_ENV", "1", dict(dp_size=1, attn_tp_size=4), (10, 3)),  # dep's at tep4
        ("DP_TOPK_NARROW_ENV", "1", {}, (10, 0)),  # another GPU
        ("DP_TOPK_NARROW_ENV", "1", dict(dp_size=2), (10, 3)),  # dep2: not measured
        ("TEP_TOPK_NARROW_ENV", "1", dict(dp_size=1, attn_tp_size=8), (10, 3)),  # tep8
    ],
)
def test_off_where_it_cannot_apply(monkeypatch, env, value, kw, capability):
    _clear(monkeypatch)
    if env is not None:
        monkeypatch.setenv(getattr(M, env), value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    def check(core, device):
        raise AssertionError("no check where the switch does not apply")

    monkeypatch.setattr(M, "_topk_cowrite_check", check)
    assert M._topk_cowrite_build(_core(**kw), torch.device("cuda")) is False


def test_a_bad_value_raises(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(M.DP_TOPK_NARROW_ENV, "32w")
    with pytest.raises(ValueError):
        M._topk_cowrite_build(_core(), torch.device("cuda"))


@pytest.mark.parametrize("failure", ["differs", "raises"])
def test_a_failed_check_declines_this_rank(monkeypatch, sm103, failure):
    """No collective: the rank keeps the scattered write, which gives the same bits."""
    _clear(monkeypatch)
    monkeypatch.setenv(M.DP_TOPK_NARROW_ENV, "1")

    def check(core, device):
        if failure == "raises":
            raise RuntimeError("launch failed")
        return "the token selection differs at 12 rows"

    monkeypatch.setattr(M, "_topk_cowrite_check", check)
    assert M._topk_cowrite_build(_core(), torch.device("cuda")) is False


_GENERATOR = torch.Generator


class _Gen:
    def __init__(self, device=None):
        self.g = _GENERATOR()

    def manual_seed(self, seed):
        self.g.manual_seed(seed)
        return self.g


@pytest.mark.parametrize("broken", [None, "topk", "sel"])
def test_check_compares_both_outputs(monkeypatch, broken):
    """The check runs both forms at one row and every row count of this rank the cluster serves
    (the trunk's 4 x 1 / 3 / 8 and a draft step's 1 / 3 / 8 / 16 rows here; 64 is past the
    cluster's 37), on the dtypes `_selection` passes, and names the output that differs."""
    monkeypatch.setattr(torch, "Generator", _Gen)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(M, "_TOPK_COWRITE_CHECK_WIDTH", 4100)
    calls = []

    def op(scores, ends, pos, lens, topk, cr, token_topk, cowrite=False, **place):
        rows = scores.shape[0]
        calls.append((rows, cowrite, place))
        assert (ends.dtype, pos.dtype, lens.dtype) == (torch.int32, torch.int64, torch.int32)
        topk.fill_(1 if cowrite and broken == "topk" else 0)
        return torch.full((rows, 3), 1 if cowrite and broken == "sel" else 0)

    monkeypatch.setattr(M, "qsa_topk_expand_cluster", op)
    core = _core(model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 3, 8, 16]))
    why = M._topk_cowrite_check(core, torch.device("cpu"))
    if broken is None:
        assert why == ""
        assert [(r, c) for r, c, _ in calls] == [
            (r, c) for r in (1, 3, 4, 8, 12, 16, 32) for c in (False, True)
        ]
    else:
        assert ("topk_indices" if broken == "topk" else "token selection") in why


def test_check_covers_every_placement(monkeypatch):
    """Past 37 rows (`DP_TOPK_CLUSTER_ROWS_ENV`) every row count runs under the trunk's placement,
    and a draft step's row count also under the draft's where it differs: each launch plan the
    rank runs is checked."""
    monkeypatch.setattr(torch, "Generator", _Gen)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(M, "_TOPK_COWRITE_CHECK_WIDTH", 4100)

    def placement(scores, draft):
        rows = scores.shape[0]
        if rows <= 37:
            return {}
        if draft:
            return dict(solo_blocks=32768, pairs=0)
        return dict(solo_blocks=8192, pairs=74 if rows > 76 else 0)

    monkeypatch.setattr(M, "_topk_placement", placement)
    calls = []

    def op(scores, ends, pos, lens, topk, cr, token_topk, cowrite=False, **place):
        calls.append((scores.shape[0], cowrite, tuple(sorted(place.items()))))
        topk.fill_(0)
        return torch.zeros((scores.shape[0], 3))

    monkeypatch.setattr(M, "qsa_topk_expand_cluster", op)
    core = _core(
        dp_size=8,
        dp_topk_cluster_rows=256,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[2, 12, 20, 40]),
    )
    assert M._topk_cowrite_check(core, torch.device("cpu")) == ""
    plans = {(r, p) for r, c, p in calls if c}
    assert {(r, p) for r, c, p in calls if not c} == plans
    trunk_solo = (("pairs", 0), ("solo_blocks", 8192))
    trunk_map = (("pairs", 74), ("solo_blocks", 8192))
    draft_solo = (("pairs", 0), ("solo_blocks", 32768))
    assert plans == {
        (1, ()),
        (2, ()),
        (8, ()),
        (12, ()),
        (20, ()),
        (40, trunk_solo),
        (40, draft_solo),
        (48, trunk_solo),
        (80, trunk_map),
        (160, trunk_map),
    }


def test_decode_forms_names_it(monkeypatch):
    st = M._RouteFront(None)
    tep = SimpleNamespace(
        dp_size=1,
        mapping=SimpleNamespace(tp_rank=0, tp_size=4),
        _route_front=st,
        _tep_finar=None,
        tep_fast=True,
        tep_draft_layer=False,
        topk_cowrite=True,
    )
    assert "topk cowrite True" in M._decode_forms(tep)
