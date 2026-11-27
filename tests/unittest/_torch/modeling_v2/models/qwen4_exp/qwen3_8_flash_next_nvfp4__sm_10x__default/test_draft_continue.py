# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft-extend step's continuing rows past one rank (`_draft_continue`, `_draft_rows`).

Each request goes on past its K/V and index-key writes on row `r * p + num_accepted[r] - 1`
only, the row the spec worker continues from. Under attention DP the draft MoE's round trip
then pads by the group's request counts, so the decision must be the same on every rank: it is
read from the group's counts alone, and a rank whose own batch disagrees with them fails loudly
instead of issuing other collectives than its peers. No kernel runs: the continuing rows are
formed by the op's reference.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _reference_ops(monkeypatch):
    monkeypatch.setattr(M, "draft_continue_select", M.draft_continue_select.reference)
    monkeypatch.setattr(M, "draft_rows_gather", M.draft_rows_gather.reference)


def _md(seqs: int, p: int, accepted=None):
    rows = seqs * p
    if accepted is None:
        accepted = [1 + (r % p) for r in range(seqs)]
    return SimpleNamespace(
        qsa_mtp_num_accepted=torch.tensor(accepted, dtype=torch.int32),
        qsa_visible_blocks=torch.arange(100, 100 + rows, dtype=torch.int32),
        qsa_sequence_lengths=torch.arange(200, 200 + rows, dtype=torch.int32),
        qsa_logical_positions=torch.arange(300, 300 + rows, dtype=torch.int64),
        qsa_req_idx_per_token=torch.arange(rows, dtype=torch.int32) // p,
    )


def _core(dp_size: int = 1, dp_rank: int = 0, one_rank: bool = False):
    return SimpleNamespace(dp_size=dp_size, dp_rank=dp_rank, one_rank=one_rank)


def _spec(seqs, gens=None):
    return SimpleNamespace(all_rank_num_seqs=seqs, all_rank_num_gens=seqs if gens is None else gens)


def test_attention_tp_takes_each_requests_last_accepted_row():
    md = _md(5, 4, accepted=[1, 4, 2, 3, 9])  # 9: clamped to the request's last row
    cont = M._draft_continue(_core(), md, None, None, 20, 5, 4, False)
    assert cont.rows.tolist() == [0, 7, 9, 14, 19]
    assert cont.visible.tolist() == [100 + r for r in cont.rows.tolist()]
    assert cont.seq_lens.tolist() == [200 + r for r in cont.rows.tolist()]
    assert cont.logical.tolist() == [300 + r for r in cont.rows.tolist()]
    assert cont.req_idx.tolist() == [0, 1, 2, 3, 4]
    assert cont.counts is None


@pytest.mark.parametrize(
    "core, p, share, has_accepted",
    [
        (_core(one_rank=True), 4, False, True),  # one rank: `_mtp_layer` runs instead
        (_core(), 1, False, True),  # a draft decode step: one row a request already
        (_core(), 4, True, True),  # IndexShare captures every row's selection
        (_core(), 4, False, False),  # no accepted counts from the worker
    ],
)
def test_every_row_goes_on(core, p, share, has_accepted):
    md = _md(3, p)
    if not has_accepted:
        md.qsa_mtp_num_accepted = None
    assert M._draft_continue(core, md, None, None, 3 * p, 3, p, share) is None


def test_attention_tp_needs_p_rows_a_request():
    assert M._draft_continue(_core(), _md(3, 4), None, None, 13, 3, 4, False) is None


def test_attention_dp_takes_the_groups_counts():
    md = _md(3, 4)
    cont = M._draft_continue(
        _core(dp_size=4, dp_rank=1), md, _spec([2, 3, 3, 1]), [8, 12, 12, 4], 12, 3, 4, False
    )
    assert cont is not None and cont.counts == [2, 3, 3, 1]


@pytest.mark.parametrize(
    "spec, tokens",
    [
        (None, [12, 12]),  # no group counts from the worker
        (_spec([3, 3], gens=[3, 2]), [12, 12]),  # a peer holds a context request
        (_spec([3, 3]), [12, 13]),  # a peer's rows are not 4 a request
        (_spec([3, 3]), None),  # no row counts for this draft step
    ],
)
def test_attention_dp_declines_on_any_rank_of_the_group(spec, tokens):
    md = _md(3, 4)
    assert M._draft_continue(_core(dp_size=2), md, spec, tokens, 12, 3, 4, False) is None


def test_attention_dp_rank_disagreeing_with_the_group_fails():
    """Silently keeping every row here would run other collectives than the peers'."""
    with pytest.raises(AssertionError, match="group's counts"):
        M._draft_continue(_core(dp_size=2), _md(3, 4), _spec([2, 3]), [8, 12], 12, 3, 4, False)


def test_draft_rows_gathers_each_operand():
    cont = torch.tensor([3, 4, 9], dtype=torch.int64)
    x = torch.arange(40.0).reshape(10, 4)
    q_index = torch.arange(10 * 2 * 3.0).reshape(10, 2, 3)
    gate = torch.arange(10 * 2 * 8.0).reshape(10, 2, 8)[:, :, :4]  # a strided 3-D view
    out = M._draft_rows(cont, {"x": x, "q_index": q_index, "gate": gate})
    assert torch.equal(out["x"], x[cont])
    assert out["q_index"].shape == (3, 2, 3) and torch.equal(out["q_index"], q_index[cont])
    assert torch.equal(out["gate"], gate[cont])
