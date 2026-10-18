# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``tp_agreement``: an install refusal every rank of the TP group raises alike.

Host-only, with a stand-in for ``Distributed`` (this rank's reason plus what the
peers send). The installers that use ``agreed_refusal`` (``attention_output_into_hop``,
``replicated_embedding``, ``lm_head_inbox``, ``drafter_hops``, ``unpool_into_boundary``,
``replicated_experts`` with ``persistent_experts`` / ``three_window_placement`` /
``bf16_routed_experts``) are tested in their own test files: a peer's refusal reaches this
rank before the installer's own collective, and nothing is built then. ``agreed_install``
(the shell's ``early_expert_ids``, ``full_attention_split_kv`` and ``next_qkv_prefetch``
installs) is tested here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch as tp_agreement,
)

pytestmark = pytest.mark.cpu_only

_TP8 = SimpleNamespace(tp_size=8)


class _Group:
    """Stands in for ``Distributed``: this rank (TP rank 0) first, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.calls = []

    def get(self, mapping):
        self.calls.append(mapping)
        return self

    def tp_allgather(self, obj, small_payload=False):
        assert small_payload
        return [obj] + self.peers


@pytest.fixture
def group(monkeypatch):
    def make(peers):
        stand_in = _Group(peers)
        monkeypatch.setattr(tp_agreement, "Distributed", stand_in)
        return stand_in

    return make


@pytest.mark.parametrize("mapping", [None, object(), "mapping", SimpleNamespace(tp_size=1)])
def test_a_group_of_one_returns_this_ranks_reason_without_a_collective(group, mapping):
    stand_in = group([])
    assert tp_agreement.agreed_refusal(mapping, None) is None
    assert tp_agreement.agreed_refusal(mapping, "why") == "why"
    assert stand_in.calls == []


def test_no_rank_refusing_is_no_refusal(group):
    stand_in = group([None] * 7)
    assert tp_agreement.agreed_refusal(_TP8, None) is None
    assert stand_in.calls == [_TP8]


def test_the_same_reason_on_every_rank_is_returned_as_it_is(group):
    group(["why"] * 7)
    assert tp_agreement.agreed_refusal(_TP8, "why") == "why"


def test_a_peers_reason_reaches_a_rank_that_would_install(group):
    group([None] * 6 + ["the peer's reason"])
    assert tp_agreement.agreed_refusal(_TP8, None) == "the peer's reason (on TP ranks [7] of 8)"


def test_this_ranks_reason_is_named_with_its_rank(group):
    group([None] * 7)
    assert tp_agreement.agreed_refusal(_TP8, "mine") == "mine (on TP ranks [0] of 8)"


def test_distinct_reasons_are_all_named(group):
    group(["b"] * 3 + [None] * 3 + ["a"])
    assert tp_agreement.agreed_refusal(_TP8, "b") == "a; b (on TP ranks [0, 1, 2, 3, 7] of 8)"


# --- agreed_install (installers with no collective of their own) ---------------------------------


def test_agreed_install_returns_the_result_when_no_rank_refuses(group):
    stand_in = group([None] * 7)
    assert tp_agreement.agreed_install(_TP8, lambda: 5) == 5
    assert stand_in.calls == [_TP8]


def test_agreed_install_raises_this_ranks_refusal_on_every_rank(group):
    group([None] * 7)

    def install():
        raise ValueError(
            "[musespark_open] next_qkv_prefetch: boundary_early_release converted no boundary -> QKV edge"
        )

    with pytest.raises(ValueError, match=r"QKV edge \(on TP ranks \[0\] of 8\)") as raised:
        tp_agreement.agreed_install(_TP8, install)
    assert isinstance(raised.value.__cause__, ValueError)


def test_agreed_install_raises_a_peers_refusal_on_a_rank_that_installed(group):
    group(
        [None] * 3
        + [
            "[musespark_open] full_attention_split_kv cannot replace the full-attention decode FMHA: x"
        ]
        + [None] * 3
    )
    installed = []
    with pytest.raises(ValueError, match=r"decode FMHA: x \(on TP ranks \[4\] of 8\)") as raised:
        tp_agreement.agreed_install(_TP8, lambda: installed.append(1))
    assert installed == [1] and raised.value.__cause__ is None


def test_agreed_install_agrees_the_listed_refusals_only(group):
    stand_in = group([None] * 7)

    def compile_fails():
        raise RuntimeError("compile failed")

    with pytest.raises(ValueError, match=r"compile failed \(on TP ranks \[0\] of 8\)"):
        tp_agreement.agreed_install(_TP8, compile_fails, (ValueError, RuntimeError))
    with pytest.raises(RuntimeError, match="compile failed"):
        tp_agreement.agreed_install(_TP8, compile_fails)
    assert len(stand_in.calls) == 1


def test_agreed_install_in_a_group_of_one_runs_no_collective(group):
    stand_in = group([])
    with pytest.raises(ValueError, match=r"^why$"):
        tp_agreement.agreed_install(None, lambda: (_ for _ in ()).throw(ValueError("why")))
    assert tp_agreement.agreed_install(SimpleNamespace(tp_size=1), lambda: "ok") == "ok"
    assert stand_in.calls == []
