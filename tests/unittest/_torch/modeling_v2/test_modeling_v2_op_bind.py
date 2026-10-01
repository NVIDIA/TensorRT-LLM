# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three binding stages, and the one invariant that keeps certification intact.

A catalog entry now takes its arguments at three different times: engine
constants and op-owned fixtures at construction, per-forward state once a
step, and runtime tensors at the call. `raw_call` is the surface underneath
all of that -- it takes everything explicitly, which is what lets CELLS keep
certifying the whole input space no matter what a target chose to bind.

A per-layer weight table is also bound at construction, under `layered=`
rather than the flat `bound=` kwargs, and is projected onto one row by a
`layer=` keyword at the call -- still construction-time state, just indexed
per call instead of read back by name.
"""

from __future__ import annotations

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper


class _Spy(OpWrapper):
    """Records exactly what reached `raw_call`."""

    def raw_call(self, a=None, b=None, c=None):
        self.seen = dict(a=a, b=b, c=c)
        return self.seen

    def reference(self, a=None, b=None, c=None):
        return dict(a=a, b=b, c=c)


def test_an_unbound_entry_is_a_passthrough():
    """Nothing bound: the call reaches raw_call unchanged.

    This is what makes Task 1 behaviour-neutral -- every existing call site
    still works because an entry constructed with no arguments forwards
    verbatim.
    """
    assert _Spy()(a=1, b=2, c=3) == dict(a=1, b=2, c=3)


def test_construction_supplies_what_the_caller_no_longer_passes():
    op = _Spy(a=1)
    assert op(b=2, c=3) == dict(a=1, b=2, c=3)


def test_bind_step_supplies_the_per_forward_layer():
    op = _Spy(a=1)
    op.bind_step(b=2)
    assert op(c=3) == dict(a=1, b=2, c=3)


def test_a_later_stage_wins_over_an_earlier_one():
    """Call time beats step, step beats construction.

    Not a convenience: a decode target states `num_ctx_tokens=0` at
    construction while the step projection would report whatever the metadata
    carried, and the narrower statement has to survive.
    """
    op = _Spy(a="bound", b="bound")
    op.bind_step(b="step", c="step")
    assert op(c="call") == dict(a="bound", b="step", c="call")


def test_rebinding_a_step_replaces_the_previous_one():
    """Stale step state is the failure mode this design introduces: a forward
    that forgot to rebind would otherwise reuse the last one's block offsets.
    Replacing rather than merging is what keeps a rebind total.
    """
    op = _Spy()
    op.bind_step(a=1, b=1)
    op.bind_step(a=2)
    assert op() == dict(a=2, b=None, c=None)


def test_layer_selects_the_bound_row():
    """A `layered` table built at construction, read by row at the call.

    This is the mechanism gpt_oss's per-layer weight tables rest on: built
    once from the real weights, then indexed by `layer=i` every forward
    instead of being read out of a per-target tuple by the call site itself.
    """
    op = _Spy(a="bound", layered=dict(b=("row0", "row1", "row2")))
    assert op(c=3, layer=1) == dict(a="bound", b="row1", c=3)


def test_layer_overrides_bound_and_step_for_the_same_argument():
    """The layer projection is the last stage, and wins.

    A target that both binds a default and layers an override for the same
    argument name gets the layered row -- not the default, and not whatever a
    stale step happened to carry for it.
    """
    op = _Spy(b="bound-default", layered=dict(b=("row0", "row1")))
    op.bind_step(b="step-value")
    assert op(layer=1) == dict(a=None, b="row1", c=None)


def test_an_entry_with_no_layered_table_is_unaffected_by_the_absence_of_layer():
    """An entry that never binds `layered=` is not even aware the mechanism
    exists: with `_layered` empty, a call that passes no `layer=` runs the
    exact same merge it would have before `layered` existed. This is what
    keeps deepseek's 20-odd call sites -- none of which pass `layer=` -- and
    their own tests working untouched.
    """
    op = _Spy(a=1)
    op.bind_step(b=2)
    assert op(c=3) == dict(a=1, b=2, c=3)


def test_an_entry_must_implement_raw_call():
    class Incomplete(OpWrapper):
        def reference(self):
            return None

    with pytest.raises(TypeError, match="raw_call"):
        Incomplete()


def test_a_stale_step_binding_is_caught_under_validation():
    """The failure mode this design introduces, and the only place it is seen.

    Outside validation nothing compares generations -- an entry running on last
    forward's block offsets produces a wrong answer and no error. That is the
    accepted cost of keeping the hot path free; this test is the record that
    the check exists and fires.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.catalog import _op

    __extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook
    from _validating import validating

    op = _Spy()
    _op.advance_step_generation()
    op.bind_step(a=1)
    _op.advance_step_generation()  # a new forward that forgot to rebind
    with pytest.raises(AssertionError, match="previous step's metadata"):
        with validating(op):
            op()
