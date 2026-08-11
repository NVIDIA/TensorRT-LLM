# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A collection run must refuse the regimes that fit the scheduler, not the head.

Both regimes below produce shards that look perfectly well-formed. The run
completes, the row counts are plausible, the fitter converges, and the emitted
temperatures are wrong -- there is no downstream check that can tell. That is
why these are construction-time failures rather than warnings.

Measured on this branch: collecting while the planner trimmed gave
``ece_before = [.244 .319 .361 .389 .394]``; pinning the window to the full
block, same model and same fitter, gave ``[.157 .208 .270 .321 .344]``. The
difference is entirely the censored label.

SGLang guards the first regime next to its own collector
(``dspark_planner.py:100-108``) and names the second in the sibling
``ConfidenceMetricsProbe`` -- "padded verify rows corrupt the per-position
prefix label" (``dspark_observability.py:688-697``).
"""

import contextlib
import os
import tempfile

import pytest

from tensorrt_llm._torch.speculative.dspark_sts import (STS_COLLECT_ENV,
                                                        make_recorder_from_env)

_PIN = "TLLM_DSPARK_FORCE_VERIFY_LEN"


@contextlib.contextmanager
def collecting(pin=None):
    """Set the collection env for the block, restore it after.

    A context manager rather than a pytest fixture: these run inside the
    container, which ships no pytest, under a minimal shim that does not
    implement fixtures or tmp_path.
    """
    saved = {k: os.environ.get(k) for k in (STS_COLLECT_ENV, _PIN)}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[STS_COLLECT_ENV] = os.path.join(tmp, "collect")
            if pin is None:
                os.environ.pop(_PIN, None)
            else:
                os.environ[_PIN] = str(pin)
            yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_collection_off_builds_nothing():
    """The serving default must be untouched by any of this."""
    os.environ.pop(STS_COLLECT_ENV, None)
    assert make_recorder_from_env(block_size=5) is None
    assert make_recorder_from_env(block_size=5, has_cost_table=True,
                                  ragged_mode="compact") is None


def test_a_loaded_cost_table_without_a_pin_is_refused():
    """A table means the planner trims, and trimming censors the label."""
    with collecting():
        with pytest.raises(ValueError, match="censors the acceptance label"):
            make_recorder_from_env(block_size=5, has_cost_table=True)


def test_a_pinned_window_makes_the_table_harmless():
    """The uncensored regimes must all build a recorder.

    Pinning to the full block makes the label uncensored by construction;
    without a table the planner cannot trim, so nothing censors anything;
    static mode contributes no padding rows.
    """
    with collecting(pin=5):
        rec = make_recorder_from_env(block_size=5, has_cost_table=True)
        assert rec is not None and rec.block_size == 5
    with collecting():
        assert make_recorder_from_env(block_size=5,
                                      has_cost_table=False) is not None
        assert make_recorder_from_env(block_size=5,
                                      ragged_mode="static") is not None


def test_compact_mode_is_refused():
    """Padding rows contribute prefix labels that measure nothing.

    Refused even with the pin set: the pin fixes the *window*, while this is
    about rows that belong to no real request at all.
    """
    with collecting(pin=5):
        with pytest.raises(ValueError, match="Padded verify rows"):
            make_recorder_from_env(block_size=5, ragged_mode="compact")


class _Head:
    pass


class _Bare:
    """DSparkDraftModel layout: stages under .mtp_layers."""

    def __init__(self, head):
        stage = type("Stage", (), {})()
        if head is not None:
            stage.confidence_head = head
        self.mtp_layers = [object(), object(), stage]


class _Wrapper:
    """DSparkForCausalLM layout: bare model under .dspark_model.

    This is the layout the chained getattr missed: the wrapper has no
    mtp_layers of its own, so `getattr(model, "mtp_layers", [None])[-1]`
    resolved to None and silently disabled calibration for every
    confidence-scheduled run of the first campaign.
    """

    def __init__(self, head):
        self.dspark_model = _Bare(head)


def test_head_resolves_across_both_layouts_and_absence_is_none():
    from tensorrt_llm._torch.speculative.dspark_sts import resolve_confidence_head
    h = _Head()
    assert resolve_confidence_head(_Bare(h)) is h
    assert resolve_confidence_head(_Wrapper(h)) is h
    assert resolve_confidence_head(_Bare(None)) is None
    assert resolve_confidence_head(_Wrapper(None)) is None
    assert resolve_confidence_head(object()) is None
