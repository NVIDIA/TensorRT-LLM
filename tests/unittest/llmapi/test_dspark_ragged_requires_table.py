# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged verification must refuse to start without a profiled cost table.

Without one the planner's budget degenerates to verify-all -- correctly, since
a flat cost model makes every extra verify token look free -- so the ragged
path runs and changes nothing. As a runtime warning that produced 27-minute
runs which "succeeded" having done nothing, detectable only from a counter
(``steps_ragged: 0``) that nobody reads unless they already suspect the
problem. Config errors belong at construction.

``TLLM_DSPARK_FORCE_VERIFY_LENS`` is the deliberate exception: producing a cost
table requires a working ragged run, and that knob is what breaks the circle.
"""

import os
from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def _cfg(**kwargs):
    base = dict(max_draft_len=5,
                speculative_model="/nonexistent/model",
                enable_confidence_scheduling=True,
                enable_ragged_verify=True)
    base.update(kwargs)
    return DSparkDecodingConfig(**base)


@patch.dict(os.environ, {}, clear=False)
def test_ragged_without_a_cost_table_is_rejected():
    os.environ.pop("TLLM_DSPARK_FORCE_VERIFY_LENS", None)
    with pytest.raises(ValueError, match="requires a profiled cost table"):
        _cfg()


@patch.dict(os.environ, {}, clear=False)
def test_ragged_with_an_sps_table_is_accepted():
    os.environ.pop("TLLM_DSPARK_FORCE_VERIFY_LENS", None)
    cfg = _cfg(confidence_sps_table_path="/tmp/table.json")
    assert cfg.enable_ragged_verify


@patch.dict(os.environ, {"TLLM_DSPARK_FORCE_VERIFY_LENS": "1"}, clear=False)
def test_forced_windows_bootstrap_without_a_table():
    """The escape hatch must stay open, or the table can never be produced."""
    cfg = _cfg()
    assert cfg.enable_ragged_verify


@patch.dict(os.environ, {}, clear=False)
def test_uniform_confidence_scheduling_still_needs_no_table():
    """Only the ragged path is gated.

    Uniform tier selection degrades gracefully without a table -- it picks the
    max tier, which is the pre-existing behaviour -- so requiring one there
    would break working configurations.
    """
    os.environ.pop("TLLM_DSPARK_FORCE_VERIFY_LENS", None)
    cfg = DSparkDecodingConfig(max_draft_len=5,
                               speculative_model="/nonexistent/model",
                               enable_confidence_scheduling=True,
                               enable_ragged_verify=False)
    assert not cfg.enable_ragged_verify
