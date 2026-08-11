# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged verification must refuse to start without a profiled cost table.

Without one the planner's budget degenerates to verify-all -- correctly, since
a flat cost model makes every extra verify token look free -- so the ragged
path runs and changes nothing. As a runtime warning that produced 27-minute
runs which "succeeded" having done nothing, detectable only from a counter
(``steps_ragged: 0``) that nobody reads unless they already suspect the
problem. Config errors belong at construction.

There is no exception: a crafted (non-flat) cost table is how a test makes the
planner trim, which exercises the real confidence-driven path rather than
bypassing it. See
tests/unittest/_torch/speculative/hw_agnostic/test_dspark_confidence_schedule.py.
"""

import pytest

from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def _cfg(**kwargs):
    base = dict(max_draft_len=5,
                speculative_model="/nonexistent/model",
                enable_confidence_scheduling=True,
                enable_ragged_verify=True)
    base.update(kwargs)
    return DSparkDecodingConfig(**base)


def test_ragged_without_a_cost_table_is_rejected():
    with pytest.raises(ValueError, match="requires a profiled cost table"):
        _cfg()
    # The same gate must open once a table is named -- a gate that rejects
    # everything would pass the raise above.
    assert _cfg(confidence_sps_table_path="/tmp/table.json").enable_ragged_verify


def test_confidence_scheduling_without_ragged_is_rejected():
    """There is no uniform tier ladder to fall back to.

    Scheduling used to have a middle state: pick one verify length for the
    whole batch from the captured tiers. It could not act on per-request
    confidence -- which is the entire point of a confidence head -- and it
    degenerated to the full block whenever acceptance was high, i.e. on every
    workload measured on this checkpoint. It is gone, so this combination now
    names a path that does not exist and must not silently mean something else.

    The two remaining states are: schedule per request (ragged), or verify the
    whole drafted block (confidence scheduling off).
    """
    with pytest.raises(ValueError, match="enable_ragged_verify"):
        DSparkDecodingConfig(max_draft_len=5,
                             speculative_model="/nonexistent/model",
                             enable_confidence_scheduling=True,
                             enable_ragged_verify=False)
